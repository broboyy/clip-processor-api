import os
import json
import uuid
import subprocess
import threading
import time
from flask import Flask, request, jsonify
import yt_dlp
import whisper
import cv2
import mediapipe as mp

app = Flask(__name__)

TEMP_DIR = "/tmp/clips"
os.makedirs(TEMP_DIR, exist_ok=True)

# ─── HELPER: Download YouTube video ───────────────────────────────────────────
def download_youtube(url, output_path):
    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ─── HELPER: Get video duration ───────────────────────────────────────────────
def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


# ─── HELPER: Transcribe with Whisper ──────────────────────────────────────────
def transcribe_video(video_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path, language="id", task="transcribe")
    return result


# ─── HELPER: Detect face position for crop center ─────────────────────────────
def detect_face_center(video_path, timestamp_sec):
    mp_face = mp.solutions.face_detection
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 0.5  # default center

    h, w = frame.shape[:2]
    with mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            face_cx = bbox.xmin + bbox.width / 2
            return max(0.1, min(0.9, face_cx))
    return 0.5


# ─── HELPER: Crop video to 9:16 with face tracking ───────────────────────────
def crop_to_916(input_path, output_path, start, end, face_x_ratio=0.5):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", input_path],
        capture_output=True, text=True
    )
    w, h = map(int, probe.stdout.strip().split(","))

    target_w = int(h * 9 / 16)
    if target_w > w:
        target_w = w

    cx = int(face_x_ratio * w)
    x_offset = cx - target_w // 2
    x_offset = max(0, min(x_offset, w - target_w))

    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vf", f"crop={target_w}:{h}:{x_offset}:0,scale=1080:1920",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, capture_output=True)


# ─── HELPER: Generate SRT subtitle ────────────────────────────────────────────
def generate_srt(segments, start_offset, end_offset, srt_path):
    def fmt(t):
        t = max(0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    idx = 1
    for seg in segments:
        s_start = seg["start"]
        s_end = seg["end"]
        if s_end < start_offset or s_start > end_offset:
            continue
        adj_start = max(0, s_start - start_offset)
        adj_end = min(end_offset - start_offset, s_end - start_offset)
        text = seg["text"].strip()
        if not text:
            continue
        lines.append(f"{idx}\n{fmt(adj_start)} --> {fmt(adj_end)}\n{text}\n")
        idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── HELPER: Burn subtitle into video ─────────────────────────────────────────
def burn_subtitle(input_path, srt_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            f"subtitles={srt_path}:force_style='"
            "FontName=Arial,FontSize=14,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            "Outline=2,Shadow=1,Alignment=2,MarginV=40'"
        ),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, capture_output=True)


# ─── MAIN ENDPOINT ────────────────────────────────────────────────────────────
@app.route("/process", methods=["POST"])
def process_video():
    data = request.get_json()
    youtube_url = data.get("url")
    num_clips = int(data.get("num_clips", 3))
    timestamps = data.get("timestamps", [])  # dari Gemini: [{start, end, reason}]

    if not youtube_url:
        return jsonify({"error": "url required"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # 1. Download video
        raw_path = os.path.join(job_dir, "source.mp4")
        download_youtube(youtube_url, raw_path)

        # 2. Transcribe
        transcript = transcribe_video(raw_path)
        segments = transcript.get("segments", [])

        # 3. Jika timestamps tidak dikirim, bagi rata sebagai fallback
        if not timestamps:
            duration = get_duration(raw_path)
            clip_duration = min(60, duration / num_clips)
            timestamps = [
                {"start": i * clip_duration, "end": (i + 1) * clip_duration, "reason": "auto"}
                for i in range(num_clips)
            ]

        timestamps = timestamps[:num_clips]

        # 4. Proses setiap clip
        clip_paths = []
        for i, ts in enumerate(timestamps):
            start = float(ts["start"])
            end = float(ts["end"])
            mid = (start + end) / 2

            # Deteksi wajah
            face_x = detect_face_center(raw_path, mid)

            # Crop 9:16
            cropped = os.path.join(job_dir, f"clip_{i}_cropped.mp4")
            crop_to_916(raw_path, cropped, start, end, face_x)

            # Buat subtitle
            srt_path = os.path.join(job_dir, f"clip_{i}.srt")
            generate_srt(segments, start, end, srt_path)

            # Burn subtitle
            final_path = os.path.join(job_dir, f"clip_{i}_final.mp4")
            burn_subtitle(cropped, srt_path, final_path)

            clip_paths.append({
                "index": i + 1,
                "path": final_path,
                "start": start,
                "end": end,
                "reason": ts.get("reason", "")
            })

        # 5. Return paths
        return jsonify({
            "job_id": job_id,
            "clips": clip_paths,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/file/<job_id>/<filename>", methods=["GET"])
def get_file(job_id, filename):
    from flask import send_file
    path = os.path.join(TEMP_DIR, job_id, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="video/mp4")
    return jsonify({"error": "file not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
