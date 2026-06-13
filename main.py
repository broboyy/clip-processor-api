import os
import uuid
import subprocess
import requests
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
TEMP_DIR = "/tmp/clips"
os.makedirs(TEMP_DIR, exist_ok=True)


# ── Download YouTube video ─────────────────────────────────────────────────────
def download_youtube(url, output_path):
    subprocess.run([
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--quiet",
        url
    ], check=True)


# ── Get video info via ffprobe ─────────────────────────────────────────────────
def get_video_info(path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "csv=p=0", path
    ], capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    w = int(parts[0]) if len(parts) > 0 else 1280
    h = int(parts[1]) if len(parts) > 1 else 720
    dur = float(parts[2]) if len(parts) > 2 else 60
    return w, h, dur


# ── Crop video to 9:16 center ──────────────────────────────────────────────────
def crop_to_916(input_path, output_path, start, end):
    w, h, _ = get_video_info(input_path)
    target_w = int(h * 9 / 16)
    if target_w > w:
        target_w = w
    x_offset = (w - target_w) // 2
    duration = end - start

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vf", f"crop={target_w}:{h}:{x_offset}:0,scale=1080:1920",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ], capture_output=True, check=True)


# ── Transcribe via Gemini API ──────────────────────────────────────────────────
def transcribe_with_gemini(video_path, gemini_key):
    # Upload file ke Gemini Files API
    with open(video_path, "rb") as f:
        video_data = f.read()

    # Step 1: Upload
    upload_resp = requests.post(
        f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={gemini_key}",
        headers={"Content-Type": "video/mp4"},
        data=video_data,
        timeout=120
    )
    if upload_resp.status_code != 200:
        return []

    file_uri = upload_resp.json().get("file", {}).get("uri", "")
    if not file_uri:
        return []

    # Step 2: Transcribe
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
        json={
            "contents": [{
                "parts": [
                    {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
                    {"text": "Transkripsi video ini ke bahasa Indonesia. Balas HANYA JSON array berikut tanpa markdown:\n[{\"start\":0.0,\"end\":3.0,\"text\":\"teks subtitle\"}]\nTimestamp dalam detik, setiap segment max 7 kata."}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4000}
        },
        timeout=120
    )

    if resp.status_code != 200:
        return []

    raw = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return eval(raw) if raw.startswith("[") else []
    except:
        return []


# ── Generate SRT subtitle ──────────────────────────────────────────────────────
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
        s_start = float(seg.get("start", 0))
        s_end = float(seg.get("end", 0))
        if s_end < start_offset or s_start > end_offset:
            continue
        adj_start = max(0, s_start - start_offset)
        adj_end = min(end_offset - start_offset, s_end - start_offset)
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{idx}\n{fmt(adj_start)} --> {fmt(adj_end)}\n{text}\n")
        idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return idx > 1


# ── Burn subtitle ──────────────────────────────────────────────────────────────
def burn_subtitle(input_path, srt_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            f"subtitles={srt_path}:force_style='"
            "FontName=Arial,FontSize=14,PrimaryColour=&HFFFFFF,"
            "OutlineColour=&H000000,Outline=2,Shadow=1,Alignment=2,MarginV=40'"
        ),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ], capture_output=True)


# ── MAIN ENDPOINT ──────────────────────────────────────────────────────────────
@app.route("/process", methods=["POST"])
def process_video():
    data = request.get_json()
    youtube_url = data.get("url")
    num_clips = int(data.get("num_clips", 3))
    timestamps = data.get("timestamps", [])
    gemini_key = data.get("gemini_key", "")

    if not youtube_url:
        return jsonify({"error": "url required"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        # 1. Download
        raw_path = os.path.join(job_dir, "source.mp4")
        download_youtube(youtube_url, raw_path)

        # 2. Fallback timestamps jika tidak ada
        if not timestamps:
            _, _, duration = get_video_info(raw_path)
            clip_dur = min(60, duration / max(num_clips, 1))
            timestamps = [
                {"start": i * clip_dur, "end": (i + 1) * clip_dur, "reason": f"Segment {i+1}"}
                for i in range(num_clips)
            ]

        timestamps = timestamps[:num_clips]

        # 3. Transcribe via Gemini (jika ada key)
        segments = []
        if gemini_key:
            segments = transcribe_with_gemini(raw_path, gemini_key)

        # 4. Proses setiap clip
        clip_paths = []
        for i, ts in enumerate(timestamps):
            start = float(ts["start"])
            end = float(ts["end"])

            # Crop 9:16
            cropped = os.path.join(job_dir, f"clip_{i}_cropped.mp4")
            crop_to_916(raw_path, cropped, start, end)

            final_path = os.path.join(job_dir, f"clip_{i}_final.mp4")

            # Subtitle jika ada transcript
            if segments:
                srt_path = os.path.join(job_dir, f"clip_{i}.srt")
                has_subs = generate_srt(segments, start, end, srt_path)
                if has_subs:
                    burn_subtitle(cropped, srt_path, final_path)
                else:
                    os.rename(cropped, final_path)
            else:
                os.rename(cropped, final_path)

            clip_paths.append({
                "index": i + 1,
                "path": final_path,
                "start": start,
                "end": end,
                "reason": ts.get("reason", "")
            })

        return jsonify({"job_id": job_id, "clips": clip_paths, "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/file/<job_id>/<filename>")
def get_file(job_id, filename):
    path = os.path.join(TEMP_DIR, job_id, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="video/mp4")
    return jsonify({"error": "file not found"}), 404


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
