# Demucs Audio Separator API

REST API for separating vocals and accompaniment from audio files using Facebook Research's [Demucs](https://github.com/facebookresearch/demucs).

## Endpoints

### `GET /health`
Health check.

### `POST /separate` (Standalone)
Upload an audio file. Returns download links for **vocals** and **no_vocals** (accompaniment).

```bash
curl -X POST http://localhost:8686/separate \
  -F "file=@song.mp3"
```

Response:
```json
{
  "job_id": "uuid",
  "vocals": "/download/{job_id}/vocals.wav",
  "no_vocals": "/download/{job_id}/no_vocals.wav"
}
```

### `POST /separate/stream` (Streaming)
Upload an audio file and stream back a single stem directly.

```bash
# Get vocals
curl -X POST "http://localhost:8686/separate/stream?stem=vocals" \
  -F "file=@song.mp3" -o vocals.wav

# Get accompaniment (no vocals)
curl -X POST "http://localhost:8686/separate/stream?stem=no_vocals" \
  -F "file=@song.mp3" -o no_vocals.wav
```

### `GET /download/{job_id}/{filename}`
Download a separated file from a previous `/separate` job.

### `GET /jobs/{job_id}`
List available files for a job.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Deploy via Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8686
```
