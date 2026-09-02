# Kid Studio ACE-Step 1.5 RunPod Worker

RunPod Serverless music and children’s-song generator for Kid Studio.

## Cost-aware models

- `cheap`: ACE-Step 1.5 2B turbo, 8 steps
- `balanced`: ACE-Step 1.5 2B turbo, 12 steps
- `high`: ACE-Step 1.5 XL 4B turbo, loaded only when explicitly requested

The Director supplies the planned caption and lyrics directly. The optional ACE 5Hz language model is deliberately disabled to reduce VRAM, downloads and GPU time.

## Deployment

- Branch: `main`
- Dockerfile: `/Dockerfile`
- Type: Queue
- GPU: 24 GB recommended
- Minimum workers: 0
- Maximum workers: 1 initially
- Network volume: `/runpod-volume`
- Suggested execution timeout: 1800 seconds

The image pins a verified ACE-Step upstream commit. Model checkpoints download on first warmup to `/runpod-volume/ace-step/checkpoints` and persist for future workers.

## Input

The generate operation accepts `caption` (or `prompt`), `lyrics`, `instrumental`, `vocal_language`, `duration_seconds` (10–240), optional BPM/key/time signature, quality and seed.

Output is MP3 with model, quality, seed, inference time and SHA-256 metadata. RunPod adds queue and total execution times for actual GPU-cost accounting.
