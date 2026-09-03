# Kid Studio Seed-VC singing worker

RunPod Serverless worker for consented, character-linked singing voice conversion.
It uses Seed-VC's dedicated 44.1 kHz F0-conditioned singing model and the actively
maintained `adefossez/demucs` fork. Upstreams are pinned by commit.

Deploy this directory's `Dockerfile` as a separate Queue endpoint. Use a 24 GB GPU
first and a 48 GB fallback, zero active workers, and one GPU per worker.

The `generate` request requires `guide_audio`, `voice_clone_consent: true`, and
performers containing an exact `character_asset_name`, consented `reference_audio`,
and explicit `start_seconds`/`end_seconds` segments. Conversion preserves the guide
melody and timing; the worker places every converted part back on the original
timeline and mixes it with the separated instrumental stem.
