# FUNCTIONING ADULTS

Vertical (1080×1920) deadpan stick-figure shorts, rendered entirely in code.

```
engine/ink.py     hand-drawn ink: tapered wobbly strokes, 12fps line boil, misregistered fills, paper texture
engine/cast.py    Greg + Masked Robber, built to the model sheets in the repo root (Dennis/Linda/Richard to come)
engine/voice.py   series voice casting (Kokoro TTS, local)
episodes/ep001/   script.py (dialogue), episode.py (shots, timing, poses, mix), sfx.py
```

Setup (once):
```
pip install skia-python kokoro-onnx soundfile pillow numpy   # skia needs libegl1 libgl1
mkdir -p ~/fa-tools && cd ~/fa-tools
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Render: `python3 episodes/ep001/episode.py` → `build/ep001/*.mp4` + `.srt`
Contact sheet only: `python3 episodes/ep001/episode.py --stills`

Comedy timing lives in `START` / `SHOTS` at the top of each episode file.
