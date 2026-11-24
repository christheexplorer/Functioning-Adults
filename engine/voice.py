"""Dialogue recording via Kokoro TTS (local ONNX model)."""
import os, json, numpy as np, soundfile as sf

MODEL_DIR = os.environ.get("FA_MODEL_DIR", os.path.expanduser("~/fa-tools"))
SR = 24000

# Series voice casting. Keep stable across episodes.
CAST = {
    "GREG":    dict(voice="am_puck",    speed=1.00),
    "ROBBER":  dict(voice="am_fenrir",  speed=0.90),
    "DENNIS":  dict(voice="am_michael", speed=0.88),
    "RICHARD": dict(voice="bm_george",  speed=0.95),
    "LINDA":   dict(voice="af_nicole",  speed=0.95),
}

_k = None
def _kokoro():
    global _k
    if _k is None:
        from kokoro_onnx import Kokoro
        _k = Kokoro(os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin"))
    return _k

def trim(x, thresh=0.01, pad=0.03):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * SR)); b = min(len(x), idx[-1] + int(pad * SR))
    return x[a:b]

def record(lines, outdir, seed_overrides=None):
    """lines: list of (id, speaker, text[, speed]). Writes wavs + durations.json, cached."""
    os.makedirs(outdir, exist_ok=True)
    meta_path = os.path.join(outdir, "durations.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for ln in lines:
        lid, spk, text = ln[:3]
        cfg = dict(CAST[spk])
        if len(ln) > 3 and ln[3]:
            cfg["speed"] = ln[3]
        key = f"{spk}|{cfg['voice']}|{cfg['speed']}|{text}"
        path = os.path.join(outdir, f"{lid}.wav")
        if meta.get(lid, {}).get("key") == key and os.path.exists(path):
            continue
        s, sr = _kokoro().create(text, voice=cfg["voice"], speed=cfg["speed"], lang="en-us")
        s = trim(np.asarray(s, dtype=np.float32))
        sf.write(path, s, SR)
        meta[lid] = dict(key=key, dur=len(s) / SR, speaker=spk, text=text)
    json.dump(meta, open(meta_path, "w"), indent=1)
    return meta
