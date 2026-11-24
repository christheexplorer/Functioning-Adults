"""Synthesised, deliberately un-cartoony sound effects (no samples needed)."""
import numpy as np

rng = np.random.default_rng(42)


def _env(n, a, d, sr):
    e = np.ones(n)
    na = max(1, int(a * sr))
    e[:na] = np.linspace(0, 1, na)
    e *= np.exp(-np.arange(n) / (d * sr))
    return e


def _lp(x, alpha):
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def _bp(x, lo_a, hi_a):
    return _lp(x, hi_a) - _lp(x, lo_a)


def room_tone(dur, sr):
    """Late-night store: faint fluorescent hum + cooler air."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    hum = 0.0035 * np.sin(2 * np.pi * 120 * t) + 0.0015 * np.sin(2 * np.pi * 240 * t + 0.4)
    air = rng.normal(0, 1, n)
    k = 48
    air = np.convolve(air, np.ones(k) / k, mode="same") * 0.022
    out = (hum + air).astype(np.float32)
    f = int(0.02 * sr)
    out[:f] *= np.linspace(0, 1, f)
    return out


def footstep(sr, k=0):
    n = int(0.14 * sr)
    x = rng.normal(0, 1, n)
    x = _lp(x, 0.05) * _env(n, 0.003, 0.025, sr)
    thump = np.sin(2 * np.pi * (70 + 8 * k) * np.arange(n) / sr) * _env(n, 0.002, 0.03, sr)
    return (x * 1.2 + thump * 0.6).astype(np.float32)


def paper_bag(sr):
    n = int(0.45 * sr)
    x = rng.normal(0, 1, n)
    crinkle = _bp(x, 0.08, 0.6) * _env(n, 0.005, 0.12, sr)
    crinkle *= (rng.random(n) > 0.55) * 0.8 + 0.2
    thud = np.sin(2 * np.pi * 95 * np.arange(n) / sr) * _env(n, 0.002, 0.04, sr)
    return (crinkle * 0.8 + thud * 0.7).astype(np.float32)


def pos_beep(sr, f=2350, dur=0.07):
    n = int(dur * sr)
    t = np.arange(n) / sr
    e = np.ones(n)
    r = int(0.004 * sr)
    e[:r] = np.linspace(0, 1, r)
    e[-r:] = np.linspace(1, 0, r)
    tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    return (tone * e * 0.6).astype(np.float32)


def swivel(sr):
    n = int(0.42 * sr)
    x = rng.normal(0, 1, n)
    x = _bp(x, 0.02, 0.12)
    grain = (np.sin(2 * np.pi * np.cumsum(np.linspace(18, 32, n)) / sr) > 0.6).astype(float)
    e = np.sin(np.linspace(0, np.pi, n))
    return (x * (0.4 + grain) * e * 0.8).astype(np.float32)


def drawer(sr):
    """Cash drawer release: latch clack, roll, stop thunk, coins settling."""
    n = int(0.55 * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    clack = rng.normal(0, 1, int(0.03 * sr)) * _env(int(0.03 * sr), 0.0005, 0.006, sr)
    out[: len(clack)] += clack * 0.9
    m = int(0.18 * sr)
    roll = _lp(rng.normal(0, 1, m), 0.12) * np.linspace(0.6, 1.0, m) * 0.7
    out[int(0.01 * sr): int(0.01 * sr) + m] += roll
    s = int(0.19 * sr)
    k = n - s
    thunk = (np.sin(2 * np.pi * 85 * t[:k]) * 0.9 + _lp(rng.normal(0, 1, k), 0.2) * 0.8) * _env(k, 0.001, 0.05, sr)
    out[s:] += thunk
    for j in range(5):
        c = s + int(rng.uniform(0.01, 0.2) * sr)
        L = int(0.025 * sr)
        if c + L < n:
            out[c:c + L] += np.sin(2 * np.pi * rng.uniform(3200, 5200) * t[:L]) * _env(L, 0.0005, 0.006, sr) * 0.18
    return out.astype(np.float32)


def chip_insert(sr):
    n = int(0.22 * sr)
    x = rng.normal(0, 1, n)
    slide = _bp(x, 0.05, 0.4) * np.linspace(0.2, 0.6, n) * (np.arange(n) < int(0.16 * sr))
    click = np.zeros(n)
    c = int(0.16 * sr)
    L = n - c
    click[c:] = rng.normal(0, 1, L) * _env(L, 0.0005, 0.005, sr)
    return (slide * 0.5 + click * 1.0).astype(np.float32)


def decline(sr):
    """Terminal error: three short low beeps."""
    out = []
    for i in range(3):
        b = pos_beep(sr, f=880, dur=0.11)
        out.append(b * 1.6)
        out.append(np.zeros(int(0.07 * sr), np.float32))
    return np.concatenate(out).astype(np.float32)
