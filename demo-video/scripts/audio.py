# Synthesizes the music bed and SFX kit (deterministic, no downloads). 44.1 kHz 16-bit mono WAV.
import numpy as np, wave
SR = 44100
rng = np.random.default_rng(42)
def save(name, x, gain=0.9):
    x = x / (np.max(np.abs(x)) + 1e-9) * gain
    with wave.open(name, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
def t_(d): return np.arange(int(SR * d)) / SR
def env(n, a=0.005, r=0.2):
    e = np.ones(n); na = max(1, int(a * SR)); nr = max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na); e[-nr:] = np.minimum(e[-nr:], np.linspace(1, 0, nr)); return e
def lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR); y = np.zeros_like(x); s = 0.0
    for i, v in enumerate(x): s = (1 - a) * v + a * s; y[i] = s
    return y

# ---- SFX
t = t_(0.7); n = rng.standard_normal(len(t)); sweep = np.sin(2*np.pi*np.cumsum(300 + 3200*(t/0.7)**2)/SR)
w = lp(n, 1800) * np.sin(np.pi * np.clip(t/0.7, 0, 1))**1.5 * 0.7 + sweep * 0.12 * np.sin(np.pi*np.clip(t/0.7,0,1))
save("public/sfx/whoosh.wav", w, 0.7)
t = t_(0.18); save("public/sfx/pop.wav", np.sin(2*np.pi*np.cumsum(900*np.exp(-t*30)+220)/SR) * np.exp(-t*22), 0.8)
t = t_(0.05); save("public/sfx/tick.wav", np.sin(2*np.pi*2400*t) * np.exp(-t*90) + 0.3*rng.standard_normal(len(t))*np.exp(-t*140), 0.5)
t = t_(0.9); save("public/sfx/hit.wav", np.sin(2*np.pi*np.cumsum(120*np.exp(-t*6)+42)/SR) * np.exp(-t*4.2) + 0.25*lp(rng.standard_normal(len(t)),900)*np.exp(-t*18), 0.95)
t = t_(1.6); r = lp(rng.standard_normal(len(t)), 3000)*(t/1.6)**2.2 + 0.2*np.sin(2*np.pi*np.cumsum(200+1800*(t/1.6)**2.5)/SR)*(t/1.6)**2
save("public/sfx/riser.wav", r * env(len(t), 0.01, 0.05), 0.6)
t = t_(2.4); ch = sum(np.sin(2*np.pi*f*t)*a for f, a in [(880,1),(1318.5,.6),(1760,.45),(2637,.2)]) * np.exp(-t*2.2) * env(len(t),0.003,0.4)
save("public/sfx/chime.wav", ch, 0.7)

# ---- Music: A minor, 92 BPM, 64 s. soft pad + sub + sparse pluck arpeggio + gentle pulse
BPM = 92; beat = 60 / BPM; DUR = 64.0
N = int(SR * DUR); m = np.zeros(N)
chords = [(57,60,64,67),(53,57,60,64),(48,52,55,60),(55,59,62,67)]   # Am7 Fmaj7 Cmaj7 G (midi)
f = lambda midi: 440 * 2 ** ((midi - 69) / 12)
bar = beat * 4
nbars = int(DUR / bar) + 1
def add(x, start):
    i = int(start * SR); j = min(N, i + len(x));
    if i < N: m[i:j] += x[:j - i]
for b in range(nbars):
    ch = chords[b % 4]; s = b * bar
    tt = t_(bar + 1.2)
    pad = sum(np.sin(2*np.pi*f(n_)*tt + 0.6*np.sin(2*np.pi*0.3*tt)) + 0.5*np.sin(2*np.pi*f(n_)*1.004*tt) for n_ in ch)
    pad = pad * env(len(tt), 0.9, 1.1) * 0.05
    add(pad, s)
    sub = np.sin(2*np.pi*f(ch[0]-24)*tt) * env(len(tt), 0.05, 0.6) * 0.22
    add(sub, s)
    # arpeggio plucks: 8th notes, enter after bar 2
    if b >= 2:
        for k in range(8):
            note = ch[[0,2,1,3,2,3,1,2][k]] + 12
            tp = t_(0.55); pl = np.sin(2*np.pi*f(note)*tp) * np.exp(-tp*7) * (0.07 if b < 10 else 0.09)
            add(pl, s + k * beat / 2)
    # soft kick on beats after bar 4
    if b >= 4:
        for k in range(4):
            tk = t_(0.35); kk = np.sin(2*np.pi*np.cumsum(95*np.exp(-tk*22)+46)/SR)*np.exp(-tk*9)*0.20
            add(kk, s + k * beat)
        for k in range(8):   # light hats
            th = t_(0.04); add(rng.standard_normal(len(th))*np.exp(-th*110)*0.012, s + k*beat/2 + beat/4)
fade = np.ones(N); fade[:int(SR*1.5)] = np.linspace(0,1,int(SR*1.5)); fade[-int(SR*4):] = np.linspace(1,0,int(SR*4))
save("public/audio/music.wav", m * fade, 0.8)
print("ok")
