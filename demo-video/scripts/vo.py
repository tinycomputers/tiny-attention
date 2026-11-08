# Narration: each sentence is synthesized on its own (clean intonation resets), trimmed, joined with natural
# pauses, then cleaned up (high-pass, presence EQ, gentle compression, loudness target). Writes durations to src/vo.json.
import io, json, os, subprocess, wave
import numpy as np
from piper import PiperVoice, SynthesisConfig

SCRIPT = {
    "hook": ["Attention is the engine inside every modern AI model.",
             "It also has a memory problem."],
    "intro": ["This is TinyAttention.",
              "A real FlashAttention engine, written in SystemVerilog.",
              "And it runs live, right here in your browser."],
    "tiles": ["Instead of storing the whole score matrix, it works one small tile at a time.",
              "An online softmax keeps only a running max and a running sum."],
    "engine": ["Two eight-bit systolic arrays do the math.",
               "Nothing on screen is faked.",
               "Every value you see comes straight from the hardware design."],
    "memory": ["Both schedules compute attention on the same inputs.",
               "But flash moves nearly four times less memory on this run.",
               "And the gap keeps growing as the sequence gets longer."],
    "trust": ["Every result is checked bit-exact against a golden model.",
              "Step through it clock by clock, and watch each tile resolve."],
    "cta": ["Open the engine.",
            "Pick a preset, press run, and watch attention unfold, one tile at a time."],
}
PAUSE = 0.3  # seconds between sentences inside one scene

VOICE = os.environ.get("VOICE", "en_US-ryan-high")
voice = PiperVoice.load(f"voices/{VOICE}.onnx")
sr = voice.config.sample_rate
cfg = SynthesisConfig(length_scale=1.0, noise_scale=0.5, noise_w_scale=0.6)

def synth(text):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=cfg)
    buf.seek(0)
    with wave.open(buf, "rb") as w:
        assert w.getframerate() == sr
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

def trim(x, th=0.01):
    idx = np.flatnonzero(np.abs(x) > th)
    if idx.size == 0:
        return x
    a = max(0, idx[0] - int(0.04 * sr))
    b = min(len(x), idx[-1] + int(0.08 * sr))
    return x[a:b]

CHAIN = ("highpass=f=90,equalizer=f=3200:t=q:w=1.1:g=1.5,"
         "acompressor=threshold=-20dB:ratio=2.5:attack=6:release=120:makeup=2,"
         "loudnorm=I=-16:TP=-1.5:LRA=7")

out = {}
for k, sents in SCRIPT.items():
    parts = []
    for i, s in enumerate(sents):
        parts.append(trim(synth(s)))
        if i < len(sents) - 1:
            parts.append(np.zeros(int(sr * PAUSE), dtype=np.float32))
    x = np.concatenate(parts).astype(np.float32)
    n = int(0.03 * sr); x[:n] *= np.linspace(0, 1, n)
    m = int(0.12 * sr); x[-m:] *= np.linspace(1, 0, m)
    tmp = f"out/vo_{k}_raw.wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    dst = f"public/audio/vo_{k}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", CHAIN, "-ar", "44100", "-ac", "1",
                    "-c:a", "pcm_s16le", dst], check=True)
    with wave.open(dst) as w:
        sec = w.getnframes() / w.getframerate()
    out[k] = {"text": " ".join(sents), "sec": round(sec, 3)}
    print(k, round(sec, 2))
json.dump(out, open("src/vo.json", "w"), indent=1)
print("total", round(sum(v["sec"] for v in out.values()), 1))
