# TinyAttention demo video (Remotion)

About 61 s, 1920x1080, 30 fps. Narration is a local Piper voice (`en_US-ryan-high` by default; set `VOICE=<name>` for
`scripts/vo.py` to use another Piper voice); music and SFX are synthesized. The voice models are not committed (`voices/` is ignored).
Every number on screen comes from the real WASM engine (`scripts/trace.mts` -> `src/trace.json`).

```bash
npm i
python3 -m venv .venv-tts && .venv-tts/bin/pip install piper-tts numpy
.venv-tts/bin/python -m piper.download_voices --download-dir voices en_US-ryan-high
.venv-tts/bin/python scripts/vo.py      # narration -> public/audio, durations -> src/vo.json
.venv-tts/bin/python scripts/audio.py   # music + SFX
node scripts/trace.mts                  # engine trace -> src/trace.json
npx remotion studio src/index.ts        # preview
npm run render                          # out/tinyattention-demo.mp4
cp out/tinyattention-demo.mp4 ../docs/media/   # the copy the top-level README links to
```

The README's preview GIF (`docs/media/tinyattention-demo.gif`) is a 12 s excerpt of the same render, made with ffmpeg
(`palettegen` / `paletteuse`, 960 px wide, 12 fps).
