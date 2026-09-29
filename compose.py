#!/usr/bin/env python3
"""
Compose BasikCode social shorts for Speech to Text Offline from raw app screen recordings.

Input per clip (from e2e/flows/transcriber_shorts_capture.sh in the app repo):
  <capture_dir>/<id>.mp4   screenrecord of the app transcribing the segment, airplane mode ON
  <capture_dir>/<id>.json  {"t_open", "t_done", "transcript"} (seconds from recording start)
  seg/<id>.wav             the audio segment that was transcribed (played under the video)

Output: clips/<month>/<NN>_<id>.mp4 (1080x1920, 30 fps, H.264 + AAC) and manifest/<month>.csv
(Date, Time, Text, Picture Url 1) for basikcode-social, which owns scheduling. Dates/times in the
manifest are placeholders — the shared calendar only uses the ROW ORDER.

Timeline of each clip:
  hook (2.4 s, over the recording)  ->  recording, the wait time-compressed so the text appears as
  the speech ends  ->  transcript held ~4.5 s with a "done" line  ->  3 s end card.
The wait is compressed because the emulator decodes slower than a phone; no speed is claimed.

  python3 compose.py --capture <artifacts>/shorts --month 2026-11
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS = 1080, 1920, 30
ICON = Path.home() / "dev/repos/ondevice-audio-transcriber/app-icon-512.png"
BASE_URL = "https://fabien0008.github.io/transcriber-clips"
PLAY = ("https://play.google.com/store/apps/details?id=com.basikcode.transcriber"
        "&referrer=utm_source%3Dsocial%26utm_medium%3Dshort%26utm_campaign%3D{month}")
TEAL, INK, SURFACE = (14, 124, 134), (26, 26, 26), (255, 254, 250)

# Hook shown over the first seconds, per clip. Everything claimed here is literally true of the
# recording: airplane mode is on and the transcription happens on the device.
HOOKS = {
    "en_wind_a": "Airplane mode on.\nIt still transcribes.",
    "en_wind_b": "Your recording → text.\nThe audio stays on your phone.",
    "en_wind_c": "Speech to text, offline.\nNo account needed.",
    "fr_verne": "Mode avion activé.\nEt ça transcrit quand même.",
    "es_flamencos_a": "Modo avión activado.\nY transcribe igual.",
    "es_flamencos_b": "De audio a texto,\nsin internet.",
    "pt_azevedo": "Modo avião ligado.\nE transcreve mesmo assim.",
}
L10N = {  # on-video "done" line, end-card line
    "en": ("Transcribed 100% on the phone", "Free on Google Play"),
    "fr": ("Transcrit à 100 % sur le téléphone", "Gratuit sur Google Play"),
    "es": ("Transcrito 100% en el teléfono", "Gratis en Google Play"),
    "pt": ("Transcrito 100% no celular", "Grátis no Google Play"),
}
CREDIT = "Audio: LibriVox (public domain)"

CAPTIONS = {
    "en_wind_a": ("The Wind and the Sun argue about who is stronger 🌬️☀️ Transcribed with airplane mode on ✈️. "
                  "Speech to Text Offline turns speech into text right on your phone, and your audio never leaves it. "
                  "Works offline in dozens of languages.",
                  "#speechtotext #aesop #fables #offline #privacy #transcription #android"),
    "en_wind_b": ("The harder the wind blew, the tighter he held his cloak 🧥 This text was written by the phone itself, "
                  "in airplane mode. No account, and your audio never leaves your device.",
                  "#speechtotext #aesop #storytime #offline #privacy #notes #android"),
    "en_wind_c": ("And the sun wins ☀️ Aesop's ending, transcribed with no internet at all. Lectures, meetings, voice "
                  "notes: Speech to Text Offline writes them down on your phone, and the audio stays there.",
                  "#speechtotext #studytips #lecturenotes #offline #productivity #transcription #android"),
    "es_flamencos_a": ("Los peces aplaudían con la cola 🐟👏 Quiroga, transcrito en modo avión ✈️. Speech to Text Offline "
                       "convierte la voz en texto en tu propio teléfono: tu audio nunca sale de él. Funciona sin conexión "
                       "en decenas de idiomas.",
                       "#vozatexto #quiroga #cuentos #sininternet #privacidad #transcripcion #android"),
    "es_flamencos_b": ("¿Por qué los flamencos tenían las patas blancas? 🦩 Un clásico de Quiroga, pasado a texto sin "
                       "internet. Notas de voz, clases, reuniones: se transcriben en tu teléfono y el audio se queda ahí.",
                       "#vozatexto #flamencos #estudiantes #sininternet #productividad #notas #android"),
    "fr_verne": ("« En l'année 1872… » 📖 Le début du Tour du monde en 80 jours, transcrit en mode avion ✈️. "
                 "Speech to Text Offline transforme la parole en texte directement sur votre téléphone : votre audio "
                 "ne le quitte jamais. Fonctionne hors ligne dans des dizaines de langues.",
                 "#transcription #julesverne #livreaudio #horsligne #vieprivée #étudiant #android"),
    "pt_azevedo": ("Um passageiro de 25 anos deixa o camarote… 🚢 Conto de Aluísio Azevedo, transcrito em modo avião ✈️. "
                   "O Speech to Text Offline transforma voz em texto no seu próprio celular: seu áudio nunca sai dele. "
                   "Funciona offline em dezenas de idiomas.",
                   "#vozparatexto #literaturabrasileira #contos #offline #privacidade #estudos #android"),
}


def font(size, bold=True):
    for p in ([f"/usr/share/fonts/truetype/noto/NotoSans-{'Bold' if bold else 'Regular'}.ttf",
               f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def text_block(draw, text, fnt, y, fill, width=W, spacing=18):
    lines = text.split("\n")
    for ln in lines:
        w = draw.textlength(ln, font=fnt)
        draw.text(((width - w) / 2, y), ln, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


def hook_png(text, path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = 66
    while size > 36 and max(d.textlength(l, font=font(size)) for l in text.split("\n")) > W - 2 * 110:
        size -= 2   # shrink until the widest line fits inside the box with margin
    f = font(size)
    n = text.count("\n") + 1
    box_h = n * (size + 18) + 80
    top = 330
    d.rounded_rectangle((60, top, W - 60, top + box_h), radius=36, fill=(20, 20, 20, 215))
    text_block(d, text, f, top + 40, (255, 255, 255, 255))
    if "✈" in text or "vion" in text.lower() or "irplane" in text or "avião" in text.lower() or "avión" in text.lower():
        # The claim's proof is the real status-bar airplane icon (x≈943, y≈33 at 1080 wide),
        # too small to notice on a phone without a pointer.
        cx, cy, r = 943, 34, 38
        for w in range(6):
            d.ellipse((cx - r - w, cy - r - w, cx + r + w, cy + r + w), outline=TEAL + (255,))
    img.save(path)


def done_png(text, path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = 50
    while size > 30 and d.textlength(text, font=font(size)) > W - 2 * 130:
        size -= 2   # fit inside the band ("Transcrit à 100 % sur le téléphone" overflowed)
    f = font(size)
    top, bh = 330, size + 60
    tw = d.textlength(text, font=f)
    d.rounded_rectangle(((W - tw) / 2 - 50, top, (W + tw) / 2 + 50, top + bh), radius=bh // 2, fill=TEAL + (240,))
    d.text(((W - tw) / 2, top + 26), text, font=f, fill=(255, 255, 255, 255))
    img.save(path)


def end_png(line, path):
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    icon = Image.open(ICON).convert("RGBA").resize((380, 380))
    img.paste(icon, ((W - 380) // 2, 520), icon)
    y = text_block(d, "Speech to Text Offline", font(72), 980, INK)
    y = text_block(d, line, font(56, bold=False), y + 30, TEAL)
    text_block(d, CREDIT, font(30, bold=False), 1560, (110, 110, 110))
    img.save(path)


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"command failed: {' '.join(cmd[:6])}...\n{r.stderr[-1500:]}")
    return r


def duration(path):
    return float(run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)).stdout)


def detect_text_time(rec, t_open, t_done):
    """First time the transcript box fills with dark text. The capture's poll timestamps are only
    ~2-4 s precise; the video itself says exactly when the text lands."""
    x0, y0, x1, y1 = 100, 1150, 980, 1700
    fps = 10
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t_open}", "-to", f"{t_done + 2}", "-i", str(rec),
                        "-vf", f"fps={fps},crop={x1-x0}:{y1-y0}:{x0}:{y0},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    n = (x1 - x0) * (y1 - y0)
    fr = np.frombuffer(p.stdout, np.uint8)
    fr = fr[: len(fr) // n * n].reshape(-1, n)
    dark = (fr < 90).sum(axis=1)
    base = dark[: max(3, fps)].min()
    hit = np.nonzero(dark > base + 3000)[0]
    return t_open + (hit[0] / fps if len(hit) else (t_done - t_open))


def compose(cid, capture, month, idx, tmp):
    lang = cid.split("_")[0]
    rec = capture / f"{cid}.mp4"
    meta = json.loads((capture / f"{cid}.json").read_text())
    wav = HERE / "seg" / f"{cid}.wav"
    D = duration(wav)
    t_open = max(0.0, float(meta["t_open"]) - 0.6)
    t_text = detect_text_time(rec, float(meta["t_open"]), float(meta["t_done"]))
    rec_len = duration(rec)
    hold = min(4.5, max(2.0, rec_len - t_text - 0.2))
    B = max(D, 7.0)                      # text lands as the speech ends (min 7 s so the wait reads)
    speed = (t_text - t_open) / B        # >1 = compressed
    done_line, end_line = L10N[lang]
    hook_png(HOOKS[cid], tmp / "hook.png"); done_png(done_line, tmp / "done.png"); end_png(end_line, tmp / "end.png")
    total = B + hold + 3.0
    out = HERE / "clips" / month / f"{idx:02d}_{cid}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    venc = ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-r", str(FPS), "-pix_fmt", "yuv420p", "-an"]
    norm = f"fps={FPS},scale={W}:{H},setsar=1"
    # Each part is rendered to its own file with an explicit -t. Overlay PNGs are looped and
    # overlay uses shortest=1, so the recording (not the image) bounds each part.
    # Each part is rendered to its own file with an explicit -t. The first version built one
    # filter graph with looped PNG overlays; a looped image never ends, so the "done" segment
    # ran forever and the end card was cut off by -t (and the done line never showed).
    p1, p2, p3 = tmp / "p1.mp4", tmp / "p2.mp4", tmp / "p3.mp4"
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{t_open}", "-to", f"{t_text}", "-i", str(rec),
        "-loop", "1", "-i", str(tmp / "hook.png"), "-filter_complex",
        f"[0:v]{norm},setpts=(PTS-STARTPTS)/{speed}[v];[v][1:v]overlay=enable='lt(t,2.4)':shortest=1",
        "-t", f"{B}", *venc, str(p1))
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{t_text}", "-i", str(rec), "-loop", "1", "-i", str(tmp / "done.png"),
        "-filter_complex", f"[0:v]{norm},setpts=PTS-STARTPTS[v];[v][1:v]overlay=shortest=1",
        "-t", f"{hold}", *venc, str(p2))
    run("ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(tmp / "end.png"),
        "-vf", norm, "-t", "3", *venc, str(p3))
    run("ffmpeg", "-v", "error", "-y", "-i", str(p1), "-i", str(p2), "-i", str(p3), "-i", str(wav),
        "-filter_complex", f"[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];[3:a]apad=whole_dur={total},afade=t=in:d=0.1[a]",
        "-map", "[v]", "-map", "[a]", "-t", f"{total}", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out))
    return out, dict(D=round(D, 1), t_text=round(t_text, 1), speed=round(speed, 2), total=round(total, 1))


def caption(cid, month):
    end_line = L10N[cid.split("_")[0]][1]
    body, tags = CAPTIONS[cid]
    return f"{body}\n\n{end_line}: {PLAY.format(month=month)}\n\n{tags}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--month", required=True)
    ap.add_argument("--only")
    a = ap.parse_args()
    # Interleave languages so consecutive posts differ.
    order = ["en_wind_a", "es_flamencos_a", "fr_verne", "en_wind_b", "pt_azevedo", "es_flamencos_b", "en_wind_c"]
    tmp = HERE / ".tmp"; tmp.mkdir(exist_ok=True)
    rows = []
    for i, cid in enumerate(order, 1):
        if a.only and cid != a.only:
            continue
        out, info = compose(cid, a.capture, a.month, i, tmp)
        print(f"{out.name}: {info}", flush=True)
        rows.append({"Date": f"{a.month}-{i:02d}", "Time": "18:00:00", "Text": caption(cid, a.month),
                     "Picture Url 1": f"{BASE_URL}/clips/{a.month}/{out.name}"})
    if not a.only:
        man = HERE / "manifest" / f"{a.month}.csv"
        man.parent.mkdir(exist_ok=True)
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Date", "Time", "Text", "Picture Url 1"])
            w.writeheader(); w.writerows(rows)
        print(f"manifest: {man} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
