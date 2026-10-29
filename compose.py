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
import argparse, csv, json, re, subprocess, sys
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
    # Arabic (2026-10-23): recorded with the free accuracy pack installed, which the done line says.
    "ar_aurora": "وضع الطيران مفعّل.\nومع ذلك يحوّل الصوت إلى نص.",
    "ar_italian": "من تسجيلك إلى نص.\nوالصوت يبقى على هاتفك.",
    "ar_tourism": "كلام عربي يصبح نصًا.\nبدون إنترنت.",
    "ar_travel": "تحويل الكلام إلى نص.\nبدون حساب.",
    "ar_transport": "وضع الطيران مفعّل.\nوالنص يظهر رغم ذلك.",
    # Turkish, Indonesian, Hindi (2026-10-29): same recipe as Arabic, FLEURS + the free accuracy pack.
    "tr_polo": "Uçak modu açık.\nYine de yazıya döküyor.",
    "tr_science": "Kaydın yazıya dönüşüyor.\nSes telefonunda kalıyor.",
    "tr_catalan": "Türkçe konuşma yazıya dönüşüyor.\nİnternetsiz.",
    "tr_spanish": "Sesten yazıya.\nHesap gerekmez.",
    "tr_taxes": "Uçak modu açık.\nMetin yine de çıkıyor.",
    "id_satellite": "Mode pesawat aktif.\nTetap bisa transkripsi.",
    "id_ai": "Rekamanmu jadi teks.\nAudionya tetap di ponselmu.",
    "id_egypt": "Suara bahasa Indonesia jadi teks.\nTanpa internet.",
    "id_swim": "Ucapan jadi teks.\nTanpa akun.",
    "id_wheels": "Mode pesawat aktif.\nTeksnya tetap muncul.",
    "hi_pencil": "एयरप्लेन मोड चालू है।\nफिर भी आवाज़ टेक्स्ट बन रही है।",
    "hi_controller": "आपकी रिकॉर्डिंग से टेक्स्ट।\nआवाज़ आपके फ़ोन पर ही रहती है।",
    "hi_birds": "हिंदी बोलें, टेक्स्ट पाएँ।\nबिना इंटरनेट।",
    "hi_weather": "बोली हुई बात टेक्स्ट में।\nबिना अकाउंट।",
    "hi_goats": "एयरप्लेन मोड चालू है।\nटेक्स्ट फिर भी आ गया।",
}
L10N = {  # on-video "done" line, end-card line, link-in-bio line
    "en": ("Transcribed 100% on the phone", "Free on Google Play", "Link in bio"),
    "fr": ("Transcrit à 100 % sur le téléphone", "Gratuit sur Google Play", "Lien dans la bio"),
    "es": ("Transcrito 100% en el teléfono", "Gratis en Google Play", "Enlace en la bio"),
    "pt": ("Transcrito 100% no celular", "Grátis no Google Play", "Link na bio"),
    # Arabic needs the free accuracy pack for this quality: the done line says so, never implied.
    "ar": ("تم التحويل على الهاتف بحزمة الدقة المجانية", "مجاني على Google Play", "الرابط في البايو"),
    "tr": ("Ücretsiz doğruluk paketiyle telefonda yazıya döküldü", "Google Play'de ücretsiz", "Bağlantı profilde"),
    "id": ("Ditranskripsi di ponsel dengan paket akurasi gratis", "Gratis di Google Play", "Tautan di bio"),
    "hi": ("मुफ़्त सटीकता पैक से फ़ोन पर ही ट्रांसक्राइब हुआ", "Google Play पर मुफ़्त", "लिंक बायो में"),
}
CREDITS = {l: "Audio: FLEURS (Google), CC BY 4.0" for l in ("ar", "tr", "id", "hi")}   # FLEURS needs attribution; LibriVox is public domain
CREDIT = "Audio: LibriVox (public domain)"
# Arabic segments: FLEURS ar_eg utterances the pack transcribes with 0 character errors (eval_omni_ar_eg.json),
# neutral topics only; the reference text, to refuse a capture whose on-device text differs.
SEG_DIRS = {l: f"seg_{l}" for l in ("ar", "tr", "id", "hi")}

# Caption = TITLE line + body. basikcode-social's youtube.py uses the first line as the YouTube
# title (+ " #shorts", cut at ~90 chars), and TikTok shows it first too — so it must be a
# short, complete, searchable line: the key phrase plus this clip's hook. The body must not repeat it.
# Honesty rules (Fabien-approved, 2026-09-29): only the AUDIO never leaves the phone (the app does
# send analytics, crash reports and ad requests); "dozens of languages", never "99"; one distinct
# caption per clip, opening on what is said in it.
CAPTIONS = {
    "en_wind_a": ("Speech to text in airplane mode ✈️ The Wind and the Sun",
                  "They argue about who is stronger 🌬️☀️ Speech to Text Offline turns speech into text right on "
                  "your phone, and your audio never leaves it. Works offline in dozens of languages.",
                  "#speechtotext #aesop #fables #offline #privacy #transcription #android"),
    "en_wind_b": ("Offline speech to text: the wind, the traveler and his cloak 🧥",
                  "The harder the wind blew, the tighter he held on. This text was written by the phone itself, "
                  "in airplane mode. No account, and your audio never leaves your device.",
                  "#speechtotext #aesop #storytime #offline #privacy #notes #android"),
    "en_wind_c": ("Aesop's ending, transcribed with no internet ☀️",
                  "And the sun wins. Lectures, meetings, voice notes: Speech to Text Offline writes them down on "
                  "your phone, and the audio stays there.",
                  "#speechtotext #studytips #lecturenotes #offline #productivity #transcription #android"),
    "es_flamencos_a": ("Voz a texto en modo avión ✈️ Los peces aplaudían con la cola 🐟",
                       "Un cuento de Quiroga, transcrito sin internet. Speech to Text Offline convierte la voz en "
                       "texto en tu propio teléfono: tu audio nunca sale de él. Funciona sin conexión en decenas de idiomas.",
                       "#vozatexto #quiroga #cuentos #sininternet #privacidad #transcripcion #android"),
    "es_flamencos_b": ("¿Por qué los flamencos tenían las patas blancas? 🦩 Voz a texto sin internet",
                       "Un clásico de Quiroga, pasado a texto en modo avión ✈️. Notas de voz, clases, reuniones: se "
                       "transcriben en tu teléfono y el audio se queda ahí.",
                       "#vozatexto #flamencos #estudiantes #sininternet #productividad #notas #android"),
    "fr_verne": ("Transcription en mode avion ✈️ « En l'année 1872… » (Jules Verne)",
                 "Le début du Tour du monde en 80 jours 📖 Speech to Text Offline transforme la parole en texte "
                 "directement sur votre téléphone : votre audio ne le quitte jamais. Fonctionne hors ligne dans "
                 "des dizaines de langues.",
                 "#transcription #julesverne #livreaudio #horsligne #vieprivée #étudiant #android"),
    "ar_aurora": ("تحويل الصوت إلى نص بدون إنترنت ✈️ الشفق القطبي",
                  "جملة عن الشفق القطبي، حوّلها الهاتف إلى نص وهو في وضع الطيران. Speech to Text Offline يحوّل الكلام "
                  "إلى نص على هاتفك، والتسجيل الصوتي لا يغادره. للعربية: حزمة دقة مجانية داخل التطبيق.",
                  "#تحويل_الصوت_إلى_نص #تفريغ_صوتي #الشفق_القطبي #بدون_انترنت #خصوصية #اندرويد"),
    "ar_italian": ("الإيطالية لغة الحياة اليومية 🇮🇹 من صوت إلى نص على الهاتف",
                   "هذا النص كتبه الهاتف نفسه من التسجيل، بلا إنترنت. الملاحظات الصوتية والمحاضرات والاجتماعات "
                   "تتحول إلى نص على جهازك، والصوت يبقى عليه.",
                   "#تحويل_الصوت_إلى_نص #تفريغ_صوتي #تعلم_اللغات #ملاحظات_صوتية #بدون_انترنت #اندرويد"),
    "ar_tourism": ("مبانٍ أُعيد بناؤها للسياح 🏛️ تفريغ صوتي بلا إنترنت",
                   "جملة عربية تحولت إلى نص على الهاتف في وضع الطيران ✈️. التطبيق مجاني، والعربية تعمل بحزمة "
                   "دقة مجانية تُنزَّل مرة واحدة.",
                   "#تفريغ_صوتي #تحويل_الكلام_إلى_نص #سياحة #بدون_انترنت #خصوصية #اندرويد"),
    "ar_travel": ("الصبر ضروري حتى عند العودة إلى الديار ✈️ كلام يتحول إلى نص",
                  "تسجيل صوتي يصبح نصًا على هاتفك مباشرة، بدون حساب وبدون إنترنت. مفيد للرسائل الصوتية "
                  "والدروس والمقابلات.",
                  "#تحويل_الصوت_إلى_نص #رسائل_صوتية #سفر #بدون_انترنت #دراسة #اندرويد"),
    "ar_transport": ("الجميع يشتكي من المواصلات 🚌 من صوت إلى نص في وضع الطيران",
                     "حتى بدون إنترنت، يكتب الهاتف ما يقال. Speech to Text Offline يعمل على جهازك، والتسجيل الصوتي "
                     "لا يغادره.",
                     "#تحويل_الصوت_إلى_نص #تفريغ_صوتي #مواصلات #بدون_انترنت #خصوصية #اندرويد"),
    "tr_polo": ("Uçak modunda sesten yazıya ✈️ Arjantin ve polo 🐎",
                "Arjantin ve polo hakkında bir cümle, telefonda internetsiz yazıya döküldü. Speech to Text Offline konuşmayı "
                "telefonunda yazıya çevirir, ses kaydın telefondan çıkmaz. Türkçe için uygulamada ücretsiz bir doğruluk paketi var.",
                "#sestenyazıya #transkripsiyon #polo #internetsiz #gizlilik #android"),
    "tr_science": ("Bilimin temel amacı 🔬 telefonda sesten yazıya",
                   "Bu metni telefon kendisi yazdı, internet olmadan. Sesli notlar, dersler ve toplantılar cihazında yazıya "
                   "dönüşür, ses orada kalır.",
                   "#sestenyazıya #bilim #dersnotu #seslinot #internetsiz #android"),
    "tr_catalan": ("Katalanca tabelalar 🪧 internetsiz sesten yazıya",
                   "Uçak modunda bir Türkçe cümle telefonda yazıya döküldü ✈️. Uygulama ücretsiz, Türkçe için doğruluk "
                   "paketi bir kez indirilir.",
                   "#sestenyazıya #transkripsiyon #diller #internetsiz #gizlilik #android"),
    "tr_spanish": ("Üç asır süren koloni dönemi 📜 konuşma yazıya dönüşüyor",
                   "Ses kaydın doğrudan telefonunda yazıya dönüşür, hesap ve internet gerekmez. Sesli mesajlar, dersler ve "
                   "röportajlar için kullanışlı.",
                   "#sestenyazıya #tarih #seslimesaj #internetsiz #ders #android"),
    "tr_taxes": ("Vergi ve tarifeler 🏛️ uçak modunda sesten yazıya",
                 "İnternet olmasa bile telefon söyleneni yazıyor. Speech to Text Offline cihazında çalışır, ses kaydın oradan "
                 "çıkmaz.",
                 "#sestenyazıya #transkripsiyon #tarih #internetsiz #gizlilik #android"),
    "id_satellite": ("Ubah suara jadi teks dalam mode pesawat ✈️ Satelit di luar angkasa 🛰️",
                     "Kalimat tentang satelit ini ditulis oleh ponsel sendiri, tanpa internet. Speech to Text Offline mengubah "
                     "ucapan jadi teks di ponselmu, dan rekaman suaramu tidak pernah keluar dari situ. Untuk bahasa Indonesia "
                     "ada paket akurasi gratis di dalam aplikasi.",
                     "#suarajaditeks #transkripsi #satelit #tanpainternet #privasi #android"),
    "id_ai": ("Riset kecerdasan buatan 🤖 suara jadi teks di ponsel",
              "Teks ini ditulis ponsel dari rekaman, tanpa internet. Catatan suara, kuliah, dan rapat jadi teks di perangkatmu, "
              "audionya tetap di sana.",
              "#suarajaditeks #kecerdasanbuatan #catatankuliah #catatansuara #tanpainternet #android"),
    "id_egypt": ("Monumen Mesir kuno 🏛️ transkripsi suara tanpa internet",
                 "Satu kalimat bahasa Indonesia jadi teks di ponsel dalam mode pesawat ✈️. Aplikasinya gratis, dan paket akurasi "
                 "untuk bahasa Indonesia cukup diunduh sekali.",
                 "#suarajaditeks #transkripsi #sejarah #tanpainternet #privasi #android"),
    "id_swim": ("Keluar dari arus, berenang jadi mudah 🌊 ucapan jadi teks",
                "Rekaman suara langsung jadi teks di ponselmu, tanpa akun dan tanpa internet. Berguna untuk pesan suara, "
                "pelajaran, dan wawancara.",
                "#suarajaditeks #pesansuara #pantai #tanpainternet #belajar #android"),
    "id_wheels": ("Roda kayu dan besi 🛞 suara jadi teks dalam mode pesawat",
                  "Tanpa internet pun, ponsel menulis apa yang diucapkan. Speech to Text Offline bekerja di perangkatmu, dan "
                  "rekaman suaramu tidak keluar dari situ.",
                  "#suarajaditeks #transkripsi #sejarah #tanpainternet #privasi #android"),
    "hi_pencil": ("एयरप्लेन मोड में आवाज़ से टेक्स्ट ✈️ जब पेंसिल आई ✏️",
                  "पेंसिल के बारे में यह वाक्य फ़ोन ने खुद, बिना इंटरनेट, टेक्स्ट में बदला। Speech to Text Offline आपकी बोली "
                  "को आपके फ़ोन पर ही टेक्स्ट में बदलता है, और आपकी रिकॉर्डिंग फ़ोन से बाहर नहीं जाती। हिंदी के लिए ऐप में "
                  "मुफ़्त सटीकता पैक है।",
                  "#आवाज़सेटेक्स्ट #speechtotext #हिंदी #बिनाइंटरनेट #प्राइवेसी #android"),
    "hi_controller": ("हवा में घुमाकर खेलने वाला कंट्रोलर 🎮 फ़ोन पर आवाज़ से टेक्स्ट",
                      "यह टेक्स्ट फ़ोन ने रिकॉर्डिंग से खुद लिखा, बिना इंटरनेट। वॉइस नोट, क्लास और मीटिंग आपके डिवाइस पर ही "
                      "टेक्स्ट बनते हैं, आवाज़ वहीं रहती है।",
                      "#आवाज़सेटेक्स्ट #गेमिंग #वॉइसनोट #बिनाइंटरनेट #पढ़ाई #android"),
    "hi_birds": ("टी-रेक्स की तरह दो पैरों पर 🦖 बिना इंटरनेट ट्रांसक्रिप्शन",
                 "एयरप्लेन मोड में एक हिंदी वाक्य फ़ोन पर टेक्स्ट बना ✈️। ऐप मुफ़्त है, और हिंदी का सटीकता पैक एक बार "
                 "डाउनलोड करना होता है।",
                 "#आवाज़सेटेक्स्ट #ट्रांसक्रिप्शन #विज्ञान #बिनाइंटरनेट #प्राइवेसी #android"),
    "hi_weather": ("आंधी, ओले और भारी बारिश ⛈️ बोली हुई बात टेक्स्ट में",
                   "आपकी रिकॉर्डिंग सीधे आपके फ़ोन पर टेक्स्ट बनती है, बिना अकाउंट और बिना इंटरनेट। वॉइस मैसेज, क्लास और "
                   "इंटरव्यू के लिए काम का।",
                   "#आवाज़सेटेक्स्ट #मौसम #वॉइसमैसेज #बिनाइंटरनेट #पढ़ाई #android"),
    "hi_goats": ("10,000 साल पहले बकरी पालन 🐐 एयरप्लेन मोड में आवाज़ से टेक्स्ट",
                 "इंटरनेट न हो तब भी फ़ोन बोली हुई बात लिख देता है। Speech to Text Offline आपके डिवाइस पर काम करता है, और "
                 "आपकी रिकॉर्डिंग वहाँ से बाहर नहीं जाती।",
                 "#आवाज़सेटेक्स्ट #इतिहास #ट्रांसक्रिप्शन #बिनाइंटरनेट #प्राइवेसी #android"),
    "pt_azevedo": ("Voz para texto em modo avião ✈️ Um conto de Aluísio Azevedo 🚢",
                   "Um passageiro de 25 anos deixa o camarote… O Speech to Text Offline transforma voz em texto no "
                   "seu próprio celular: seu áudio nunca sai dele. Funciona offline em dezenas de idiomas.",
                   "#vozparatexto #literaturabrasileira #contos #offline #privacidade #estudos #android"),
}


def font(size, bold=True, arabic=False, script=None):
    script = "Arabic" if arabic else script
    name = f"NotoSans{script or ''}-{'Bold' if bold else 'Regular'}"
    for p in ([f"/usr/share/fonts/truetype/noto/{name}.ttf",
               f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


ARABIC = re.compile(r"[\u0600-\u06FF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def runs(line, size, bold=True):
    """A line as (text, font, direction) runs in VISUAL order, left to right. Noto Sans Arabic has no
    Latin letters and Noto Sans no Arabic, so a mixed line ("مجاني على Google Play") is drawn run by
    run; raqm shapes each Arabic run (joined letters, right to left). An Arabic line reads right to
    left, so its runs are laid out in reverse."""
    if DEVANAGARI.search(line):
        # Noto Sans Devanagari has no Latin letters either; left to right, so runs keep their order.
        parts = [p for p in re.split(r"([A-Za-z][A-Za-z0-9 .%:,()'-]*[A-Za-z0-9%)])", line) if p.strip()]
        return [(p.strip(), font(size, bold, script="Devanagari" if DEVANAGARI.search(p) else None), "ltr") for p in parts]
    if not ARABIC.search(line):
        return [(line, font(size, bold), None)]
    parts = [p for p in re.split(r"([A-Za-z][A-Za-z0-9 .%:,()-]*[A-Za-z0-9%)])", line) if p]
    out = [(p.strip(), font(size, bold, arabic=bool(ARABIC.search(p))), "rtl" if ARABIC.search(p) else "ltr")
           for p in parts if p.strip()]
    return list(reversed(out))


def line_width(draw, line, size, bold=True):
    rs = runs(line, size, bold)
    gap = size * 0.28 if len(rs) > 1 else 0
    return sum(draw.textlength(t, font=f, direction=d) for t, f, d in rs) + gap * (len(rs) - 1)


def draw_line(draw, x, y, line, size, fill, bold=True):
    """y = top of the line, as for a Latin line. All runs share ONE baseline (the Latin font's), else
    the Arabic font's taller metrics drop its run below the Latin one and out of the pill."""
    rs = runs(line, size, bold)
    gap = size * 0.28 if len(rs) > 1 else 0
    base = y + font(size, bold).getmetrics()[0]
    for t, f, d in rs:
        draw.text((x, base), t, font=f, fill=fill, direction=d, anchor="ls")
        x += draw.textlength(t, font=f, direction=d) + gap


def text_block(draw, text, fnt, y, fill, width=W, spacing=18):
    bold = "Bold" in (getattr(fnt, "path", "") or "")
    for ln in text.split("\n"):
        w = line_width(draw, ln, fnt.size, bold)
        draw_line(draw, (width - w) / 2, y, ln, fnt.size, fill, bold)
        y += fnt.size + spacing
    return y


def hook_png(text, path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = 66
    while size > 36 and max(line_width(d, l, size) for l in text.split("\n")) > W - 2 * 110:
        size -= 2   # shrink until the widest line fits inside the box with margin
    f = font(size)
    n = text.count("\n") + 1
    box_h = n * (size + 18) + 80
    top = 330
    d.rounded_rectangle((60, top, W - 60, top + box_h), radius=36, fill=(20, 20, 20, 215))
    text_block(d, text, f, top + 40, (255, 255, 255, 255))
    if "✈" in text or "vion" in text.lower() or "irplane" in text or "avião" in text.lower() or "avión" in text.lower() \
            or "الطيران" in text or "Uçak modu" in text or "Mode pesawat" in text or "एयरप्लेन" in text:
        # The claim's proof is the real status-bar airplane icon (x≈943, y≈33 at 1080 wide),
        # too small to notice on a phone without a pointer.
        cx, cy, r = 943, 34, 38
        for w in range(6):
            d.ellipse((cx - r - w, cy - r - w, cx + r + w, cy + r + w), outline=TEAL + (255,))
    img.save(path)


def free_band_top(frame, x0, x1, bh, want=330):
    """Top of the band nearest `want` that covers the least dark ink (text, teal buttons) in the result
    frame, so the pill never hides the transcript. The Arabic run scrolls the result card up and the
    fixed y=330 pill hid its first line (2026-10-24). Kept above the bottom ~20% that TikTok covers."""
    rgb = np.asarray(Image.open(frame).convert("RGB").resize((W, H)), dtype=np.int16)[:, int(max(0, x0)):int(min(W, x1))]
    g = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    # Dark text, and coloured buttons too: a two-line pill over the red Record button left its red ends
    # sticking out on both sides (Turkish clip, 2026-10-29).
    ink = np.concatenate([[0], np.cumsum(((g < 95) | (sat > 90)).sum(axis=1))])
    # Top part only (header, Open file / Record area): the transcript is always lower down, and a search over
    # the whole screen put a two-line pill on the transcript's blank bottom and its link (2026-10-29).
    tops = list(range(130, 460 - min(bh, 200) + 200, 5))
    cost = [ink[t + bh + 12] - ink[t - 12] for t in tops]
    low = min(cost)
    return min((t for t, c in zip(tops, cost) if c <= low + 50), key=lambda t: abs(t - want))


def done_png(text, path, frame=None):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size, lines = 50, [text]
    while size > 40 and line_width(d, text, size) > W - 2 * 130:
        size -= 2   # fit inside the band ("Transcrit à 100 % sur le téléphone" overflowed)
    if line_width(d, text, size) > W - 2 * 130:
        # Still too wide at a readable size (Turkish, Indonesian done lines): two balanced lines.
        words = text.split()
        k = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
        lines, size = [" ".join(words[:k]), " ".join(words[k:])], 48
        while size > 34 and max(line_width(d, l, size) for l in lines) > W - 2 * 130:
            size -= 2
    lh = size + 14
    bh = len(lines) * lh + 46
    tw = max(line_width(d, l, size) for l in lines)
    top = free_band_top(frame, (W - tw) / 2 - 60, (W + tw) / 2 + 60, bh) if frame else 330
    d.rounded_rectangle(((W - tw) / 2 - 50, top, (W + tw) / 2 + 50, top + bh), radius=min(bh // 2, 48), fill=TEAL + (240,))
    for n, l in enumerate(lines):
        draw_line(d, (W - line_width(d, l, size)) / 2, top + 23 + n * lh, l, size, (255, 255, 255, 255))
    img.save(path)


def end_png(line, bio, path, credit=CREDIT):
    """One centred block, all inside the safe zone. TikTok/YouTube Shorts cover the bottom ~20% and the
    right edge with their own buttons, so the first version's credit at y=1560 would have been
    hidden. Caption links can't be tapped there either, hence "Link in bio" (the BasikCode bio
    page lists this app)."""
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    icon = Image.open(ICON).convert("RGBA").resize((340, 340))
    img.paste(icon, ((W - 340) // 2, 470), icon)
    size = 70
    while d.textlength("Speech to Text Offline", font=font(size)) > W - 2 * 150:
        size -= 2   # keep clear of the right edge, where TikTok puts its buttons
    y = text_block(d, "Speech to Text Offline", font(size), 870, INK)
    y = text_block(d, line, font(54, bold=False), y + 16, TEAL)
    tw = line_width(d, bio, 46)
    d.rounded_rectangle(((W - tw) / 2 - 44, y + 36, (W + tw) / 2 + 44, y + 36 + 92), radius=46, fill=TEAL)
    draw_line(d, (W - tw) / 2, y + 56, bio, 46, (255, 255, 255))
    text_block(d, credit, font(30, bold=False), y + 190, (110, 110, 110))
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
    # Tall crop: in RTL locales the app scrolls the result card up when the text lands.
    x0, y0, x1, y1 = 100, 300, 980, 1750
    fps = 10
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t_open}", "-to", f"{t_done + 2}", "-i", str(rec),
                        "-vf", f"fps={fps},crop={x1-x0}:{y1-y0}:{x0}:{y0},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    n = (x1 - x0) * (y1 - y0)
    fr = np.frombuffer(p.stdout, np.uint8)
    fr = fr[: len(fr) // n * n].reshape(-1, n)
    dark = (fr < 90).sum(axis=1)
    base = np.median(dark)                              # the long "transcribing" stretch, not the first second
    # The text stays on screen until the recording ends: it landed right after the LAST frame that
    # was still near the "transcribing" baseline (robust to a busy home screen in the first second).
    quiet = np.nonzero(dark <= base + 1500)[0]
    hit = quiet[-1:] + 1 if len(quiet) and quiet[-1] + 1 < len(dark) else np.array([], int)
    return t_open + (hit[0] / fps if len(hit) else (t_done - t_open))


def norm_ar(t, lang="ar"):
    """On-screen text compared the way readers see it: no diacritics/tatweel, alef/ya/ta-marbuta variants,
    no case, no punctuation (the pack writes none; the reference has some). Turkish dotted/dotless i are
    folded first (Python's lower() turns İ into i + a combining dot). Devanagari vowel signs are kept."""
    import unicodedata
    t = unicodedata.normalize("NFKC", t)
    if lang == "tr":
        t = t.replace("İ", "i").replace("I", "ı")
    t = t.lower()
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t); t = re.sub(r"[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"\[[0-9:]+\]", " ", t)                 # timestamps, if the setting was on
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0900-\u0963\u0966-\u097F]", " ", t)).strip()


def compose(cid, capture, month, idx, tmp):
    lang = cid.split("_")[0]
    rec = capture / f"{cid}.mp4"
    meta = json.loads((capture / f"{cid}.json").read_text())
    ref = HERE / SEG_DIRS.get(lang, "seg") / f"{cid}.txt"
    if ref.exists():
        # A showcase: the text on screen must be exactly what was said (Fabien, 2026-10-23: "très vigilant").
        got, want = norm_ar(meta["transcript"], lang), norm_ar(ref.read_text(encoding="utf-8"), lang)
        if got != want:
            sys.exit(f"{cid}: on-screen text differs from the reference, clip refused\n  got:  {got}\n  want: {want}")
    wav = HERE / SEG_DIRS.get(lang, "seg") / f"{cid}.wav"
    D = duration(wav)
    t_open = max(0.0, float(meta["t_open"]) - 0.6)
    rec_len = duration(rec)
    # t_done can fall after the recording's end (slow pack runs); a seek past the end makes the
    # looped "done" overlay never terminate, so keep 2 s of real video after the text lands.
    t_text = min(detect_text_time(rec, float(meta["t_open"]), float(meta["t_done"])), rec_len - 2.0)
    hold = 4.0   # time to read the result; if the recording ends sooner its last (static) frame is held
    B = max(D, 7.0)                      # text lands as the speech ends (min 7 s so the wait reads)
    speed = (t_text - t_open) / B        # >1 = compressed
    done_line, end_line, bio_line = L10N[lang]
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{min(t_text + 1.5, rec_len - 0.3)}", "-i", str(rec),
        "-frames:v", "1", str(tmp / "result.png"))
    hook_png(HOOKS[cid], tmp / "hook.png"); done_png(done_line, tmp / "done.png", tmp / "result.png")
    end_png(end_line, bio_line, tmp / "end.png", CREDITS.get(lang, CREDIT))
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
        "-filter_complex", f"[0:v]{norm},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={hold}[v];[v][1:v]overlay=shortest=1",
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
    title, body, tags = CAPTIONS[cid]
    assert len(title) <= 80 and "\n" not in title, (cid, len(title))  # YouTube title = title + " #shorts"
    return f"{title}\n{body}\n\n{end_line}: {PLAY.format(month=month)}\n\n{tags}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--month", required=True)
    ap.add_argument("--only")
    ap.add_argument("--manifest-only", action="store_true", help="rewrite captions; keep the rendered clips")
    a = ap.parse_args()
    # One order per month; languages interleaved so consecutive posts differ. A month's list is only ever
    # appended to, so the published file names (NN_<id>) never change.
    orders = {
        "2026-11": ["en_wind_a", "es_flamencos_a", "fr_verne", "en_wind_b", "pt_azevedo", "es_flamencos_b", "en_wind_c",
                    "ar_aurora", "ar_italian", "ar_tourism", "ar_travel", "ar_transport"],
        # tr_spanish and hi_weather dropped (2026-10-29): not exact on the phone (Turkish dotted-I glitch, fixed
        # in 2.6.56; two Hindi vowel signs off). Numbers are positions in this list, so clips were re-rendered.
        "2026-12": ["tr_polo", "hi_pencil", "id_satellite", "tr_science", "hi_controller", "id_ai", "tr_catalan",
                    "hi_birds", "id_egypt", "id_swim", "tr_taxes", "hi_goats", "id_wheels"],
    }
    order = orders[a.month]
    tmp = HERE / ".tmp"; tmp.mkdir(exist_ok=True)
    rows = []
    for i, cid in enumerate(order, 1):
        if a.only and cid != a.only:
            continue
        if a.manifest_only:
            out = HERE / "clips" / a.month / f"{i:02d}_{cid}.mp4"
            assert out.exists(), out
        else:
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
