"""Character error rate of the app's model (host whisper-base) on FLEURS, vs the reference text.
Decides which FLEURS utterances are clean enough to show in a promo clip."""
import sys, csv, re, unicodedata, subprocess, json, numpy as np, sherpa_onnx
from pathlib import Path
M = "/home/fabien/dev/repos/ondevice-audio-transcriber/app/src/main/assets/sherpa-whisper-base/"
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M+"base-encoder.int8.onnx", decoder=M+"base-decoder.int8.onnx",
                                                 tokens=M+"base-tokens.txt", language="", task="transcribe", num_threads=4)
def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[ً-ْـ]", "", t)          # Arabic diacritics / tatweel
    t = re.sub(r"[إأآ]", "ا", t)                           # alef variants
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h)+1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j]+1, d[j-1]+1, prev + (rc != hc))
    return d[len(h)] / max(len(r), 1)
def audio(path):
    pcm = subprocess.run(["ffmpeg","-v","error","-i",str(path),"-ac","1","-ar","16000","-f","s16le","-"],capture_output=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32)/32768
lang, n = sys.argv[1], int(sys.argv[2])
base = Path("src_audio/fleurs")
rows = list(csv.reader(open(base/f"{lang}.tsv", encoding="utf-8"), delimiter="\t"))
seen, out = set(), []
for r in rows:
    if r[0] in seen: continue            # one recording per sentence
    seen.add(r[0])
    wav = next((base/lang).rglob(r[1]), None)
    if not wav: continue
    x = audio(wav); s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s)
    out.append({"id": r[0], "file": str(wav), "dur": round(len(x)/16000, 1), "lang": s.result.lang,
                "cer": round(cer(r[2], s.result.text), 3), "ref": r[2], "hyp": s.result.text.strip()})
    if len(out) >= n: break
json.dump(out, open(f"eval_{lang}.json","w"), ensure_ascii=False, indent=1)
c = np.array([o["cer"] for o in out])
print(f"{lang}: n={len(out)} median CER={np.median(c):.1%}  <=5%: {(c<=.05).sum()}  <=2%: {(c<=.02).sum()}  perfect: {(c==0).sum()}  detected-lang: {sorted(set(o['lang'] for o in out))}")
