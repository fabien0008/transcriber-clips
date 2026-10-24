"""CER of the app's accuracy pack (Omnilingual 300M int8, same files as the app) on FLEURS ar_eg:
which utterances are clean enough for an Arabic promo clip. Writes eval_omni_ar_eg.json."""
import re, unicodedata, subprocess, json, sys, numpy as np, sherpa_onnx, csv
from pathlib import Path
M = Path("/home/fabien/dev/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05")
rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(model=str(M/"model.int8.onnx"), tokens=str(M/"tokens.txt"), num_threads=4)
base = Path("/home/fabien/dev/repos/transcriber-clips/src_audio/fleurs")
def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[ً-ْـ]", "", t); t = re.sub(r"[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t); return re.sub(r"\s+", " ", t).strip()
def cer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h)+1))
    for i, rc in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, hc in enumerate(h, 1): p, d[j] = d[j], min(d[j]+1, d[j-1]+1, p+(rc != hc))
    return d[len(h)] / max(len(r), 1)
def audio(p):
    raw = subprocess.run(["ffmpeg","-v","quiet","-i",str(p),"-ac","1","-ar","16000","-f","f32le","-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)
out, seen = [], set()
for r in csv.reader(open(base/"ar_eg.tsv"), delimiter="\t"):
    if r[0] in seen: continue
    wav = next((base/"ar_eg").rglob(r[1]), None)
    if not wav: continue
    seen.add(r[0]); x = audio(wav)
    s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s)
    out.append({"id": r[0], "file": str(wav), "dur": round(len(x)/16000, 1), "cer": round(cer(r[2], s.result.text), 3), "ref": r[2], "hyp": s.result.text.strip()})
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
c = np.array([o["cer"] for o in out])
print(f"n={len(out)} median CER={np.median(c):.1%} perfect={(c==0).sum()} <=2%={(c<=.02).sum()} <=5%={(c<=.05).sum()}")
