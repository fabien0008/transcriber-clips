"""Transcribe audio segments on the host with the app's own whisper-base model (auto language),
to pick clean clip segments before recording them in the app."""
import sys, subprocess, numpy as np, sherpa_onnx
M = "/home/fabien/dev/repos/ondevice-audio-transcriber/app/src/main/assets/sherpa-whisper-base/"
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M+"base-encoder.int8.onnx", decoder=M+"base-decoder.int8.onnx",
                                                 tokens=M+"base-tokens.txt", language="", task="transcribe", num_threads=4)
def seg(path, start, dur):
    pcm = subprocess.run(["ffmpeg","-v","error","-ss",str(start),"-t",str(dur),"-i",path,"-ac","1","-ar","16000","-f","s16le","-"],capture_output=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32)/32768
if __name__ == "__main__":
    path, dur = sys.argv[1], float(sys.argv[2])
    for s in sys.argv[3:]:
        st = rec.create_stream(); st.accept_waveform(16000, seg(path, float(s), dur)); rec.decode_stream(st)
        print(f"[{s}s +{dur}s] lang={st.result.lang} :: {st.result.text.strip()}")
