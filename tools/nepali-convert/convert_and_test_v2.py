#!/usr/bin/env python3
"""Convert sumanpaudel1997/nepali-asr-indicwav2vec to OpenVINO IR + smoke-test on a Nepali wav.

Strategy: skip the SDPA masking path that breaks OVC tracing. Export the encoder only
(hidden_states out) as OV IR; the CTC head is one linear, kept in numpy for decoding.
"""
import os, sys, json, time, wave
import numpy as np

LAB = "/home/shlok/Projects/omarchy-asr/nepali-lab"
MODEL_DIR = f"{LAB}/models/indicwav2vec"
OUT_DIR = f"{LAB}/models/indicwav2vec_ov"
WAV = f"{LAB}/smoke/nep_trim1.wav"

def load_wav_mono_f32(path):
    with wave.open(path, "rb") as w:
        nch = w.getnchannels(); rate = w.getframerate(); n = w.getnframes()
        sw = w.getsampwidth()
        if nch != 1: raise SystemExit(f"only mono supported (got nch={nch})")
        if rate != 16000: raise SystemExit(f"need 16kHz (got {rate})")
        if sw == 4: data = np.frombuffer(w.readframes(n), dtype="<f4")
        elif sw == 2: data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
        else: raise SystemExit(f"unsupported sampwidth={sw}")
    return data, rate

def main():
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    import openvino as ov
    from openvino import convert_model, serialize

    print("[1] load HF model + processor")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, attn_implementation="eager")
    model.eval()
    print(f"    loaded in {time.time()-t0:.1f}s")

    lm_head_weight = model.lm_head.weight.detach().cpu().numpy()  # [V, H]
    print(f"    lm_head: {lm_head_weight.shape}")

    print("[2] trace encoder to OpenVINO IR (bypass SDPA mask)")
    os.makedirs(OUT_DIR, exist_ok=True)

    class EncoderOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_values, attention_mask):
            out = self.m.wav2vec2(input_values, attention_mask=attention_mask)
            return out.last_hidden_state

    enc = EncoderOnly(model); enc.eval()
    dummy = torch.zeros(1, 16000 * 5, dtype=torch.float32)
    dummy_mask = torch.ones(1, 16000 * 5, dtype=torch.long)
    with torch.no_grad():
        ov_model = convert_model(
            enc, example_input=(dummy, dummy_mask), input=[1, 80000, 1, 80000])
    ir_xml = f"{OUT_DIR}/model.xml"; ir_bin = f"{OUT_DIR}/model.bin"
    serialize(ov_model, ir_xml, ir_bin)
    print(f"    saved {ir_xml} (+ .bin)")

    print("[3] smoke-test on", WAV)
    pcm, rate = load_wav_mono_f32(WAV)
    print(f"    wav: {len(pcm)} samples ({len(pcm)/rate:.2f}s)")
    inputs = proc(pcm, sampling_rate=rate, return_tensors="np")
    input_values = inputs.input_values.astype(np.float32)
    # build attention mask (1s where audio, 0s for pad). With no pad, all 1s.
    attn_mask = np.ones_like(input_values, dtype=np.int64)

    core = ov.Core()
    print("    devices:", core.get_available_devices())
    for dev in ("GPU", "CPU"):
        try:
            t0 = time.time()
            comp = core.compile_model(ir_xml, dev)
            hidden = comp([input_values, attn_mask])[0][0]
            enc_ms = (time.time() - t0) * 1000

            t0 = time.time()
            logits = hidden @ lm_head_weight.T
            ids = logits.argmax(axis=-1)
            blank_id = 80
            prev = -1
            out_ids = []
            for i in ids:
                if i != prev and i != blank_id:
                    out_ids.append(int(i))
                prev = i
            dec_ms = (time.time() - t0) * 1000

            vocab = json.load(open(f"{MODEL_DIR}/vocab.json"))
            inv = {v: k for k, v in vocab.items()}
            text = "".join(inv.get(i, "?") for i in out_ids)
            print(f"    [{dev}] enc={enc_ms:.0f}ms head+dec={dec_ms:.0f}ms; hidden {hidden.shape}; ids: {out_ids[:30]}...")
            print(f"    [{dev}] TEXT: {text!r}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [{dev}] FAIL: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()