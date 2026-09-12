#!/usr/bin/env python3
"""ONNX export the indicwav2vec encoder, then ovc to OpenVINO IR, then smoke-test on Nepali wav.
Avoids the transformers-5.5/ovc tracing bug by going through ONNX.
"""
import os, sys, json, time, wave
import numpy as np

LAB = "/home/shlok/Projects/omarchy-asr/nepali-lab"
MODEL_DIR = f"{LAB}/models/indicwav2vec"
OUT_ONNX = f"{LAB}/models/indicwav2vec_enc.onnx"
OUT_OV   = f"{LAB}/models/indicwav2vec_ov"
WAV = f"{LAB}/smoke/nep_trim1.wav"

def load_wav_mono_f32(path):
    # Python 3.12's stdlib `wave` rejects WAVE_FORMAT_IEEE_FLOAT (tag 3).
    # Parse the RIFF header manually to be format-agnostic.
    with open(path, "rb") as f:
        head = f.read(64)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise SystemExit(f"not a RIFF/WAVE file: {path}")
    # walk chunks
    fmt_tag = bits = rate = nch = 0
    data_off = data_len = 0
    pos = 12
    with open(path, "rb") as f:
        body = f.read()
    while pos + 8 <= len(body):
        cid = body[pos:pos+4]
        csz = int.from_bytes(body[pos+4:pos+8], "little")
        cbody = pos + 8
        if cid == b"fmt ":
            fmt_tag = int.from_bytes(body[cbody:cbody+2], "little")
            nch     = int.from_bytes(body[cbody+2:cbody+4], "little")
            rate    = int.from_bytes(body[cbody+4:cbody+8], "little")
            bits    = int.from_bytes(body[cbody+14:cbody+16], "little")
        elif cid == b"data":
            data_off = cbody
            data_len = csz
            break
        pos = cbody + csz + (csz & 1)
    if nch != 1: raise SystemExit(f"only mono supported (got nch={nch})")
    if rate != 16000: raise SystemExit(f"need 16kHz (got {rate})")
    blob = body[data_off:data_off+data_len]
    if fmt_tag == 3 and bits == 32:
        data = np.frombuffer(blob, dtype="<f4")
    elif fmt_tag == 1 and bits == 16:
        data = np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise SystemExit(f"unsupported fmt_tag={fmt_tag} bits={bits}")
    return data, rate

def main():
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    import openvino as ov

    # --- MONKEY PATCH (lab only) ---
    # transformers 5.5 has a bug where Wav2Vec2Encoder.forward unconditionally calls
    # create_bidirectional_mask -> eager_mask -> sdpa_mask which crashes during
    # torch.jit.trace / ONNX export with `IndexError: tuple index out of range`
    # (q_length is a tuple, not a tensor, in the trace context).
    # Bidirectional attention doesn't actually need a materialized mask, so we
    # short-circuit the function to return None.
    import transformers.masking_utils as _mu
    def _noop_bidir(*args, **kwargs):
        return None
    _mu.create_bidirectional_mask = _noop_bidir
    # Wav2Vec2Encoder also calls it directly via the package re-export
    import transformers.models.wav2vec2.modeling_wav2vec2 as _w2v
    _w2v.create_bidirectional_mask = _noop_bidir
    print("    [patch] create_bidirectional_mask -> None (tracing bypass)")

    print("[1] load HF model + processor (eager attn)")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, attn_implementation="eager")
    model.eval()
    print(f"    loaded in {time.time()-t0:.1f}s")

    lm_head_weight = model.lm_head.weight.detach().cpu().numpy()  # [V, H]
    print(f"    lm_head: {lm_head_weight.shape}")

    class EncoderOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_values, attention_mask):
            out = self.m.wav2vec2(input_values, attention_mask=attention_mask)
            return out.last_hidden_state

    enc = EncoderOnly(model); enc.eval()
    dummy = torch.zeros(1, 16000 * 5, dtype=torch.float32)
    dummy_mask = torch.ones(1, 16000 * 5, dtype=torch.long)

    print("[2] torch.onnx.export ->", OUT_ONNX)
    if not os.path.exists(OUT_ONNX):
        torch.onnx.export(
            enc, (dummy, dummy_mask), OUT_ONNX,
            input_names=["input_values", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "B", 1: "T"},
                "attention_mask": {0: "B", 1: "T"},
                "last_hidden_state": {0: "B", 1: "T_OUT"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        sz = os.path.getsize(OUT_ONNX)
        print(f"    ONNX saved {sz/1e6:.1f} MB")
    else:
        print(f"    ONNX already present ({os.path.getsize(OUT_ONNX)/1e6:.1f} MB), skipping export")

    print("[3] ovc ONNX -> OpenVINO IR")
    os.makedirs(OUT_OV, exist_ok=True)
    ir_xml = f"{OUT_OV}/model.xml"
    if not os.path.exists(ir_xml):
        import subprocess
        r = subprocess.run(["ovc", OUT_ONNX, "--output_model", f"{OUT_OV}/model"], capture_output=True, text=True)
        print("    ovc exit:", r.returncode)
        if r.stdout: print("    stdout:", r.stdout[-1000:])
        if r.stderr: print("    stderr:", r.stderr[-1000:])
        if r.returncode != 0: return
    else:
        print(f"    IR already present, skipping ovc")

    ir_xml = f"{OUT_OV}/model.xml"
    print(f"    IR: {ir_xml} (+ .bin)  (skipped re-export)")

    print("[4] smoke-test on", WAV)
    pcm, rate = load_wav_mono_f32(WAV)
    print(f"    wav: {len(pcm)} samples ({len(pcm)/rate:.2f}s)")
    inputs = proc(pcm, sampling_rate=rate, return_tensors="np")
    input_values = inputs.input_values.astype(np.float32)
    attn_mask = np.ones_like(input_values, dtype=np.int64)

    core = ov.Core()
    print("    devices:", core.get_available_devices())
    for dev in ("GPU", "CPU", "NPU"):
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
            print(f"    [{dev}] FAIL: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()