# whisper-base-en

English-only Whisper base, INT4-quantised, OpenVINO IR.

- HF source: `OpenVINO/whisper-base.en-int4-ov`
- Devices: CPU, iGPU, NPU
- Helper: `../../helpers/whisper_transcribe.py`

## devices.txt

```
CPU
GPU
NPU
```

## Fetching the weights

```
huggingface-cli download OpenVINO/whisper-base.en-int4-ov \
  --local-dir ~/.local/share/npu-asr/models/whisper-base-en
```
