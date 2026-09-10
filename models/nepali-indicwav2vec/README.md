# nepali-indicwav2vec

OpenVINO IR of `sumanpaudel1997/nepali-asr-indicwav2vec`. Suman Paudel's
Nepali fine-tune of AI4Bharat's Hindi indicwav2vec, on OpenSLR SLR54
(165h Nepali).

- HF source: `sumanpaudel1997/nepali-asr-indicwav2vec`
- Devices: CPU, iGPU
- Helper: `../../helpers/indicwav2vec_transcribe.py`

## devices.txt

```
CPU
GPU
```

## Fetching the weights + converting

```
huggingface-cli download sumanpaudel1997/nepali-asr-indicwav2vec \
  --local-dir /tmp/indicwav2vec-src

python3 ../../tools/nepali-convert/convert_onnx_then_ov.py \
  /tmp/indicwav2vec-src
```

The output is `model.xml`, `model.bin`, `decoder/lm_head.npy`, `vocab.json`.
Drop them into this directory.
