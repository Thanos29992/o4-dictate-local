# parakeet-v3

Parakeet TDT 0.6B v3, Q8_0 GGUF, runs via libtranscribe + ggml Vulkan.

- HF source: `istupakov/parakeet-tdt-0.6b-v3-onnx` → converted to Q8_0 GGUF
- Devices: iGPU (Vulkan)
- Helper: `../../daemon/parakeet_stream_transcribe.cpp`

## devices.txt

```
GPU
```

## Fetching the weights

```
huggingface-cli download istupakov/parakeet-tdt-0.6b-v3-onnx \
  --local-dir /tmp/parakeet-src

# Convert ONNX → Q8_0 GGUF using Handy's tools (see Handy.md for the recipe).
# Drop the resulting parakeet-unified-en-0.6b-Q8_0.gguf into this directory.
```

## Streaming

The helper is true streaming-capable (`transcribe_parakeet_buffered_stream_ext`).
The daemon currently invokes the one-shot path. See the roadmap.
