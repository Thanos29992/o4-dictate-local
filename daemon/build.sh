#!/bin/bash
set -e
BIN=/home/shlok/.local/bin/dictate
GGBASE=/home/shlok/.local/share/npu-asr/ggml
g++ -O2 -std=c++17 -Wall -Wno-unused-parameter \
  -I/usr/include/openvino \
  -I$GGBASE \
  -I/home/shlok/.local/share/npu-asr/src \
  /home/shlok/.local/share/npu-asr/src/dictate.cpp \
  -L$GGBASE \
  -Wl,-rpath,$GGBASE \
  -ltranscribe -lggml-vulkan -lggml-base -lggml -lopenblas \
  -lopenvino -lfftw3f -lsndfile -lpthread \
  -o "$BIN"
chmod +x "$BIN"

# Build RT streaming helper (reads f32 PCM from stdin)
RT_BIN=/home/shlok/.local/bin/parakeet_stream_transcribe_rt
g++ -O2 -std=c++17 -Wall -Wno-unused-parameter \
  -I$GGBASE \
  -I/home/shlok/.local/share/npu-asr/src \
  /home/shlok/.local/share/npu-asr/src/parakeet_stream_transcribe_rt.cpp \
  -L$GGBASE \
  -Wl,-rpath,$GGBASE \
  -ltranscribe -lggml-vulkan -lggml-base -lggml -lopenblas \
  -lfftw3f -lpthread \
  -o "$RT_BIN"
chmod +x "$RT_BIN"
echo "RT_HELPER_BUILT"
echo "BUILD_OK"
