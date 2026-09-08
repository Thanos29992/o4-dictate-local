#!/bin/bash
set -e
BIN=/home/shlok/.local/bin/dictate
g++ -O2 -std=c++17 -o "$BIN" /home/shlok/.local/share/npu-asr/src/dictate.cpp -I/usr/include/openvino -lopenvino -lfftw3f -lsndfile -lpthread
chmod +x "$BIN"
echo "BUILD_OK"
