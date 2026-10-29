#!/bin/sh
# Build the CUDA trainer (MSVC host compiler, sm_86 / RTX 3080)
set -e
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xcompiler=-O2,-EHsc -o build/${OUT:-train}.exe src/train.cu -lcublas
echo "built build/${OUT:-train}.exe"
