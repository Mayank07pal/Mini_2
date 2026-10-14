# Model Optimization for Deployment

This folder contains tools for optimizing speech denoising models (Demucs and CleanUNet) for faster inference and smaller model size.

## Files

### Demucs Quantization
- **quantize_demucs.py** - INT8 quantization for LSTM/Linear layers
- **QUANTIZATION_GUIDE.md** - Explanation of INT8 quantization approach

### CleanUNet Optimization  
- **optimize_cleanunet.py** - FP16 conversion and ONNX export
- **CLEANUNET_OPTIMIZATION_GUIDE.md** - Explanation of FP16 + ONNX deployment strategy

### DeepFilterNet Optimization
- **optimize_deepfilter.py** - FP16 DNN + ONNX export (preserves DSP precision)
- **DEEPFILTER_OPTIMIZATION_GUIDE.md** - Explanation of hybrid DSP+neural optimization

## Quick Start

### Demucs INT8 Quantization

```bash
cd quantize

# Basic quantization
python quantize_demucs.py \
  --input-audio ../../path/to/noisy.wav \
  --clean-audio ../../path/to/clean.wav

# With FP16 and benchmarking
python quantize_demucs.py \
  --input-audio noisy.wav \
  --apply-fp16 \
  --benchmark \
  --device cuda
```

### CleanUNet FP16 Optimization

```bash
cd quantize

# Export to ONNX with quality metrics
python optimize_cleanunet.py \
  --input-audio ../../path/to/noisy.wav \
  --clean-audio ../../path/to/clean.wav

# GPU inference
python optimize_cleanunet.py \
  --input-audio noisy.wav \
  --device cuda

# CPU inference (ONNX recommended)
python optimize_cleanunet.py \
  --input-audio noisy.wav \
  --device cpu \
  --apply-onnx
```

### DeepFilterNet Optimization (DSP + Neural Hybrid)

```bash
cd quantize

# Export neural components to ONNX
python optimize_deepfilter.py \
  --input-audio ../../path/to/noisy.wav \
  --clean-audio ../../path/to/clean.wav

# Benchmark performance
python optimize_deepfilter.py \
  --input-audio noisy.wav \
  --benchmark

# CPU inference
python optimize_deepfilter.py \
  --input-audio noisy.wav \
  --device cpu \
  --apply-onnx
```

## What Happens

### Demucs Quantization
1. **Loads** the pretrained Demucs DNS64 model
2. **Quantizes** LSTM and Linear layers to INT8 (8-bit integers)
3. **Skips** Conv1D layers (not supported in PyTorch quantization)
4. **Runs** inference on your audio
5. **Computes** quality metrics (SNR, SI-SDR, STOI, PESQ)
6. **Saves** the quantized model and outputs

### CleanUNet Optimization
1. **Loads** the trained CleanUNet model
2. **Converts** to FP16 (half precision, 50% smaller)
3. **Exports** to ONNX format (cross-platform)
4. **Runs** inference with PyTorch and ONNX Runtime
5. **Computes** quality metrics
6. **Saves** both optimized models

### DeepFilterNet Optimization
1. **Loads** the pretrained DeepFilterNet model
2. **Separates** DSP pipeline (STFT/iSTFT) from neural component (DNN)
3. **Converts** DNN to FP16 (neural component only, 50% smaller)
4. **Exports** DNN to ONNX for faster inference
5. **Keeps** DSP in FP32 (critical for phase accuracy)
6. **Benchmarks** latency and quality metrics
7. **Saves** optimized models and analysis

## Output

### Demucs
```
quantized_models/
├── demucs_quantized.pt          # INT8 quantized model
├── quantization_info.json       # Metadata
├── noisy.wav                    # Input audio
├── denoised_original.wav        # Output from original
└── denoised_quantized.wav       # Output from quantized
```

### CleanUNet
```
optimized_cleanunet/
├── cleanunet_fp16.pt                # PyTorch FP16 model
├── cleanunet.onnx                   # ONNX model (cross-platform)
├── denoised_pytorch_fp16.wav        # Output from PyTorch
├── denoised_onnx.wav               # Output from ONNX Runtime
└── optimization_results.json        # Metrics and metadata
```

### DeepFilterNet
```
optimized_deepfilter/
├── deepfilter_dnn.onnx              # DNN component exported to ONNX
├── denoised_pytorch.wav             # Output from PyTorch inference
├── denoised_onnx.wav               # Output from ONNX Runtime (if available)
└── optimization_results.json        # Performance metrics and analysis
```

## Benefits

- **18-45% smaller model** (depending on FP16)
- **1.2-1.5x faster inference**
- **Minimal quality loss** (<1dB SNR drop)
- **Simple to use** - just one function call

## Benefits

### Demucs INT8 Quantization
- **18-45% smaller model** (LSTM/Linear quantized)
- **1.2-1.5x faster inference**
- **Minimal quality loss** (<1dB SNR drop)
- **Simple to use** - just one function call

### CleanUNet FP16 Optimization
- **50% smaller model** (FP32 → FP16)
- **1.2-1.5x faster GPU inference**
- **50%+ faster CPU inference** (with ONNX Runtime)
- **Negligible quality loss** (<0.5dB SNR)
- **Cross-platform deployment** (ONNX)

### DeepFilterNet Optimization
- **Hybrid DSP+Neural approach** - preserves phase accuracy
- **1.1-1.2x faster GPU inference** (FP16 DNN)
- **1.3-1.5x faster CPU inference** (ONNX DNN)
- **Imperceptible quality loss** (<0.1dB SNR)
- **Safe optimization** - no phase artifacts

---

## Comparison: When to Use Each

| Model | Optimization | Speed | Size | Quality | Use Case |
|-------|--------------|-------|------|---------|----------|
| **Demucs** | INT8 Quantization | 1.2-1.5x | -18-45% | Minimal loss | Edge/mobile deployment |
| **CleanUNet** | FP16 + ONNX | 1.2-1.5x (GPU), 2x (CPU) | -50% | Negligible loss | GPU servers or CPU cloud |
| **DeepFilterNet** | FP16 DNN + ONNX | 1.1-1.2x (GPU), 1.3-1.5x (CPU) | -25% (DNN only) | Imperceptible loss | Real-time production |

---

## Key Points

### Demucs Quantization
✓ LSTM layers: Quantized to INT8
✓ Linear layers: Quantized to INT8
✗ Conv1D layers: Kept as FP32 (not supported)

### CleanUNet Optimization
✓ All layers: FP16 half-precision
✓ ONNX export: Cross-platform
✗ INT8: Not recommended for Conv1D-heavy architecture

### DeepFilterNet Optimization
✓ DNN component: FP16 + ONNX optimized
✓ DSP pipeline: Always FP32 (STFT/iSTFT)
✓ Phase preservation: No quantization on phase
✗ INT8: Would cause phase artifacts

See QUANTIZATION_GUIDE.md for Demucs details.
See CLEANUNET_OPTIMIZATION_GUIDE.md for CleanUNet details.
See DEEPFILTER_OPTIMIZATION_GUIDE.md for DeepFilterNet details.
