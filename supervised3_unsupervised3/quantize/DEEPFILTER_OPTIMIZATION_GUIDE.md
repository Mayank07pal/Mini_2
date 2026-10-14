# DeepFilterNet Optimization Guide

## Overview

DeepFilterNet is a hybrid DSP + neural network model for real-time speech denoising. Unlike pure CNN/LSTM models, it requires careful optimization to avoid degrading audio quality.

**Key Principle:** Optimize the neural component while preserving DSP pipeline precision.

---

## Architecture: DSP + Neural Hybrid

### Full Pipeline

```
Input Audio (mono, 16kHz)
    ↓
┌─────────────────────────────────┐
│ DSP: STFT Analysis              │ ← MUST be FP32
│ • FFT transformation            │
│ • Magnitude + Phase extraction  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Neural: DNN Inference           │ ← Can be FP16/ONNX
│ • Process magnitude spectrum    │
│ • Predict frequency gain        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ DSP: Phase Preservation         │ ← MUST be FP32
│ • Keep original phase           │
│ • Apply gain to magnitude       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ DSP: iSTFT Synthesis            │ ← MUST be FP32
│ • Inverse FFT transformation    │
│ • Reconstruct audio waveform    │
└─────────────────────────────────┘
    ↓
Output Audio (denoised, 16kHz)
```

### Component Sizes (Approximate)

| Component | Parameters | % of Model |
|-----------|------------|-----------|
| DNN (GRU + Linear) | 150K - 500K | 1-5% |
| DSP (STFT/iSTFT) | Algorithmic (no params) | - |
| **Total** | 150K - 500K | 100% |

**Key Observation:** DNN is only 1-5% of model! Most of DeepFilterNet is DSP computation.

---

## Why DeepFilterNet is Sensitive to Quantization

### 1. Phase Information is Critical

**The Problem:**
- Audio quality depends heavily on phase accuracy
- Phase errors create artifacts more noticeable than magnitude errors
- Quantization introduces phase errors in STFT/iSTFT

**Example:**
```
Original phase: π/4 (exact)
After FP32:     3.14159... (99.9999% accurate)
After FP16:     3.141... (99.9% accurate)
After INT8:     3.0 (96% accurate)
                ↓
STFT inversion error: Phase wrapping artifacts
```

### 2. STFT/iSTFT Numerical Stability

**Why DSP must stay FP32:**
- FFT is numerically sensitive operation
- Inverse FFT amplifies errors
- Phase tracking requires high precision
- Window overlap-add is cumulative error prone

**Typical Phase Error Impact:**
```
FP32 → FP16: Phase error ~0.01% → Imperceptible artifacts
FP32 → INT8: Phase error ~1-2%  → Clear whistling, clicks
```

### 3. Gain Estimation Robustness

**Why neural component can use FP16:**
- DNN output is gain value (typically 0.0 to 1.0)
- Gain range is limited and well-behaved
- Output is robust to quantization
- Supervised learning already handles noise

**Gain Estimation Examples:**
```
Clean magnitude:    1.0  →  DNN → 1.0  ✓ Keep it
Noisy magnitude:    1.5  →  DNN → 0.8  ✓ Apply gain
Noise-dominated:    0.1  →  DNN → 0.2  ✓ Reduce it
```

---

## Safe vs. Unsafe Optimization

### ✓ SAFE OPTIMIZATIONS

1. **FP16 on DNN Weights Only**
   - Reduces DNN memory by 50%
   - Inference 10-20% faster on GPU
   - Imperceptible audio quality difference (<0.1dB)
   - **Recommendation:** ✓ Do this

2. **ONNX Export for DNN**
   - Pure neural network inference
   - 30-50% faster on CPU
   - Separate from DSP (no coupling)
   - **Recommendation:** ✓ Do this

3. **Batch Processing**
   - Multiple audios in single forward pass
   - Better GPU utilization
   - No quality impact
   - **Recommendation:** ✓ Do this

### ✗ UNSAFE OPTIMIZATIONS

1. **INT8 Quantization**
   - Would quantize STFT/iSTFT
   - Causes phase artifacts
   - Noticeable "metallic" distortion
   - SNR loss: 2-5dB
   - **Recommendation:** ✗ Don't do this

2. **FP16 for STFT/iSTFT**
   - FFT becomes numerically unstable
   - Phase wrapping errors accumulate
   - Creates musical artifacts
   - SNR loss: 1-3dB
   - **Recommendation:** ✗ Don't do this

3. **Aggressive DNN Quantization**
   - INT8 on neural network
   - Would require retraining
   - Coupling with DSP pipeline critical
   - **Recommendation:** ✗ Don't do this without retraining

---

## Recommended Optimization Strategy

### Tier 1: Safe + Simple (Recommended)

```python
# 1. Apply FP16 to DNN weights
model = model.half()

# 2. Use standard enhance() pipeline
# DSP stays FP32, DNN runs in FP16

# Results:
# • 10-20% faster inference on GPU
# • <0.1dB audio quality loss
# • No retraining needed
# • Easy deployment
```

**Performance:**
- Speedup: 1.1-1.2x
- Quality Loss: <0.1dB
- Implementation: 1 line of code

---

### Tier 2: Balanced (Production)

```python
# 1. Export neural components to ONNX
torch.onnx.export(
    neural_net,
    sample_input,
    "dnn.onnx",
    opset_version=12
)

# 2. Use ONNX Runtime for DNN inference
session = ort.InferenceSession("dnn.onnx")
dnn_output = session.run(None, {"input": features})

# 3. Keep DSP in PyTorch FP32
stft_output = model.stft(audio)
filtered = dnn_output * stft_output
audio_out = model.istft(filtered, phase)

# Results:
# • 30-50% faster DNN inference on CPU
# • Portable across platforms
# • Full DSP precision maintained
# • Hybrid optimization
```

**Performance:**
- Speedup: 1.3-1.5x (especially CPU)
- Quality Loss: <0.1dB
- Implementation: Medium complexity

---

### Tier 3: Maximum Performance (GPU-Only)

```python
# 1. Use NVIDIA TensorRT
engine = trt.create_engine(model, precision=trt.float16)

# 2. Run full model in TensorRT
inference = TRTInference(engine)
output = inference(audio)

# 3. Results:
# • 2-3x faster on NVIDIA GPU
# • Still maintains DSP accuracy
# • GPU-only (no CPU support)

# NOTE: Requires TensorRT installation and conversion
```

**Performance:**
- Speedup: 2-3x (NVIDIA GPU only)
- Quality Loss: <0.1dB
- Implementation: Complex

---

## Benchmarking Guide

### Latency Benchmarking

```python
import time
import numpy as np

# Warmup (important for GPU)
for _ in range(3):
    model.enhance(audio)

# Measure latency
latencies = []
for _ in range(100):
    start = time.time()
    output = model.enhance(audio)
    latencies.append((time.time() - start) * 1000)  # ms

# Analyze
latency_mean = np.mean(latencies)
latency_std = np.std(latencies)
latency_p95 = np.percentile(latencies, 95)

print(f"Latency: {latency_mean:.2f} ± {latency_std:.2f} ms")
print(f"P95: {latency_p95:.2f} ms")

# Compute Real-Time Factor (RTF)
audio_duration = len(audio) / 16000  # seconds
rtf = latency_mean / (audio_duration * 1000)
print(f"Real-Time Factor: {rtf:.2f}x")  # <1 = real-time capable
```

### Quality Benchmarking

```python
from scipy import signal
import numpy as np

def compute_snr(reference, degraded):
    """Signal-to-Noise Ratio (higher is better)"""
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - degraded) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_si_sdr(reference, degraded):
    """Scale-Invariant Signal-to-Distortion Ratio"""
    # Normalize
    ref_norm = reference / (np.linalg.norm(reference) + 1e-8)
    deg_norm = degraded / (np.linalg.norm(degraded) + 1e-8)
    
    # Optimal scaling
    s = np.dot(ref_norm, deg_norm) / np.dot(ref_norm, ref_norm)
    
    # SI-SDR
    error = deg_norm - s * ref_norm
    return 10 * np.log10(np.sum((s * ref_norm) ** 2) / (np.sum(error ** 2) + 1e-8))

# Benchmark both versions
output_fp32 = model_fp32.enhance(noisy_audio)
output_fp16 = model_fp16.enhance(noisy_audio)

snr_fp32 = compute_snr(clean_audio, output_fp32)
snr_fp16 = compute_snr(clean_audio, output_fp16)
snr_loss = snr_fp32 - snr_fp16

print(f"SNR FP32:  {snr_fp32:.2f} dB")
print(f"SNR FP16:  {snr_fp16:.2f} dB")
print(f"Loss:      {snr_loss:.4f} dB (should be <0.1)")
```

### Profiling Individual Components

```python
import torch.profiler as profiler

# Profile STFT
with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
    stft_output = model.stft(audio)

print(f"STFT time: {prof.total_average():.2f}ms")

# Profile DNN
with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
    dnn_output = model.dnn(stft_output)

print(f"DNN time: {prof.total_average():.2f}ms")

# Profile iSTFT
with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
    output = model.istft(dnn_output, phase)

print(f"iSTFT time: {prof.total_average():.2f}ms")
```

---

## Performance Expectations

### Speedup by Optimization

| Optimization | Latency Improvement | Quality Loss | Notes |
|--------------|-------------------|--------------|-------|
| None (FP32 baseline) | 1.0x | - | Baseline |
| FP16 on DNN | 1.1-1.2x | <0.1dB | GPU Tensor Cores |
| ONNX DNN (CPU) | 1.3-1.5x | <0.1dB | ONNX Runtime optimization |
| ONNX DNN (GPU) | 1.2-1.4x | <0.1dB | Less benefit on GPU |
| TensorRT (NVIDIA) | 2-3x | <0.1dB | Most speedup, GPU-only |

### Real-Time Capability

| Hardware | FP32 (1s audio) | FP16 (1s audio) | Real-Time? |
|----------|---|---|---|
| CPU (Intel i7) | 200-400ms | 150-300ms | ✓ Yes |
| GPU (RTX 3060) | 10-20ms | 8-15ms | ✓✓ Yes |
| GPU (RTX 4090) | 5-10ms | 3-7ms | ✓✓✓ Yes |
| Mobile (ARM) | 500-1000ms | 400-800ms | ✗ No (or chunked) |

**Real-Time Requirement:** Latency < 100ms for 1 second audio (or RTF < 0.1)

---

## Audio Quality Expectations

### Quality Metrics

| Metric | Baseline | After FP16 | Acceptable Range |
|--------|----------|-----------|-----------------|
| SNR (dB) | 15.5 | 15.4 (-0.1) | ±0.5dB acceptable |
| SI-SDR (dB) | 14.2 | 14.1 (-0.1) | ±0.5dB acceptable |
| PESQ | 3.2 | 3.1 (-0.05) | ±0.1 acceptable |

### Listening Test Expectations

- **FP32 vs FP16:** Imperceptible difference (if any)
- **FP32 vs INT8:** Clear distortion (if attempted)
- **Phase Artifacts:** Only visible with aggressive quantization

---

## Troubleshooting

### Issue: Poor Audio Quality After Optimization

**Symptoms:** Noticeable artifacts, "robotic" sound, clicks

**Likely Causes:**
1. Applied quantization to DSP pipeline (STFT/iSTFT)
2. Used INT8 instead of FP16
3. Model state not properly loaded

**Solutions:**
```python
# Check model precision
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

# Verify DSP is FP32
model.dsp_module.float()  # Force DSP to FP32

# Keep only DNN in FP16
model.dnn.half()
```

---

### Issue: Slow Inference Despite Optimization

**Symptoms:** ONNX not faster than PyTorch

**Likely Causes:**
1. Data transfer overhead (GPU → CPU → GPU)
2. Wrong provider selected
3. Model still on GPU

**Solutions:**
```python
# Check ONNX providers
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

# Use correct provider
providers = [
    'CUDAExecutionProvider',    # GPU
    'CPUExecutionProvider'      # CPU fallback
]

# Minimize data transfer
# Keep model on same device for processing
```

---

### Issue: ONNX Export Fails

**Symptoms:** `torch.onnx.export()` throws error

**Likely Causes:**
1. Unsupported operations in ONNX
2. Sample input shape mismatch
3. Model has control flow

**Solutions:**
```python
# Use lower opset version
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    opset_version=11  # Try 11 or 12
)

# Verify model can be traced
traced = torch.jit.trace(model, sample_input)
print("Model successfully traced")

# Check for problematic operations
print(model)  # Review architecture
```

---

## Recommended Configuration

### Development & Testing
```python
model = DeepFilterNet()
model.eval()
# Use FP32 for debugging
```

### Production (CPU)
```python
# Export to ONNX
torch.onnx.export(model.dnn, ...)

# Use ONNX Runtime with CPU provider
session = ort.InferenceSession(
    "model.onnx",
    providers=['CPUExecutionProvider']
)
```

### Production (GPU)
```python
# Apply FP16
model = model.half()

# Use PyTorch with CUDA
model = model.cuda()
output = model.enhance(audio.cuda())
```

### Real-Time Processing
```python
# Process in chunks (e.g., 512 samples = 32ms at 16kHz)
chunk_size = 512
for chunk in chunks:
    output_chunk = model.enhance(chunk)
    yield output_chunk
```

---

## Summary Table

| Aspect | Recommendation | Why |
|--------|---|---|
| **Quantization** | FP16 DNN only | Phase precision critical |
| **DSP Precision** | Always FP32 | STFT/iSTFT numerical stability |
| **Export** | ONNX for DNN | 30-50% speedup potential |
| **Benchmarking** | Measure RTF + SNR | Both latency and quality matter |
| **Deployment** | Hybrid DSP+ONNX | Best balance of speed & quality |

---

## References

- [DeepFilterNet Repository](https://github.com/Rikorose/DeepFilterNet)
- [STFT Numerical Properties](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
- [Phase in Speech Processing](https://www.dsprelated.com/freebooks/sasp/Spectrum_Analysis_Windows.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)

---

**Key Takeaway:** Optimize DeepFilterNet's neural component (FP16/ONNX) while preserving DSP pipeline precision (FP32). This hybrid approach provides 30-50% speedup with imperceptible audio quality loss.
