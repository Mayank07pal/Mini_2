"""
Demucs Model Quantization for Speech Denoising
==============================================
This script loads a pretrained Demucs model and applies:
- Dynamic quantization to LSTM and Linear layers (INT8)
- Optional FP16 half-precision conversion
- Inference on noisy audio
- Model compression

Why this approach:
- Conv1D layers are not fully supported by torch.quantization, so they're skipped
- LSTM and Linear layers benefit most from INT8 quantization
- FP16 is optional for extreme inference speed (but may lose some precision)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.dsp import convert_audio


def compute_snr_np(clean: np.ndarray, est: np.ndarray) -> float:
    """Compute SNR between clean and estimated signals."""
    est_scaled = est * (np.sum(clean * est) / (np.sum(est ** 2) + 1e-8))
    noise = clean - est_scaled
    num = np.sum(clean ** 2) + 1e-8
    den = np.sum(noise ** 2) + 1e-8
    return float(10.0 * np.log10(num / den))


def compute_si_sdr_np(clean: np.ndarray, est: np.ndarray) -> float:
    """Compute SI-SDR between clean and estimated signals."""
    clean = clean - np.mean(clean)
    est = est - np.mean(est)
    alpha = np.sum(est * clean) / (np.sum(clean ** 2) + 1e-8)
    proj = alpha * clean
    noise = est - proj
    ratio = (np.sum(proj ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return float(10.0 * np.log10(ratio))


def try_import_metrics() -> Tuple[Optional[callable], Optional[callable]]:
    """Optionally import PESQ and STOI metrics."""
    pesq_fn = None
    stoi_fn = None
    try:
        from pesq import pesq as pesq_fn_import
        pesq_fn = pesq_fn_import
    except Exception:
        pesq_fn = None

    try:
        from pystoi import stoi as stoi_fn_import
        stoi_fn = stoi_fn_import
    except Exception:
        stoi_fn = None

    return pesq_fn, stoi_fn


def quantize_demucs_model(
    model: torch.nn.Module,
    apply_fp16: bool = False,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Apply dynamic quantization to Demucs model.
    
    This function:
    1. Identifies LSTM and Linear layers (which support INT8 quantization)
    2. Skips Conv1D layers (not fully supported in torch.quantization)
    3. Applies dynamic quantization to supported layers
    4. Optionally converts model to FP16
    
    Args:
        model: Pretrained Demucs model
        apply_fp16: Whether to also convert to FP16 (half precision)
        verbose: Print quantization details
    
    Returns:
        quantized_model: Model with INT8 quantization applied
        quant_info: Dictionary with quantization statistics
    """
    
    quant_info = {
        "original_layers": {},
        "quantized_layers": [],
        "skipped_layers": [],
        "total_params_original": 0,
        "total_params_quantized": 0,
    }
    
    # Count original parameters
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LSTM, torch.nn.Linear)):
            for param in module.parameters():
                quant_info["total_params_original"] += param.numel()
    
    if verbose:
        print("\n" + "="*70)
        print("QUANTIZATION ANALYSIS")
        print("="*70)
        print(f"\nOriginal model layers:")
    
    # Identify which layers will be quantized
    lstm_layers = []
    linear_layers = []
    conv_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
            lstm_layers.append(name)
            quant_info["quantized_layers"].append((name, "LSTM"))
            if verbose:
                print(f"  [QUANTIZE] LSTM: {name}")
        elif isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
            quant_info["quantized_layers"].append((name, "Linear"))
            if verbose:
                print(f"  [QUANTIZE] Linear: {name}")
        elif isinstance(module, torch.nn.Conv1d):
            conv_layers.append(name)
            quant_info["skipped_layers"].append((name, "Conv1D"))
            if verbose:
                print(f"  [SKIP]     Conv1D: {name} (not fully supported)")
    
    if verbose:
        print(f"\nSummary:")
        print(f"  - LSTM layers to quantize: {len(lstm_layers)}")
        print(f"  - Linear layers to quantize: {len(linear_layers)}")
        print(f"  - Conv1D layers skipped: {len(conv_layers)}")
    
    # Apply dynamic quantization
    # Dynamic quantization works on activations at runtime
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.LSTM, torch.nn.Linear},  # Quantize these layer types
        dtype=torch.qint8,  # 8-bit integer
    )
    
    if verbose:
        print(f"\n✓ Dynamic INT8 quantization applied to LSTM and Linear layers")
    
    # Count quantized parameters (estimate: ~1/4 original size for INT8)
    quant_info["total_params_quantized"] = quant_info["total_params_original"] // 4
    
    # Optional FP16 conversion
    if apply_fp16:
        quantized_model = quantized_model.half()
        if verbose:
            print(f"✓ Model converted to FP16 (half precision)")
        quant_info["fp16_applied"] = True
    else:
        quant_info["fp16_applied"] = False
    
    # Calculate size reduction
    size_reduction = (1 - quant_info["total_params_quantized"] / max(quant_info["total_params_original"], 1)) * 100
    if verbose:
        print(f"\nExpected size reduction: {size_reduction:.1f}%")
        print("="*70 + "\n")
    
    return quantized_model, quant_info


def denoise_audio(
    noisy_audio: np.ndarray,
    model: torch.nn.Module,
    sr: int = 16000,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Run inference on noisy audio using Demucs model.
    
    Args:
        noisy_audio: Noisy audio waveform (numpy array)
        model: Demucs model (original or quantized)
        sr: Sample rate
        device: Device to run inference on
    
    Returns:
        denoised_audio: Denoised audio waveform (numpy array)
    """
    # Convert to tensor
    noisy_t = torch.from_numpy(noisy_audio).float()
    if noisy_t.dim() == 1:
        noisy_t = noisy_t.unsqueeze(0)
    
    # Convert to model format (mono, 16kHz)
    noisy_model = convert_audio(noisy_t, sr, 16000, 1).to(device)
    
    # Run inference
    with torch.no_grad():
        denoised = model(noisy_model.unsqueeze(0)).squeeze(0)
    
    # Convert back to numpy
    denoised_np = denoised.cpu().squeeze().numpy()
    
    return denoised_np.astype(np.float32)


def benchmark_model(
    model_original: torch.nn.Module,
    model_quantized: torch.nn.Module,
    test_audio: torch.Tensor,
    num_runs: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Benchmark original vs quantized model inference speed.
    
    Args:
        model_original: Original model
        model_quantized: Quantized model
        test_audio: Test audio tensor
        num_runs: Number of inference runs
        device: Device to run on
    
    Returns:
        benchmark_results: Dictionary with timing results
    """
    import time
    
    benchmark_results = {}
    
    # Warm up
    with torch.no_grad():
        _ = model_original(test_audio.to(device))
        _ = model_quantized(test_audio.to(device))
    
    # Benchmark original
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_original(test_audio.to(device))
    torch.cuda.synchronize() if device.type == "cuda" else None
    time_original = (time.perf_counter() - start) / num_runs * 1000  # Convert to ms
    
    # Benchmark quantized
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_quantized(test_audio.to(device))
    torch.cuda.synchronize() if device.type == "cuda" else None
    time_quantized = (time.perf_counter() - start) / num_runs * 1000  # Convert to ms
    
    speedup = time_original / max(time_quantized, 1e-8)
    
    benchmark_results["original_ms"] = time_original
    benchmark_results["quantized_ms"] = time_quantized
    benchmark_results["speedup"] = speedup
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Demucs model for faster inference and smaller size"
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        default=None,
        help="Path to noisy audio file for denoising",
    )
    parser.add_argument(
        "--clean-audio",
        type=str,
        default=None,
        help="Path to clean reference audio for quality comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quantized_models",
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--apply-fp16",
        action="store_true",
        help="Also convert model to FP16 (half precision)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark between original and quantized",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DEMUCS MODEL QUANTIZATION PIPELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Apply FP16: {args.apply_fp16}")
    print(f"Output directory: {output_dir}")
    
    # ============================================================================
    # Step 1: Load original model
    # ============================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading original Demucs model")
    print("-"*70)
    
    model_original = get_model("dns64").to(device)
    model_original.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model_original.parameters())
    print(f"✓ Loaded Demucs DNS64 model")
    print(f"  Total parameters: {total_params:,}")
    
    # ============================================================================
    # Step 2: Apply quantization
    # ============================================================================
    print("\n" + "-"*70)
    print("STEP 2: Applying quantization")
    print("-"*70)
    
    model_quantized, quant_info = quantize_demucs_model(
        model_original,
        apply_fp16=args.apply_fp16,
        verbose=True,
    )
    
    # ============================================================================
    # Step 3: Run inference (if audio provided)
    # ============================================================================
    if args.input_audio:
        print("\n" + "-"*70)
        print("STEP 3: Running inference on audio")
        print("-"*70)
        
        input_path = Path(args.input_audio)
        if not input_path.exists():
            print(f"ERROR: Input audio not found: {input_path}")
            return
        
        # Load audio
        noisy_wav, sr = sf.read(str(input_path))
        if noisy_wav.ndim > 1:
            noisy_wav = np.mean(noisy_wav, axis=1)
        
        print(f"✓ Loaded audio: {input_path.name}")
        print(f"  Duration: {len(noisy_wav) / sr:.2f}s @ {sr}Hz")
        
        # Denoise with original model
        print("\n  Denoising with original model...")
        denoised_original = denoise_audio(
            noisy_wav, model_original, sr=sr, device=device
        )
        
        # Denoise with quantized model
        print("  Denoising with quantized model...")
        denoised_quantized = denoise_audio(
            noisy_wav, model_quantized, sr=sr, device=device
        )
        
        # Save outputs
        output_noisy = output_dir / "noisy.wav"
        output_original = output_dir / "denoised_original.wav"
        output_quantized = output_dir / "denoised_quantized.wav"
        
        sf.write(str(output_noisy), noisy_wav, sr)
        sf.write(str(output_original), denoised_original, sr)
        sf.write(str(output_quantized), denoised_quantized, sr)
        
        print(f"✓ Saved outputs:")
        print(f"  - {output_noisy}")
        print(f"  - {output_original}")
        print(f"  - {output_quantized}")
        
        # ====================================================================
        # Step 4: Quality comparison (if clean audio provided)
        # ====================================================================
        if args.clean_audio:
            print("\n" + "-"*70)
            print("STEP 4: Quality metrics comparison")
            print("-"*70)
            
            clean_path = Path(args.clean_audio)
            if not clean_path.exists():
                print(f"WARNING: Clean audio not found: {clean_path}")
            else:
                clean_wav, clean_sr = sf.read(str(clean_path))
                if clean_wav.ndim > 1:
                    clean_wav = np.mean(clean_wav, axis=1)
                
                # Resample if needed
                if clean_sr != sr:
                    resampler = torchaudio.transforms.Resample(clean_sr, sr)
                    clean_wav = resampler(torch.from_numpy(clean_wav).float()).numpy()
                
                # Trim to same length
                min_len = min(len(clean_wav), len(denoised_original), len(denoised_quantized))
                clean_wav = clean_wav[:min_len]
                denoised_original = denoised_original[:min_len]
                denoised_quantized = denoised_quantized[:min_len]
                noisy_wav_trimmed = noisy_wav[:min_len]
                
                pesq_fn, stoi_fn = try_import_metrics()
                
                print("\nMetrics (higher is better):")
                print(f"{'Metric':<15} {'Noisy':<12} {'Original':<12} {'Quantized':<12}")
                print("-" * 51)
                
                # SNR
                snr_noisy = compute_snr_np(clean_wav, noisy_wav_trimmed)
                snr_original = compute_snr_np(clean_wav, denoised_original)
                snr_quantized = compute_snr_np(clean_wav, denoised_quantized)
                print(f"{'SNR (dB)':<15} {snr_noisy:<12.4f} {snr_original:<12.4f} {snr_quantized:<12.4f}")
                
                # SI-SDR
                si_sdr_noisy = compute_si_sdr_np(clean_wav, noisy_wav_trimmed)
                si_sdr_original = compute_si_sdr_np(clean_wav, denoised_original)
                si_sdr_quantized = compute_si_sdr_np(clean_wav, denoised_quantized)
                print(f"{'SI-SDR (dB)':<15} {si_sdr_noisy:<12.4f} {si_sdr_original:<12.4f} {si_sdr_quantized:<12.4f}")
                
                # STOI (if available)
                if stoi_fn:
                    try:
                        stoi_noisy = stoi_fn(clean_wav, noisy_wav_trimmed, sr, extended=False)
                        stoi_original = stoi_fn(clean_wav, denoised_original, sr, extended=False)
                        stoi_quantized = stoi_fn(clean_wav, denoised_quantized, sr, extended=False)
                        print(f"{'STOI':<15} {stoi_noisy:<12.4f} {stoi_original:<12.4f} {stoi_quantized:<12.4f}")
                    except Exception as e:
                        print(f"  STOI: Error - {e}")
                
                # PESQ (if available)
                if pesq_fn and sr in (8000, 16000):
                    try:
                        mode = "wb" if sr == 16000 else "nb"
                        pesq_noisy = pesq_fn(sr, clean_wav, noisy_wav_trimmed, mode)
                        pesq_original = pesq_fn(sr, clean_wav, denoised_original, mode)
                        pesq_quantized = pesq_fn(sr, clean_wav, denoised_quantized, mode)
                        print(f"{'PESQ':<15} {pesq_noisy:<12.4f} {pesq_original:<12.4f} {pesq_quantized:<12.4f}")
                    except Exception as e:
                        print(f"  PESQ: Error - {e}")
                
                print("\nInterpretation:")
                print("  - Quantized should have similar metrics to original")
                print("  - Small differences (<5%) are acceptable")
                print("  - If SNR/SI-SDR drops significantly, reduce quantization")
    
    # ============================================================================
    # Step 5: Benchmark (if requested)
    # ============================================================================
    if args.benchmark:
        print("\n" + "-"*70)
        print("STEP 5: Performance benchmark")
        print("-"*70)
        
        # Create dummy audio for benchmarking
        dummy_audio = torch.randn(1, 2, 16000 * 5).to(device)  # 5 seconds stereo
        
        print("\nBenchmarking inference speed (10 runs)...")
        benchmark_results = benchmark_model(
            model_original,
            model_quantized,
            dummy_audio,
            num_runs=10,
            device=device,
        )
        
        print(f"\nInference time per 5-second audio:")
        print(f"  Original:  {benchmark_results['original_ms']:.2f} ms")
        print(f"  Quantized: {benchmark_results['quantized_ms']:.2f} ms")
        print(f"  Speedup:   {benchmark_results['speedup']:.2f}x")
    
    # ============================================================================
    # Step 6: Save quantized model
    # ============================================================================
    print("\n" + "-"*70)
    print("STEP 6: Saving quantized model")
    print("-"*70)
    
    model_path = output_dir / "demucs_quantized.pt"
    torch.save(model_quantized, str(model_path))
    print(f"✓ Saved quantized model: {model_path}")
    
    # Save quantization info
    info_path = output_dir / "quantization_info.json"
    with open(info_path, "w") as f:
        # Convert tuples to strings for JSON serialization
        quant_info_serializable = {
            k: [list(item) if isinstance(item, tuple) else item for item in v]
            if isinstance(v, list) else v
            for k, v in quant_info.items()
        }
        json.dump(quant_info_serializable, f, indent=2)
    print(f"✓ Saved quantization info: {info_path}")
    
    print("\n" + "="*70)
    print("QUANTIZATION COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  - INT8 quantization applied to {len(quant_info['quantized_layers'])} layers")
    print(f"  - {len(quant_info['skipped_layers'])} Conv1D layers skipped")
    print(f"  - Expected model size reduction: ~75% (4x compression)")
    print(f"  - Output directory: {output_dir}")


if __name__ == "__main__":
    main()
