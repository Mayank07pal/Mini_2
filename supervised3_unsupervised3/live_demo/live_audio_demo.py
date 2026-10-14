"""
Live audio recording with UrbanSound8K noise injection and multi-model denoising.
Records audio from microphone, adds noise from urbansound8k/fold1, and tests on 4 denoising models.
"""

import argparse
import json
import math
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import random
import time
import select

try:
    import msvcrt  # Windows keypress detection
except Exception:
    msvcrt = None

URBANSOUND8K_ROOT = Path(r"C:\Users\mayan\Desktop\Mini Project\Only-Noisy-Training\Datasets\UrbanSound8K")
DEFAULT_LIVEDATA_DIR = Path(__file__).parent / "livedata"


def get_torchaudio():
    """Load torchaudio only when resampling or a model requires it."""
    import torchaudio
    return torchaudio

# Metrics computation functions (shared across all models)
def compute_snr_np(clean: np.ndarray, est: np.ndarray) -> float:
    """Signal-to-Noise Ratio"""
    est_scaled = est * (np.sum(clean * est) / (np.sum(est ** 2) + 1e-8))
    noise = clean - est_scaled
    num = np.sum(clean ** 2) + 1e-8
    den = np.sum(noise ** 2) + 1e-8
    return float(10.0 * np.log10(num / den))


def compute_si_sdr_np(clean: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio"""
    clean = clean - np.mean(clean)
    est = est - np.mean(est)
    alpha = np.sum(est * clean) / (np.sum(clean ** 2) + 1e-8)
    proj = alpha * clean
    noise = est - proj
    ratio = (np.sum(proj ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return float(10.0 * np.log10(ratio))


def try_import_metrics():
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


def record_from_microphone(duration: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Record audio from microphone.

    If `duration` > 0: record for that many seconds (existing behavior).
    If `duration` <= 0: record until the user presses SPACE (Windows).
    Falls back to pressing Enter if space detection isn't available.
    """
    if duration is not None and duration > 0:
        print(f"\n🎤 Recording {duration}s from microphone at {sr}Hz...")
        num_frames = int(duration * sr)
        audio = np.zeros((num_frames, 1), dtype=np.float32)
        frames_recorded = [0]
        
        def callback(indata, frames_count, time_info, status):
            if status:
                pass
            start = frames_recorded[0]
            end = start + frames_count
            if end <= num_frames:
                audio[start:end] = indata[:frames_count]
                frames_recorded[0] = end
            # Real-time level bar
            level = float(np.sqrt(np.mean(indata ** 2)))
            bar_width = 35
            bar_filled = int(min(level * bar_width / 0.5, bar_width))
            bar = "█" * bar_filled
            pct = (end / num_frames) * 100
            print(f"\r[{bar:<{bar_width}}] {level:.4f} | {pct:.0f}%", end="", flush=True)
        
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
            sd.sleep(int(duration * 1000))
        print("\n✓ Recording complete")
        return audio.squeeze()

    # Stream-based recording until keypress
    print(f"\n🎤 Recording until you press SPACE (or Enter) at {sr}Hz...")
    frames = []
    max_level = [0.0]

    def callback(indata, frames_count, time_info, status):
        if status:
            pass
        frames.append(indata.copy())
        level = float(np.sqrt(np.mean(indata ** 2)))
        max_level[0] = max(max_level[0], level)
        bar_width = 35
        bar_filled = int(min(level * bar_width / 0.5, bar_width))
        bar = "█" * bar_filled
        print(f"\r[{bar:<{bar_width}}] {level:.4f} | Frames: {len(frames)}", end="", flush=True)

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
        print("Press SPACE to stop recording...")
        try:
            while True:
                if msvcrt is not None:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        # spacebar is b' '
                        if ch == b' ':
                            break
                else:
                    # Fallback: wait for Enter
                    try:
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            _ = sys.stdin.readline()
                            break
                    except Exception:
                        pass
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n⚠️ Recording interrupted by user")

    if not frames:
        return np.zeros(0, dtype=np.float32)

    audio_np = np.concatenate(frames, axis=0).squeeze()
    print("✓ Recording complete")
    return audio_np


def load_urbansound_noise(sr: int = 16000):
    """Pick a random WAV under `URBANSOUND8K_ROOT/audio`, load and resample to `sr`.

    Returns (noise_wav, noise_sr, source_path) or None on failure.
    """
    try:
        audio_root = URBANSOUND8K_ROOT / "audio"
        if not audio_root.exists():
            print(f"⚠️ UrbanSound8K audio root not found: {audio_root}")
            return None

        wav_paths = list(audio_root.rglob("*.wav"))
        if not wav_paths:
            print(f"⚠️ No WAV files found under: {audio_root}")
            return None

        selected = random.choice(wav_paths)
        noise_wav, noise_sr = sf.read(str(selected))

        if noise_wav.ndim > 1:
            noise_wav = np.mean(noise_wav, axis=1)

        # Resample to requested sr if needed
        if noise_sr != sr:
            torchaudio = get_torchaudio()
            resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=sr)
            noise_wav = resampler(torch.from_numpy(noise_wav).float().unsqueeze(0)).squeeze(0).numpy()
            noise_sr = sr

        print(f"✓ Loaded noise: {selected.name}")
        return noise_wav.astype(np.float32), noise_sr, selected
    except Exception as e:
        print(f"⚠️ Failed to load noise file: {e}")
        return None


def mix_audio(clean: np.ndarray, noise: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Mix clean audio with noise at specified SNR."""
    # Trim noise to match clean length
    if len(noise) > len(clean):
        start_idx = np.random.randint(0, len(noise) - len(clean))
        noise = noise[start_idx:start_idx + len(clean)]
    elif len(noise) < len(clean):
        # Repeat and concatenate noise
        repetitions = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, repetitions)[:len(clean)]
    
    # Compute scaling factor for desired SNR
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    # If noise is silent or extremely low-power, fall back to random noise
    if noise_power < 1e-12:
        print("⚠️ Detected silent/near-zero noise; using generated Gaussian noise instead")
        noise = np.random.normal(0.0, 1.0, size=len(clean)).astype(np.float32)
        noise_power = np.mean(noise ** 2)

    # SNR = 10 * log10(Ps / Pn) => Ps/Pn = 10^(SNR/10)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_scaling = np.sqrt(clean_power / (snr_linear * (noise_power + 1e-8)))

    noisy = clean + noise_scaling * noise
    return noisy.astype(np.float32)


def resample_audio(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if src_sr == target_sr:
        return audio
    torchaudio = get_torchaudio()
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=target_sr)
    audio_t = torch.from_numpy(audio).float()
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)
    resampled = resampler(audio_t).squeeze(0).numpy()
    return resampled.astype(np.float32)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is stereo."""
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    return audio


def trim_pair(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim two arrays to same length."""
    n = min(ref.shape[-1] if ref.ndim > 0 else 0,
            deg.shape[-1] if deg.ndim > 0 else 0)
    return ref[..., :n], deg[..., :n]


def save_audio_visualization(
    clean_audio: np.ndarray,
    input_audio: np.ndarray,
    noise_audio: Optional[np.ndarray],
    top_output: Optional[np.ndarray],
    top_model_name: str,
    sr: int,
    output_path: Path,
) -> None:
    """Save waveforms and spectrograms for input, noise, and the best output."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ Matplotlib is not installed; audio visualization was not saved")
        return

    signals = [
        ("Clean input audio", clean_audio),
        ("Added noise", noise_audio),
        ("Noisy input audio", input_audio),
        (f"Top model output: {top_model_name}", top_output),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)

    for row, (label, signal) in enumerate(signals):
        if signal is None or len(signal) == 0:
            axes[row, 0].text(0.5, 0.5, "No noise added", ha="center", va="center")
            axes[row, 1].text(0.5, 0.5, "No noise added", ha="center", va="center")
            axes[row, 0].set_title(f"{label} waveform")
            axes[row, 1].set_title(f"{label} spectrogram")
            axes[row, 0].set_axis_off()
            axes[row, 1].set_axis_off()
            continue

        signal = np.asarray(signal, dtype=np.float32).squeeze()
        time_axis = np.arange(len(signal)) / sr
        axes[row, 0].plot(time_axis, signal, linewidth=0.5)
        axes[row, 0].set_title(f"{label} waveform")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].specgram(signal, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
        axes[row, 1].set_title(f"{label} spectrogram")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Frequency (Hz)")

    figure.suptitle("Audio Waveforms and Spectrograms", fontsize=16)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"✓ Saved audio visualization: {output_path}")


def mixdown_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    return np.squeeze(audio)


def align_noise_to_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Return a noise array exactly `target_len` long by trimming or repeating."""
    if len(noise) >= target_len:
        start_idx = np.random.randint(0, len(noise) - target_len + 1)
        return noise[start_idx:start_idx + target_len].astype(np.float32)
    else:
        repetitions = (target_len // len(noise)) + 1
        return np.tile(noise, repetitions)[:target_len].astype(np.float32)


# ============================================================================
# Model-specific denoising functions
# ============================================================================

def denoise_with_deepfilter(noisy: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, dict]:
    """Denoise using DeepFilterNet."""
    try:
        # DeepFilterNet expects an older torchaudio path
        fake_module = types.ModuleType("torchaudio.backend.common")
        fake_module.AudioMetaData = object
        sys.modules["torchaudio.backend.common"] = fake_module
        
        from df.enhance import enhance, init_df
        
        print("  → Initializing DeepFilterNet...")
        model, df_state, _ = init_df()
        target_sr = int(df_state.sr())
        
        # Resample if needed
        if sr != target_sr:
            noisy_resampled = resample_audio(noisy, sr, target_sr)
        else:
            noisy_resampled = noisy
        
        # Convert to tensor
        noisy_t = torch.from_numpy(noisy_resampled).float()
        if noisy_t.dim() == 1:
            noisy_t = noisy_t.unsqueeze(0)
        
        with torch.no_grad():
            denoised = enhance(model, df_state, noisy_t)
        
        denoised_np = denoised.cpu().numpy()
        denoised_np = mixdown_to_mono(denoised_np)
        
        # Resample back if needed
        if sr != target_sr:
            denoised_np = resample_audio(denoised_np, target_sr, sr)
        
        return denoised_np.astype(np.float32), {"sr": sr, "model": "deepfilter"}
    except Exception as e:
        print(f"  ⚠️ DeepFilterNet error: {e}")
        return None, {"error": str(e)}


def denoise_with_demucs(noisy: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, dict]:
    """Denoise using Denoiser (DNS48)."""
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        
        print("  → Initializing Denoiser (DNS48)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = pretrained.dns48().to(device)
        model.eval()
        
        # Convert to tensor
        noisy_t = torch.from_numpy(noisy).float()
        if noisy_t.dim() == 1:
            noisy_t = noisy_t.unsqueeze(0)
        
        # Convert audio to model format
        noisy_model = convert_audio(noisy_t, sr, model.sample_rate, model.chin).to(device)
        
        with torch.no_grad():
            denoised = model(noisy_model.unsqueeze(0)).squeeze(0)
        
        denoised_np = denoised.cpu().squeeze().numpy()
        denoised_np = mixdown_to_mono(denoised_np)
        
        # Resample back to original sr if needed
        if sr != model.sample_rate:
            denoised_np = resample_audio(denoised_np, model.sample_rate, sr)
        
        return denoised_np.astype(np.float32), {"sr": sr, "model": "demucs"}
    except Exception as e:
        print(f"  ⚠️ Denoiser (DNS48) error: {e}")
        return None, {"error": str(e)}


def denoise_with_denoiser(noisy: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, dict]:
    """Denoise using Denoiser (DNS48)."""
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        
        print("  → Initializing Denoiser (DNS48)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = pretrained.dns48().to(device)
        model.eval()
        
        # Convert to tensor
        noisy_t = torch.from_numpy(noisy).float()
        if noisy_t.dim() == 1:
            noisy_t = noisy_t.unsqueeze(0)
        
        # Convert audio to model format (16kHz, mono)
        noisy_converted = convert_audio(noisy_t, sr, 16000, 1).to(device)
        
        with torch.no_grad():
            denoised = model(noisy_converted)
        
        denoised_np = denoised.cpu().squeeze().numpy()
        
        # Resample back to original sr if needed
        if sr != 16000:
            denoised_np = resample_audio(denoised_np, 16000, sr)
        
        return denoised_np.astype(np.float32), {"sr": sr, "model": "denoiser"}
    except Exception as e:
        print(f"  ⚠️ Denoiser error: {e}")
        return None, {"error": str(e)}


def denoise_with_cleanunet(noisy: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, dict]:
    """Denoise using CleanUNet."""
    try:
        # Add CleanUNet to path
        cleanunet_path = Path(__file__).parent.parent.parent / "CleanUNet"
        sys.path.insert(0, str(cleanunet_path))
        
        from network import CleanUNet
        
        print("  Initializing CleanUNet...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        config_path = cleanunet_path / "configs" / "DNS-large-high.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load model
        model = CleanUNet(**config["network_config"]).to(device)
        
        # Load checkpoint
        checkpoint_path = cleanunet_path / "exp" / "DNS-large-high" / "checkpoint" / "pretrained.pkl"
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        
        # Extract state dict from checkpoint
        state_dict = checkpoint["model_state_dict"]
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Convert to tensor and resample to 16kHz
        noisy_t = torch.from_numpy(noisy).float()
        if noisy_t.dim() == 1:
            noisy_t = noisy_t.unsqueeze(0)
        
        # Resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            torchaudio = get_torchaudio()
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            noisy_t = resampler(noisy_t)
            model_sr = target_sr
        else:
            model_sr = sr
        
        # Run model
        noisy_batch = noisy_t.unsqueeze(0).to(device)
        with torch.no_grad():
            denoised = model(noisy_batch)
        
        # Handle if model returns tuple
        if isinstance(denoised, tuple):
            denoised = denoised[0]
        
        denoised = denoised.squeeze()
        denoised_np = denoised.detach().cpu().numpy()
        denoised_np = mixdown_to_mono(denoised_np)
        
        # Resample back to original sr if needed
        if sr != model_sr:
            denoised_np = resample_audio(denoised_np, model_sr, sr)
        
        return denoised_np.astype(np.float32), {"sr": sr, "model": "cleanunet"}
    except Exception as e:
        print(f"  CleanUNet error: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}


# ============================================================================
# Main processing function
# ============================================================================

def run_live_demo(
    output_dir: Path,
    recording_duration: float = 5.0,
    sr: int = 16000,
    snr_db: float = 10.0,
    noise_gain_db: float = 0.0,
    mode: str = "add_noise",
) -> dict:
    """Record audio, optionally add noise, and denoise with 4 models."""
    
    output_dir = Path(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Step 1: Record audio from microphone
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Recording from microphone")
    print("=" * 70)
    
    clean_audio = record_from_microphone(duration=recording_duration, sr=sr)
    clean_path = run_dir / "in.wav"
    sf.write(clean_path, clean_audio, sr)
    print(f"✓ Saved clean audio: {clean_path}")
    
    noise_audio = None
    noise_source_path = None
    if mode == "filter_raw":
        print("\n" + "=" * 70)
        print("STEP 2: Filtering raw microphone audio (no noise added)")
        print("=" * 70)
        noisy_audio = clean_audio
        noisy_path = run_dir / "raw_input.wav"
        sf.write(noisy_path, noisy_audio, sr)
        print(f"✓ Saved raw model input: {noisy_path}")
    else:
        # ========================================================================
        # Step 2: Load and mix noise
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 2: Loading UrbanSound8K noise and mixing")
        print("=" * 70)

        noise_res = load_urbansound_noise(sr=sr)
        if noise_res is None:
            print("⚠️ Generating random noise instead")
            noise_audio = np.random.normal(0, 0.1, len(clean_audio)).astype(np.float32)
        else:
            noise_wav, noise_sr, noise_source_path = noise_res
            noise_audio = noise_wav.astype(np.float32)

        if abs(noise_gain_db) > 1e-6:
            gain = 10 ** (noise_gain_db / 20.0)
            noise_audio = (noise_audio * gain).astype(np.float32)

        noisy_audio = mix_audio(clean_audio, noise_audio, snr_db=snr_db)
        noisy_path = run_dir / "noisy.wav"
        sf.write(noisy_path, noisy_audio, sr)
        print(f"✓ Saved noisy audio: {noisy_path}")

        try:
            noise_to_save = align_noise_to_length(noise_audio, len(clean_audio))
            noise_save_path = run_dir / "noise_used.wav"
            sf.write(noise_save_path, noise_to_save, sr)
            print(f"✓ Saved noise used: {noise_save_path}")
        except Exception as e:
            print(f"⚠️ Failed to save noise used: {e}")

    # ============================================================================
    # Step 3: Denoise with 4 models
    # ============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Denoising with 4 models")
    print("=" * 70)
    
    pesq_fn, stoi_fn = try_import_metrics()
    results = {}
    denoised_signals = {}
    
    models = [
        ("deepfilter", denoise_with_deepfilter),
        ("demucs", denoise_with_demucs),
        ("denoiser", denoise_with_denoiser),
        ("cleanunet", denoise_with_cleanunet),
    ]
    
    for model_name, denoise_fn in models:
        print(f"\n--- Model {model_name.upper()} ---")
        
        denoised, metadata = denoise_fn(noisy_audio, sr=sr)
        
        if denoised is None:
            print(f"✗ {model_name} failed")
            results[model_name] = {"status": "failed", "error": metadata.get("error", "Unknown")}
            continue
        
        # Save denoised audio
        output_path = run_dir / f"{model_name}.wav"
        sf.write(output_path, denoised, sr)
        print(f"✓ Saved denoised audio: {output_path}")
        
        # Compute metrics
        metrics = {}
        
        # Trim to same length
        clean_t, denoised_t = trim_pair(clean_audio, denoised)
        clean_mono = mixdown_to_mono(clean_t)
        denoised_mono = mixdown_to_mono(denoised_t)
        
        # SNR and SI-SDR (always computed)
        metrics["snr"] = compute_snr_np(clean_mono, denoised_mono)
        metrics["si_sdr"] = compute_si_sdr_np(clean_mono, denoised_mono)
        
        print(f"  SNR: {metrics['snr']:.2f} dB")
        print(f"  SI-SDR: {metrics['si_sdr']:.2f} dB")
        
        # Optional: PESQ and STOI
        if stoi_fn is not None:
            try:
                metrics["stoi"] = float(stoi_fn(clean_mono, denoised_mono, sr, extended=False))
                print(f"  STOI: {metrics['stoi']:.4f}")
            except Exception as e:
                metrics["stoi"] = None
                print(f"  STOI: (error)")
        
        if pesq_fn is not None:
            try:
                clean_pesq = resample_audio(clean_mono, sr, 16000)
                denoised_pesq = resample_audio(denoised_mono, sr, 16000)
                clean_pesq, denoised_pesq = trim_pair(clean_pesq, denoised_pesq)
                metrics["pesq"] = float(pesq_fn(16000, clean_pesq, denoised_pesq, "wb"))
                print(f"  PESQ: {metrics['pesq']:.4f}")
            except Exception as e:
                metrics["pesq"] = None
                print(f"  PESQ: (error)")
        
        results[model_name] = {
            "status": "success",
            "output_file": str(output_path),
            "metrics": metrics,
        }
        denoised_signals[model_name] = denoised_mono.astype(np.float32)

    # Novelty: Create an ensemble denoised output from the best three models (by SI-SDR)
    successful = [
        (name, r["metrics"]["si_sdr"])
        for name, r in results.items()
        if r.get("status") == "success" and "metrics" in r and name in denoised_signals
    ]
    ranked_successful = sorted(successful, key=lambda x: x[1], reverse=True)
    for ensemble_size in range(1, min(3, len(ranked_successful)) + 1):
        top_pairs = ranked_successful[:ensemble_size]
        top_models = [name for name, _ in top_pairs]
        scores = np.array([score for _, score in top_pairs], dtype=np.float64)
        # Softmax weights from SI-SDR scores so better model gets higher contribution.
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)

        signals = [denoised_signals[name] for name in top_models]
        n = min(*(len(signal) for signal in signals), len(clean_audio))
        ensemble = sum(
            float(weight) * signal[:n]
            for weight, signal in zip(weights, signals)
        ).astype(np.float32)

        ensemble_name = f"ensemble_top{ensemble_size}"
        ensemble_path = run_dir / f"{ensemble_name}.wav"
        sf.write(ensemble_path, ensemble, sr)
        print(f"\n--- Novelty Denoiser: {ensemble_name.upper()} ({' + '.join(top_models)}) ---")
        print("  Weights: " + ", ".join(
            f"{name}={weight:.3f}" for name, weight in zip(top_models, weights)
        ))
        print(f"✓ Saved denoised audio: {ensemble_path}")

        clean_e = mixdown_to_mono(clean_audio[:n])
        ensemble_metrics = {
            "snr": compute_snr_np(clean_e, ensemble),
            "si_sdr": compute_si_sdr_np(clean_e, ensemble),
        }

        if stoi_fn is not None:
            try:
                ensemble_metrics["stoi"] = float(stoi_fn(clean_e, ensemble, sr, extended=False))
            except Exception:
                ensemble_metrics["stoi"] = None
        if pesq_fn is not None:
            try:
                clean_pesq = resample_audio(clean_e, sr, 16000)
                ensemble_pesq = resample_audio(ensemble, sr, 16000)
                clean_pesq, ensemble_pesq = trim_pair(clean_pesq, ensemble_pesq)
                ensemble_metrics["pesq"] = float(pesq_fn(16000, clean_pesq, ensemble_pesq, "wb"))
            except Exception:
                ensemble_metrics["pesq"] = None

        print(f"  SNR: {ensemble_metrics['snr']:.2f} dB")
        print(f"  SI-SDR: {ensemble_metrics['si_sdr']:.2f} dB")

        results[ensemble_name] = {
            "status": "success",
            "output_file": str(ensemble_path),
            "components": top_models,
            "weights": {name: float(weight) for name, weight in zip(top_models, weights)},
            "metrics": ensemble_metrics,
        }
    
    # ============================================================================
    # Step 4: Summary report
    # ============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    visualization_path = run_dir / "top_model_visualization.png"
    
    summary = {
        "timestamp": timestamp,
        "recording_duration_s": recording_duration,
        "sample_rate_hz": sr,
        "snr_db": snr_db,
        "mode": mode,
        "output_dir": str(run_dir),
        "clean_audio": str(clean_path),
        "noisy_audio": str(noisy_path),
        "noise_source": str(noise_source_path) if noise_source_path is not None else None,
        "noise_used": str((run_dir / "noise_used.wav")) if (run_dir / "noise_used.wav").exists() else None,
        "audio_visualization": None,
        "models": results,
    }
    
    # Save JSON summary
    summary_path = run_dir / "sum.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary: {summary_path}")
    
    def get_composite_score(model_result):
        metrics = model_result.get("metrics", {})
        score = metrics.get("si_sdr", 0)
        if metrics.get("stoi") is not None:
            score += metrics["stoi"] * 10
        if metrics.get("pesq") is not None:
            score += metrics["pesq"] * 2
        return score

    normal_results = {
        model_name: model_result
        for model_name, model_result in results.items()
        if not model_name.startswith("ensemble_")
    }
    ensemble_results = {
        model_name: model_result
        for model_name, model_result in results.items()
        if model_name.startswith("ensemble_")
    }
    ordered_results = [
        *sorted(
            normal_results.items(),
            key=lambda item: (item[1].get("status") != "success", get_composite_score(item[1])),
        ),
        *sorted(
            ensemble_results.items(),
            key=lambda item: (item[1].get("status") != "success", get_composite_score(item[1])),
        ),
    ]

    def format_metric(metrics, name, suffix=""):
        value = metrics.get(name)
        return f"{value:.4f}{suffix}" if value is not None else "N/A"

    for model_name, model_result in ordered_results:
        status = "✓" if model_result["status"] == "success" else "✗"
        print(f"{status} {model_name.upper()}: {model_result['status']}")
        if model_result["status"] == "success" and "metrics" in model_result:
            m = model_result["metrics"]
            print(
                f"    PESQ: {format_metric(m, 'pesq')} | "
                f"STOI: {format_metric(m, 'stoi')} | "
                f"SI-SDR: {format_metric(m, 'si_sdr', ' dB')} | "
                f"SNR: {format_metric(m, 'snr', ' dB')}"
            )
        if model_result["status"] == "success" and "weights" in model_result:
            w = model_result["weights"]
            w_str = ", ".join([f"{k}={v:.3f}" for k, v in w.items()])
            print(f"    Weights: {w_str}")
    
    # Novelty: Automatic best model ranking
    print("\n" + "=" * 70)
    print("🏆 MODEL RANKING - ASCENDING (best model last)")
    print("=" * 70)
    model_scores = {}
    for model_name, model_result in results.items():
        if model_result["status"] != "success" or "metrics" not in model_result:
            continue
        m = model_result["metrics"]
        model_scores[model_name] = get_composite_score(model_result)
    
    if model_scores:
        normal_scores = {
            model: score
            for model, score in model_scores.items()
            if not model.startswith("ensemble_")
        }
        ensemble_scores = {
            model: score
            for model, score in model_scores.items()
            if model.startswith("ensemble_")
        }
        ranked = [
            *sorted(normal_scores.items(), key=lambda item: item[1]),
            *sorted(
                ensemble_scores.items(),
                key=lambda item: item[1],
            ),
        ]
        for i, (model, score) in enumerate(ranked, 1):
            m = results[model]["metrics"]
            print(
                f"{i}. {model.upper():12} | "
                f"PESQ: {format_metric(m, 'pesq')} | "
                f"STOI: {format_metric(m, 'stoi')} | "
                f"SI-SDR: {format_metric(m, 'si_sdr', ' dB')} | "
                f"SNR: {format_metric(m, 'snr', ' dB')} | "
                f"Score: {score:.2f}"
            )

    if model_scores:
        top_model_name = max(model_scores, key=model_scores.get)
        top_model_path = Path(results[top_model_name]["output_file"])
        top_model_output, _ = sf.read(str(top_model_path), dtype="float32")
        visualization_path = run_dir / "top_model_visualization.png"
        visualization_noise = (
            align_noise_to_length(noise_audio, len(clean_audio))
            if noise_audio is not None
            else None
        )
        save_audio_visualization(
            clean_audio=clean_audio,
            input_audio=noisy_audio,
            noise_audio=visualization_noise,
            top_output=top_model_output,
            top_model_name=top_model_name.upper(),
            sr=sr,
            output_path=visualization_path,
        )
        summary["audio_visualization"] = str(visualization_path)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"All outputs saved to: {run_dir}")
    print("=" * 70 + "\n")
    
    return summary


def ask_processing_mode() -> str:
    """Ask the user how the recorded microphone audio should be processed."""
    print("\nChoose audio processing mode:")
    print("1. Filter raw audio")
    print("2. Add noise, then filter")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "filter_raw"
        if choice == "2":
            return "add_noise"
        print("Invalid choice. Please enter 1 or 2.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live audio recording with multi-model denoising demo"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0). Use 0 to record until SPACE is pressed.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=10.0,
        help="Signal-to-noise ratio in dB (default: 10.0)",
    )
    parser.add_argument(
        "--noise-gain-db",
        type=float,
        default=0.0,
        help="Additional gain to apply to the noise in dB (positive -> louder noise)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_LIVEDATA_DIR),
        help="Base output directory for saved audio and results (default: live_demo/livedata)",
    )
    
    args = parser.parse_args()
    mode = ask_processing_mode()
    
    try:
        summary = run_live_demo(
            output_dir=args.output_dir,
            recording_duration=args.duration,
            sr=args.sr,
            snr_db=args.snr,
            noise_gain_db=args.noise_gain_db,
            mode=mode,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
