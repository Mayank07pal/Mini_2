import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from network import CleanUNet


def compute_snr_np(clean: np.ndarray, est: np.ndarray) -> float:
    est_scaled = est * (np.sum(clean * est) / (np.sum(est ** 2) + 1e-8))
    noise = clean - est_scaled
    num = np.sum(clean ** 2) + 1e-8
    den = np.sum(noise ** 2) + 1e-8
    return float(10.0 * np.log10(num / den))


def compute_si_sdr_np(clean: np.ndarray, est: np.ndarray) -> float:
    clean = clean - np.mean(clean)
    est = est - np.mean(est)
    alpha = np.sum(est * clean) / (np.sum(clean ** 2) + 1e-8)
    proj = alpha * clean
    noise = est - proj
    ratio = (np.sum(proj ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return float(10.0 * np.log10(ratio))


def try_import_metrics():
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


def load_audio_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav_np, sr = sf.read(str(path))
    wav = torch.from_numpy(wav_np).float()

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    return wav, sr


def maybe_resample(wav: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    if src_sr == target_sr:
        return wav
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=target_sr)
    return resampler(wav)


def trim_pair(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(ref.shape[-1], deg.shape[-1])
    return ref[:n], deg[:n]


def mean_non_nan(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def std_non_nan(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return float(np.std(vals)) if vals else float("nan")


def load_cleanunet_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> CleanUNet:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = CleanUNet(**config["network_config"]).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"] if isinstance(checkpoint["model"], dict) else checkpoint["model"].state_dict()
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def evaluate_dataset(
    input_dir: Path,
    clean_dir: Path,
    out_csv: Path,
    out_json: Path,
    config_path: Path,
    checkpoint_path: Path,
    max_files: Optional[int] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CleanUNet on: {device}")

    print("Loading model...")
    model = load_cleanunet_model(config_path, checkpoint_path, device)
    target_sr = 16000

    pesq_fn, stoi_fn = try_import_metrics()

    noisy_files = sorted(input_dir.glob("*.wav"))
    if max_files is not None:
        noisy_files = noisy_files[:max_files]

    print(f"Files found: {len(noisy_files)}")

    rows: list[dict] = []
    missing_clean = 0

    with torch.no_grad():
        for i, noisy_path in enumerate(noisy_files, start=1):
            clean_path = clean_dir / noisy_path.name
            if not clean_path.exists():
                missing_clean += 1
                continue

            noisy_wav, noisy_sr = load_audio_mono(noisy_path)
            clean_wav, clean_sr = load_audio_mono(clean_path)

            noisy_wav = maybe_resample(noisy_wav, noisy_sr, target_sr)
            clean_wav = maybe_resample(clean_wav, clean_sr, target_sr)

            expected_length = noisy_wav.shape[-1]

            noisy_batch = noisy_wav.to(device).unsqueeze(0)
            denoised = model(noisy_batch)
            if isinstance(denoised, tuple):
                denoised = denoised[0]

            denoised = denoised.squeeze()
            denoised = denoised[:expected_length]

            den_np = denoised.detach().cpu().numpy()
            clean_np = clean_wav.squeeze(0).cpu().numpy()
            clean_np, den_np = trim_pair(clean_np, den_np)

            row = {
                "file": noisy_path.name,
                "snr": compute_snr_np(clean_np, den_np),
                "si_sdr": compute_si_sdr_np(clean_np, den_np),
                "stoi": float("nan"),
                "pesq": float("nan"),
            }

            if stoi_fn is not None:
                try:
                    row["stoi"] = float(stoi_fn(clean_np, den_np, target_sr, extended=False))
                except Exception:
                    pass

            if pesq_fn is not None:
                try:
                    row["pesq"] = float(pesq_fn(target_sr, clean_np, den_np, "wb"))
                except Exception:
                    pass

            rows.append(row)

            if i % 200 == 0:
                print(f"Processed {i}/{len(noisy_files)} files...")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "snr", "si_sdr", "stoi", "pesq"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_dir": str(input_dir),
        "clean_dir": str(clean_dir),
        "files_found": len(noisy_files),
        "files_evaluated": len(rows),
        "missing_clean": missing_clean,
        "target_sample_rate": target_sr,
        "pesq_sample_rate": target_sr,
        "snr": mean_non_nan(rows, "snr"),
        "snr_std": std_non_nan(rows, "snr"),
        "si_sdr": mean_non_nan(rows, "si_sdr"),
        "si_sdr_std": std_non_nan(rows, "si_sdr"),
        "stoi": mean_non_nan(rows, "stoi"),
        "stoi_std": std_non_nan(rows, "stoi"),
        "pesq": mean_non_nan(rows, "pesq"),
        "pesq_std": std_non_nan(rows, "pesq"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CleanUNet on dataset and report average metrics")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\Users\mayan\Desktop\Mini Project\AutoEncoder\Dataloder\US_Class1_Train_Input",
        help="Folder with noisy .wav files",
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        default=r"C:\Users\mayan\Desktop\Mini Project\AutoEncoder\Dataloder\US_Class1_Train_Output",
        help="Folder with matching clean .wav files",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=r"C:\Users\mayan\Desktop\Mini Project\CleanUNet\configs\DNS-large-high.json",
        help="Path to CleanUNet config JSON",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=r"C:\Users\mayan\Desktop\Mini Project\CleanUNet\exp\DNS-large-high\checkpoint\pretrained.pkl",
        help="Path to CleanUNet checkpoint",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=r"supervised3_unsupervised3/outputs/us_class1_train_cleanunet_metrics.csv",
        help="Per-file metrics CSV",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=r"supervised3_unsupervised3/outputs/us_class1_train_cleanunet_summary.json",
        help="Summary JSON with average metrics",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick testing",
    )

    args = parser.parse_args()

    summary = evaluate_dataset(
        input_dir=Path(args.input_dir),
        clean_dir=Path(args.clean_dir),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        config_path=Path(args.config_path),
        checkpoint_path=Path(args.checkpoint_path),
        max_files=args.max_files,
    )

    print("\nAverage metrics over evaluated files:")
    print(f"SNR    : {summary['snr']:.4f} ± {summary['snr_std']:.4f}")
    print(f"SI-SDR : {summary['si_sdr']:.4f} ± {summary['si_sdr_std']:.4f}")
    print(f"STOI   : {summary['stoi']:.4f} ± {summary['stoi_std']:.4f}")
    print(f"PESQ   : {summary['pesq']:.4f} ± {summary['pesq_std']:.4f}")
    print(f"Files found: {summary['files_found']}")
    print(f"Files evaluated: {summary['files_evaluated']} / {summary['files_found']}")
    print(f"Missing clean files: {summary['missing_clean']}")
    print(f"CleanUNet target SR: {summary['target_sample_rate']}")
    print(f"Saved CSV  : {args.out_csv}")
    print(f"Saved JSON : {args.out_json}")


if __name__ == "__main__":
    main()
