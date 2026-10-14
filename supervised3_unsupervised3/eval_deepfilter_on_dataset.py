import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# DeepFilterNet expects an older torchaudio path; provide a minimal shim.
fake_module = types.ModuleType("torchaudio.backend.common")
fake_module.AudioMetaData = object
sys.modules["torchaudio.backend.common"] = fake_module

from df.enhance import enhance, init_df


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

    return wav, sr


def maybe_resample(wav: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    if src_sr == target_sr:
        return wav
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=target_sr)
    return resampler(wav)


def resample_np_mono(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wav
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    wav_rs = torchaudio.functional.resample(wav_t, src_sr, target_sr)
    return wav_rs.squeeze(0).cpu().numpy()


def trim_pair(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(ref.shape[-1], deg.shape[-1])
    return ref[:n], deg[:n]


def mixdown_to_mono_np(wav: np.ndarray) -> np.ndarray:
    # Ensure metrics are computed on a 1-D mono signal.
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return np.mean(wav, axis=0)
    return np.squeeze(wav)


def mean_non_nan(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def std_non_nan(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return float(np.std(vals)) if vals else float("nan")


def evaluate_dataset(
    input_dir: Path,
    clean_dir: Path,
    out_csv: Path,
    out_json: Path,
    max_files: Optional[int] = None,
) -> dict:
    print("Initializing DeepFilterNet... (may download weights on first run)")
    model, df_state, _ = init_df()
    target_sr = int(df_state.sr())

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
            denoised = enhance(model, df_state, noisy_wav)
            denoised = denoised[..., :expected_length]

            den_np = denoised.cpu().numpy()
            clean_np = clean_wav.cpu().numpy()
            den_np = mixdown_to_mono_np(den_np)
            clean_np = mixdown_to_mono_np(clean_np)
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
                    # PESQ is defined for 8k (nb) and 16k (wb). We use 16k wb.
                    clean_pesq = resample_np_mono(clean_np, target_sr, 16000)
                    den_pesq = resample_np_mono(den_np, target_sr, 16000)
                    clean_pesq, den_pesq = trim_pair(clean_pesq, den_pesq)
                    row["pesq"] = float(pesq_fn(16000, clean_pesq, den_pesq, "wb"))
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
        "pesq_sample_rate": 16000,
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
    parser = argparse.ArgumentParser(description="Run DeepFilterNet on dataset and report average metrics")
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
        "--out-csv",
        type=str,
        default=r"supervised3_unsupervised3/outputs/us_class1_train_deepfilter_metrics.csv",
        help="Per-file metrics CSV",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=r"supervised3_unsupervised3/outputs/us_class1_train_deepfilter_summary.json",
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
        max_files=args.max_files,
    )

    print("\nAverage metrics over evaluated files:")
    print(f"SNR    : {summary['snr']:.4f} (std: {summary['snr_std']:.4f})")
    print(f"SI-SDR : {summary['si_sdr']:.4f} (std: {summary['si_sdr_std']:.4f})")
    print(f"STOI   : {summary['stoi']:.4f} (std: {summary['stoi_std']:.4f})")
    print(f"PESQ   : {summary['pesq']:.4f} (std: {summary['pesq_std']:.4f})")
    print(f"Files found: {summary['files_found']}")
    print(f"Files evaluated: {summary['files_evaluated']} / {summary['files_found']}")
    print(f"Missing clean files: {summary['missing_clean']}")
    print(f"DeepFilter target SR: {summary['target_sample_rate']}")
    print(f"PESQ sample rate: {summary['pesq_sample_rate']}")
    print(f"Saved CSV  : {args.out_csv}")
    print(f"Saved JSON : {args.out_json}")


if __name__ == "__main__":
    main()
