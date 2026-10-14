#!/usr/bin/env python
"""
Quick-start script for live audio denoising demo.
Minimal setup - just run and it handles the rest.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    demo_script = script_dir / "live_audio_demo.py"
    livedata_dir = script_dir / "livedata"
    
    if not demo_script.exists():
        print(f"❌ Error: {demo_script} not found")
        sys.exit(1)
    
    # Check for required packages
    required = ["sounddevice", "soundfile", "numpy", "torch", "torchaudio"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("⚠️  Missing packages. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    
    print("\n" + "="*70)
    print("🎤 LIVE AUDIO DENOISING DEMO")
    print("="*70)
    print("\nOptions:")
    print("  1. Quick demo (5 sec, default settings)")
    print("  2. Long recording (10 sec, SNR=10dB)")
    print("  3. Noisy test (5 sec, SNR=5dB - more noise)")
    print("  4. Custom parameters")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    args = [sys.executable, str(demo_script)]
    
    if choice == "1":
        print("\n▶️  Starting quick demo (5 seconds)...")
        subprocess.call(args)
    
    elif choice == "2":
        print("\n▶️  Starting long recording (10 seconds)...")
        subprocess.call(args + ["--duration", "10"])
    
    elif choice == "3":
        print("\n▶️  Starting noisy test (5 sec, high noise)...")
        subprocess.call(args + ["--snr", "5"])
    
    elif choice == "4":
        duration = input("Duration (seconds) [5]: ") or "5"
        snr = input("SNR (dB) [10]: ") or "10"
        sr = input("Sample rate (Hz) [16000]: ") or "16000"
        outdir = input(f"Output directory [{livedata_dir}]: ") or str(livedata_dir)
        
        subprocess.call(args + [
            "--duration", duration,
            "--snr", snr,
            "--sr", sr,
            "--output-dir", outdir
        ])
    
    elif choice == "5":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()
