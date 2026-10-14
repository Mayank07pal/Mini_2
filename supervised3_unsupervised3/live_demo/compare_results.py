"""
Simple comparison script to analyze live demo results.
Reads the generated JSON summaries and displays comparison.
"""

import json
from pathlib import Path
import sys

def load_summary(json_path):
    """Load summary JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def print_comparison(directory):
    """Print comparison of all models from a demo run."""
    directory = Path(directory)
    
    # Find summary JSON files
    summaries = sorted(directory.glob("*_summary.json"))
    
    if not summaries:
        print(f"❌ No summary files found in {directory}")
        return
    
    print("\n" + "="*80)
    print("LIVE DEMO RESULTS COMPARISON")
    print("="*80 + "\n")
    
    for summary_path in summaries:
        data = load_summary(summary_path)
        
        print(f"Timestamp: {data['timestamp']}")
        print(f"Duration: {data['recording_duration_s']}s | SR: {data['sample_rate_hz']}Hz | SNR: {data['snr_db']}dB")
        print("\n" + "-"*80)
        print(f"{'Model':<15} {'SNR (dB)':<12} {'SI-SDR (dB)':<14} {'STOI':<10} {'PESQ':<10}")
        print("-"*80)
        
        for model, result in data['models'].items():
            if result['status'] == 'success':
                m = result['metrics']
                snr_str = f"{m['snr']:.2f}" if m['snr'] is not None else "N/A"
                sisdr_str = f"{m['si_sdr']:.2f}" if m['si_sdr'] is not None else "N/A"
                stoi_str = f"{m['stoi']:.4f}" if m['stoi'] else "N/A"
                pesq_str = f"{m['pesq']:.2f}" if m['pesq'] else "N/A"
                
                print(f"{model:<15} {snr_str:<12} {sisdr_str:<14} {stoi_str:<10} {pesq_str:<10}")
            else:
                print(f"{model:<15} ERROR")
        
        print()

def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    print_comparison(directory)

if __name__ == "__main__":
    main()
