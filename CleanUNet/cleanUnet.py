import os
import json
import torch
import torchaudio # Safe to use for math/resampling, just not I/O
import soundfile as sf

# Import the CleanUNet architecture from NVIDIA's local network.py file
from network import CleanUNet

def denoise_with_cleanunet(input_path, output_path, config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CleanUNet on: {device}")

    # 1. Load configuration and initialize model
    print("Loading configuration...")
    with open(config_path) as f:
        config = json.load(f)
    
    model = CleanUNet(**config['network_config']).to(device)
    
    # 2. Smart Loader for weights
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model'] if isinstance(checkpoint['model'], dict) else checkpoint['model'].state_dict()
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith('module.') else k
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    # 3. Load with soundfile
    print(f"Loading {input_path}...")
    wav_numpy, sr = sf.read(input_path)
    wav = torch.from_numpy(wav_numpy).float()
    
    if wav.dim() == 2:
        wav = wav[:, 0] # Make mono if stereo

    # 4. FIX: Actually Resample the audio to 16kHz!
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)

    wav = wav.to(device)
    
    # Record the exact length so we can trim trailing silence later
    expected_length = wav.shape[-1]

    # 5. Denoise the audio
    print("Denoising... (this may take a moment)")
    with torch.no_grad():
        wav_batch = wav.unsqueeze(0).unsqueeze(0) 
        denoised_wav = model(wav_batch)
        
        if isinstance(denoised_wav, tuple):
            denoised_wav = denoised_wav[0]
            
        denoised_wav = denoised_wav.squeeze() 
        
        # FIX: Truncate to exact original length to remove any model padding
        denoised_wav = denoised_wav[:expected_length]

    # 6. Save
    print(f"Saving to {output_path}...")
    wav_out = denoised_wav.cpu().numpy()
    
    sf.write(output_path, wav_out, 16000)
    print(f"Successfully saved exactly 3-second clean audio to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "p268_002.wav"       
    OUTPUT_FILE = "clean_speech_unet.wav"
    CONFIG_FILE = "configs/DNS-large-high.json"
    CHECKPOINT_FILE = "exp/DNS-large-high/checkpoint/pretrained.pkl"
    
    denoise_with_cleanunet(INPUT_FILE, OUTPUT_FILE, CONFIG_FILE, CHECKPOINT_FILE)