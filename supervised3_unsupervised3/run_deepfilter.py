import sys
import types

# --- THE MONKEY PATCH ---
# DeepFilterNet is looking for an old torchaudio file that no longer exists.
# Since we are bypassing their audio loader anyway, we just create a "fake" 
# module to trick the import statement so the script can start.
fake_module = types.ModuleType('torchaudio.backend.common')
fake_module.AudioMetaData = object 
sys.modules['torchaudio.backend.common'] = fake_module
# ------------------------

import torch
import torchaudio # Used only for math/resampling
import soundfile as sf
from df.enhance import enhance, init_df

def denoise_with_deepfilter(input_path, output_path):
    print("Initializing DeepFilterNet... (this will download weights on first run)")
    
    # 1. Initialize the model and state. 
    model, df_state, _ = init_df()
    
    # DeepFilterNet usually operates at 48000 Hz
    target_sr = df_state.sr() 

    # 2. Bypass torchaudio I/O! Load directly with soundfile
    print(f"Loading {input_path} using soundfile...")
    wav_numpy, sr = sf.read(input_path)
    wav = torch.from_numpy(wav_numpy).float()
    
    # Convert from soundfile shape (length, channels) to PyTorch shape (channels, length)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) 
    else:
        wav = wav.transpose(0, 1)

    # 3. Safely resample if your audio doesn't match the model's native rate
    if sr != target_sr:
        print(f"Resampling audio from {sr}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    # Record EXACT LENGTH so we can trim any block padding later
    expected_length = wav.shape[-1]

    # 4. Denoise the audio
    print("Denoising... (DeepFilterNet is very fast!)")
    with torch.no_grad():
        # DeepFilterNet's enhance function expects (channels, length)
        denoised_wav = enhance(model, df_state, wav)
        
        # FIX: Truncate to exact original length to remove any model padding
        denoised_wav = denoised_wav[..., :expected_length]

    # 5. Bypass torchaudio for saving
    print(f"Saving to {output_path}...")
    wav_out = denoised_wav.cpu().numpy()
    
    # Convert back to (length, channels) for soundfile
    if wav_out.ndim == 2:
        wav_out = wav_out.transpose(1, 0)
        
    sf.write(output_path, wav_out, target_sr)
    print(f"Successfully saved exact-length clean audio to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "p268_002.wav"       
    OUTPUT_FILE = "clean_speech_deepfilter.wav"
    
    denoise_with_deepfilter(INPUT_FILE, OUTPUT_FILE)