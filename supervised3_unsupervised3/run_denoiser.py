import torch
import torchaudio # Used for math, not I/O
import soundfile as sf
from denoiser import pretrained
from denoiser.dsp import convert_audio

def denoise_audio(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 1. Load the model
    print("Loading model...")
    model = pretrained.dns48().to(device)
    model.eval()

    # 2. Bypass torchaudio! Load directly with soundfile
    print(f"Loading {input_path} using soundfile...")
    wav_numpy, sr = sf.read(input_path)
    wav = torch.from_numpy(wav_numpy).float()
    
    # Soundfile: (length, channels) -> PyTorch: (channels, length)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) 
    else:
        wav = wav.transpose(0, 1)

    # 3. Convert sample rate and channels safely
    # (convert_audio uses pure math, so it won't trigger the FFmpeg crash)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    wav = wav.to(device)
    
    # RECORD EXACT LENGTH after resampling to trim padding later
    expected_length = wav.shape[-1]

    # 4. Denoise the audio
    print("Denoising... (this may take a moment)")
    with torch.no_grad():
        denoised_wav = model(wav.unsqueeze(0)).squeeze(0)
        
        # FIX: Truncate to exact original length to remove Demucs block padding
        denoised_wav = denoised_wav[..., :expected_length]

    # 5. Bypass torchaudio for saving
    print(f"Saving to {output_path}...")
    wav_out = denoised_wav.cpu().numpy()
    
    # Convert back to (length, channels) for soundfile
    if wav_out.ndim == 2:
        wav_out = wav_out.transpose(1, 0)
        
    sf.write(output_path, wav_out, model.sample_rate)
    print(f"Successfully saved exact-length clean audio to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "p268_002.wav"
    OUTPUT_FILE = "clean_speech_demucs.wav"
    
    denoise_audio(INPUT_FILE, OUTPUT_FILE)