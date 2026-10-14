import torch
import torchaudio
import soundfile as sf

# Notice we import HubertForCTC, but we still use the Wav2Vec2Processor!
from transformers import HubertForCTC, Wav2Vec2Processor

def transcribe_with_hubert(input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HuBERT on: {device}")

    # 1. Load the Processor and Model from Hugging Face
    print("Downloading/Loading HuBERT model... (This is a large model, please wait on first run)")
    model_id = "facebook/hubert-large-ls960-ft"
    
    # HuBERT shares the same tokenizer/processor architecture as wav2vec 2.0
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = HubertForCTC.from_pretrained(model_id).to(device)

    # 2. Bypass torchaudio I/O! Load directly with soundfile
    print(f"Loading {input_path} using soundfile...")
    wav_numpy, sr = sf.read(input_path)
    
    # Force Mono
    if wav_numpy.ndim > 1:
        wav_numpy = wav_numpy[:, 0]

    wav = torch.from_numpy(wav_numpy).float()

    # --- DEBUGGING CHECK ---
    max_volume = wav.abs().max().item()
    if max_volume < 0.001:
        print("-> WARNING: Your audio file is basically silent!")

    # 3. Safely Resample
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        # Add channel dimension safely for PyTorch resampler
        wav = wav.unsqueeze(0) 
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
        wav = wav.squeeze(0) # Remove channel dimension

    # 4. Process the audio
    print("Transcribing audio...")
    input_values = processor(
        wav.numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)

    # 5. Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # 6. Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    print("\n" + "="*50)
    print("TRANSCRIPTION RESULT (HuBERT):")
    print("="*50)
    print(f"---> {transcription} <---") 
    print("="*50 + "\n")

if __name__ == "__main__":
    INPUT_FILE = "p268_002.wav"       
    
    transcribe_with_hubert(INPUT_FILE)