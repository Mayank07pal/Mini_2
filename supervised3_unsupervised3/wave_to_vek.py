import torch
import torchaudio # Used strictly for pure math/resampling
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_with_wav2vec(input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running wav2vec 2.0 on: {device}")

    # 1. Load the Processor and Model from Hugging Face
    # We are using the standard "base" model trained on 960 hours of English speech
    print("Downloading/Loading wav2vec 2.0 model...")
    model_id = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    # 2. Bypass torchaudio I/O! Load directly with soundfile
    print(f"Loading {input_path} using soundfile...")
    wav_numpy, sr = sf.read(input_path)
    
    # Force Mono: If the audio is stereo (2D array), take only the first channel
    if wav_numpy.ndim > 1:
        wav_numpy = wav_numpy[:, 0]

    # Convert to PyTorch tensor for safe mathematical resampling
    wav = torch.from_numpy(wav_numpy).float()

    # 3. Safely Resample: Wav2vec 2.0 strictly requires 16000Hz
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)

    # 4. Process the audio into the format the model expects
    print("Transcribing audio...")
    # The Hugging Face processor expects a 1D list or numpy array
    input_values = processor(
        wav.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)

    # 5. Run inference (Get the raw prediction scores)
    with torch.no_grad():
        logits = model(input_values).logits

    # 6. Decode the raw scores into actual text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Print the final result!
    print("\n" + "="*50)
    print("TRANSCRIPTION RESULT:")
    print("="*50)
    print(transcription)
    print("="*50 + "\n")

if __name__ == "__main__":
    # Point this to your audio file (make sure it contains spoken English)
    INPUT_FILE = "p268_002.wav"       
    
    transcribe_with_wav2vec(INPUT_FILE)


#     It is actually both, but it is famous because its core engine is self-supervised.

# Wav2vec 2.0 is built in two distinct phases. Here is how it works:

# Phase 1: Self-Supervised (Learning how to hear)
# This is what made wav2vec 2.0 a massive breakthrough. Researchers took tens of thousands of hours of raw audio (just people talking) with no text transcripts at all.

# During training, the model takes a slice of audio, "mutes" a fraction of a second of it, and tries to guess what that missing sound is based on the sounds before and after it. Because it is generating its own test questions and checking its own answers using the raw audio, it is completely self-supervised.

# By doing this millions of times, it learns the fundamental structure, tone, and phonemes of human speech without knowing a single word of English.

# Phase 2: Supervised (Learning how to spell)
# A model that just "understands the shape of sound" isn't useful for dictation. To make it output text, researchers add a tiny translation layer on top of the model and do a second round of training.

# This second round is supervised. They feed it audio paired with perfect text transcripts. Because the model already fully understands human speech from Phase 1, it only needs a relatively small amount of labeled data (Phase 2) to figure out which sound connects to the letter "A" or the word "Apple."

# Which one did we use?
# In the script above, we used facebook/wav2vec2-base-960h.

# The base part means it went through the massive self-supervised training.

# The 960h part means it was fine-tuned (supervised) on 960 hours of perfectly transcribed audiobooks so it knows how to type out English text.

# So, to answer your question: It is a self-supervised audio model that has been fine-tuned with supervised text data