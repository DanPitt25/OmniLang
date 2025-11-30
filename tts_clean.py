#!/usr/bin/env python3
import sys
import os
import pickle
sys.path.insert(0, './IMS-Toucan')

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Modules.ControllabilityGAN.GAN import GanWrapper
from huggingface_hub import hf_hub_download
import torch
import soundfile

# Settings
TEXT = "θat is miːnə riːtʃə"
VOICE = 0
GENDER = 50.0
DURATION = 1.0
PITCH = 1.0
ENERGY = 1.0

def get_tts():
    if os.path.exists("tts_cache.pkl"):
        with open("tts_cache.pkl", 'rb') as f:
            return pickle.load(f)
    
    import contextlib
    import io
    
    # Suppress phonemizer init messages
    with contextlib.redirect_stdout(io.StringIO()):
        tts = ToucanTTSInterface(device="cpu")
        gan_path = hf_hub_download(repo_id="Flux9665/ToucanTTS", filename="embedding_gan.pt")
        tts.wgan = GanWrapper(gan_path, device=tts.device, num_cached_voices=10)
    
    with open("tts_cache.pkl", 'wb') as f:
        pickle.dump(tts, f)
    return tts

def speak(text):
    tts = get_tts()
    
    text = text.replace("g", "ɡ").strip()
    if not text.startswith("~"): text = "~ " + text
    if not text.endswith("~"): text = text + " ~"
    print(f"Sending to TTS: '{text}'")
    print(f"Using input_is_phones=True")
    
    tts.set_language("vec")
    tts.wgan.set_latent(VOICE)
    embedding = tts.wgan.modify_embed(torch.tensor([GENDER, 0.0, 0.0, 0.0, 0.0, 0.0]))
    tts.set_utterance_embedding(embedding=embedding)
    
    wav, sr = tts(text, input_is_phones=True, duration_scaling_factor=DURATION,
                  pitch_variance_scale=PITCH, energy_variance_scale=ENERGY)
    soundfile.write("output.wav", wav, sr)
    print(f"Generated {len(wav)/sr:.2f}s")
    return wav, sr

if __name__ == "__main__":
    speak(TEXT)