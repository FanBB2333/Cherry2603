#!/usr/bin/env python3
"""
Pre-download HuggingFace models for the pipeline.
"""
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def download_models():
    """Download all required models."""
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    print(f"Downloading model: {model_name}")
    print("This may take a few minutes (approximately 1.2GB)...")

    # Download processor
    print("Downloading processor...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Download model
    print("Downloading model weights...")
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    print(f"✅ Model downloaded successfully!")
    print(f"Cache location: ~/.cache/huggingface/hub/")

if __name__ == '__main__':
    download_models()
