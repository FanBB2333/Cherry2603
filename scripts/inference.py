#!/usr/bin/env python3
"""
Run phoneme recognition inference using wav2vec2 model.
"""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf


def load_model(model_name: str):
    """Load wav2vec2 model and processor."""
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return processor, model, device


def predict_phonemes(audio_path: str, processor, model, device) -> str:
    """Predict phoneme sequence for an audio file."""
    # Load audio
    speech, sr = sf.read(audio_path)

    # Ensure mono
    if speech.ndim > 1:
        speech = speech[:, 0]

    # Process audio
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

    # Move to device
    input_values = inputs.input_values.to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription


def process_manifest(input_manifest: Path, output_manifest: Path, model_name: str):
    """Run inference on all audio files in manifest."""
    print(f"Loading model: {model_name}")
    processor, model, device = load_model(model_name)
    print(f"Model loaded on device: {device}")

    # Count total entries first
    with open(input_manifest, 'r', encoding='utf-8') as f:
        total_entries = sum(1 for _ in f)

    print(f"Total utterances to process: {total_entries}")

    manifest_entries = []

    with open(input_manifest, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            entry = json.loads(line.strip())

            # Run inference
            try:
                pred_phon = predict_phonemes(entry['wav_path'], processor, model, device)
            except Exception as e:
                print(f"Error processing {entry['utt_id']}: {e}")
                pred_phon = ""

            # Update entry with prediction
            new_entry = entry.copy()
            new_entry['pred_phon'] = pred_phon

            manifest_entries.append(new_entry)

            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / total_entries * 100
                print(f"Progress: {idx + 1}/{total_entries} ({progress:.1f}%)")

    # Write manifest atomically
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                     dir=output_manifest.parent,
                                     suffix='.tmp') as tmp_file:
        for entry in manifest_entries:
            json.dump(entry, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')
        tmp_path = Path(tmp_file.name)

    shutil.move(str(tmp_path), str(output_manifest))
    print(f"Created prediction manifest with {len(manifest_entries)} entries: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description='Run phoneme recognition inference')
    parser.add_argument('--input-manifest', type=Path, required=True,
                        help='Input manifest with audio files')
    parser.add_argument('--output-manifest', type=Path, required=True,
                        help='Output manifest with predictions')
    parser.add_argument('--model-name', type=str, required=True,
                        help='HuggingFace model name')

    args = parser.parse_args()
    process_manifest(args.input_manifest, args.output_manifest, args.model_name)


if __name__ == '__main__':
    main()
