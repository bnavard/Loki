"""
Generate text captions from audio (ASR transcription + prosody description).

For each clip, produces a structured caption combining:
  - ASR transcription via Whisper (what is said)
  - Prosody description via Qwen2-Audio (how it is said)

Input:  data/talkvid/audio/{clip_id}.wav
Output: data/derived/captions/{clip_id}.json

Usage:
    cd <repo_root>
    

    # ASR only (faster, no Qwen2-Audio):
    PYTHONPATH=. python scripts/caption/generate_captions.py --asr_only

    # Single clip test:
    PYTHONPATH=. python scripts/caption/generate_captions.py --test --clip abc123

    # Parallel across GPUs:
    PYTHONPATH=. python scripts/caption/generate_captions.py --gpu 0 --num_gpus 8
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = Path("data/talkvid/audio")
FLOWFACE_DIR = Path("data/flowface")
OUTPUT_DIR = Path("data/derived/captions")
SAMPLE_RATE = 16_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", type=str, default=str(AUDIO_DIR))
    p.add_argument("--flowface_dir", type=str, default=str(FLOWFACE_DIR))
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--test", action="store_true")
    p.add_argument("--clip", type=str, default=None)
    p.add_argument("--asr_only", action="store_true",
                   help="Skip prosody description (Qwen2-Audio), only run Whisper ASR")
    return p.parse_args()


def load_whisper(device):
    """Load Whisper large-v3 for ASR transcription."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    return model, processor


def transcribe(audio_path, whisper_model, whisper_processor, device):
    """Run Whisper ASR on an audio file. Returns transcription string."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    inputs = whisper_processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
    ).input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs, max_new_tokens=256)

    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True,
    )[0].strip()

    return transcription


def load_qwen_audio(device):
    """Load Qwen2-Audio for prosody description."""
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    return model, processor


def describe_prosody(audio_path, qwen_model, qwen_processor, device):
    """
    Use Qwen2-Audio to describe the prosody/delivery of the speech.
    Returns a description string.
    """
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    prompt = (
        "You are analyzing a short audio clip from a YouTube video where a person is "
        "speaking directly to a camera in a conversational style.\n\n"
        "Your task is to describe the vocal delivery of this speech. Address the following:\n\n"
        "1. Overall delivery style: Is it calm, energetic, subdued, animated, monotone, "
        "or something else? How would you characterize the general tone?\n\n"
        "2. Key words or phrases that stand out: Identify specific words, verbs, or "
        "adjectives where the speaker places noticeable emphasis, changes pitch, "
        "or shifts energy. Quote the exact words if possible.\n\n"
        "3. Temporal dynamics: Note where in the clip (beginning, middle, end) any "
        "shifts in delivery occur — such as the speaker speeding up, slowing down, "
        "pausing, raising or lowering their voice, or changing their energy level.\n\n"
        "4. Pace and rhythm: Is the speech fast, moderate, or slow? Is the rhythm "
        "steady or does it vary? Are there notable pauses or rushed segments?\n\n"
        "Keep your response to 3-4 sentences. Be specific and grounded in what you "
        "actually hear — avoid speculating about the speaker's internal feelings "
        "or intentions."
    )

    conversation = [
        {"role": "system", "content": "You are a speech analysis expert."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = [audio]

    inputs = qwen_processor(
        text=text, audios=audios, sampling_rate=SAMPLE_RATE,
        return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=256)

    # Decode only the generated tokens (skip the prompt)
    input_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_len:]
    prosody = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return prosody


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)
    flowface_dir = Path(args.flowface_dir)

    # Collect clip IDs that have both audio and FLAME tracking
    all_clips = sorted([
        d.name for d in flowface_dir.iterdir()
        if d.is_dir() and (d / "fit.npz").exists()
           and (audio_dir / f"{d.name}.wav").exists()
    ])
    print(f"Found {len(all_clips)} clips with audio + FLAME tracking")

    # Shard across GPUs
    my_clips = all_clips[args.gpu::args.num_gpus]
    print(f"GPU {args.gpu}/{args.num_gpus}: processing {len(my_clips)} clips")

    if args.test:
        if args.clip:
            my_clips = [args.clip]
        else:
            my_clips = my_clips[:1]
        print(f"Test mode: processing {my_clips[0]}")

    # Skip already processed
    to_process = [c for c in my_clips if not (output_dir / f"{c}.json").exists()]
    print(f"To process: {len(to_process)} | Already done: {len(my_clips) - len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    # Load models
    print("Loading Whisper large-v3...")
    whisper_model, whisper_processor = load_whisper(device)

    qwen_model, qwen_processor = None, None
    if not args.asr_only:
        print("Loading Qwen2-Audio-7B-Instruct...")
        qwen_model, qwen_processor = load_qwen_audio(device)

    n_ok = n_fail = 0
    for clip_id in tqdm(to_process, desc=f"GPU {args.gpu}"):
        try:
            audio_path = audio_dir / f"{clip_id}.wav"

            # ASR transcription
            transcription = transcribe(audio_path, whisper_model, whisper_processor, device)

            # Prosody description
            if qwen_model is not None:
                prosody = describe_prosody(audio_path, qwen_model, qwen_processor, device)
            else:
                prosody = ""

            # Combine into structured caption
            if prosody:
                caption = f"A person says: '{transcription}' {prosody}"
            else:
                caption = f"A person says: '{transcription}'"

            result = {
                "clip_id": clip_id,
                "transcription": transcription,
                "prosody": prosody,
                "caption": caption,
            }

            with open(output_dir / f"{clip_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            n_ok += 1

            if args.test:
                print(f"\n  Transcription: {transcription}")
                print(f"  Prosody: {prosody}")
                print(f"  Caption: {caption}")

        except Exception as e:
            tqdm.write(f"[FAIL] {clip_id}: {e}")
            if args.test:
                traceback.print_exc()
            n_fail += 1

    print(f"Done. Success: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
