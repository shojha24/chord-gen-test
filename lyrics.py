import os
import subprocess
import sys
import whisper

def split_stems(input_mp3, output_dir):
    """Split mp3 file into vocal/instrumental stems using Demucs."""
    cmd = [sys.executable, "-m", "demucs", "--mp3", "--two-stems=vocals", input_mp3, "-o", output_dir]
    subprocess.run(cmd, check=True)

def extract_lyrics(audio_path, model_size="base"):
    """Extract lyrics with word-by-word timestamps using OpenAI Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)
    return result["segments"]

def save_lyrics_with_timestamps(segments, output_path):
    """Merge Whisper segments and save timestamps."""
    lines = []
    for segment in segments:
        for word in segment.get("words", []):
            start = word["start"]
            end = word["end"]
            text = word["word"].strip()
            if text:  # Ignore empty "words"
                lines.append(f"{start:.3f}\t{end:.3f}\t{text}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def extract_lyrics_for_song(input_mp3, output_dir, whisper_model="small"):
    """Top-level function: splits stems, extracts vocals, runs Whisper, saves lyrics."""
    # Split into vocal stems
    split_stems(input_mp3, output_dir)
    # Determine vocal path (assumes standard Demucs folder structure)
    base_name = os.path.splitext(os.path.basename(input_mp3))[0]
    vocal_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.mp3")
    # Extract lyrics
    segments = extract_lyrics(vocal_path, model_size=whisper_model)
    # Save results
    lyrics_txt = os.path.join(output_dir, "htdemucs", base_name, "lyrics_with_timestamps.txt")
    save_lyrics_with_timestamps(segments, lyrics_txt)
    return lyrics_txt
