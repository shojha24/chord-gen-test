from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from inference import extract_lyrics_for_song, get_model, run_inference, decode_chords_with_viterbi, write_chords_to_midi, impose_midi_on_audio
from uuid import uuid4
import time
import os

app = FastAPI()

# Route to return a new job ID (4 digit UUID followed by timestamp)
@app.post("/new-job")
async def create_job(audio: UploadFile = File(...)):
    job_id = f"{uuid4()}"

    # create a directory for the job within ./temp
    os.makedirs(f"temp/{job_id}", exist_ok=True)

    # save the uploaded file to that directory
    extension = os.path.splitext(audio.filename)[1]
    temp_path = f"temp/{job_id}/audio{extension}"

    with open(temp_path, "wb") as f:
        f.write(await audio.read())
        f.close()

    return {"job_id": job_id}

@app.get("/lyrics")
async def detect_lyrics(job_id: str = None):
    # Look for the audio file in the job directory
    folder_path = f"temp/{job_id}"
    lyrics_path = os.path.join(folder_path, "lyrics_with_timestamps.txt")
    for file in os.listdir(folder_path):
        if file.startswith("audio"):
            audio_path = os.path.join(folder_path, file)
            break

    # If no audio file was found, return an error
    if not audio_path:
        return {"error": "Audio file not found"}, 404

    # Call your lyric detection function (ensure it's async or wrapped in run_in_executor)
    lyrics = await run_in_threadpool(
        extract_lyrics_for_song,
        input_mp3=audio_path,
        output_path=lyrics_path,
        whisper_model="small"
    )

    return {"status": "complete"}

@app.get("/chords")
async def detect_chords(job_id: str = None):
    self_transition_prob = 0.985
    sound_font_path = "FluidR3_GM.sf2"
    # Look for the audio file in the job directory
    folder_path = f"temp/{job_id}"
    chords_path = os.path.join(folder_path, "chords_with_timestamps.txt")
    midi_path = os.path.join(folder_path, "audio.mid")
    imposed_path = os.path.join(folder_path, "final_audio.wav")

    for file in os.listdir(folder_path):
        if file.startswith("audio"):
            audio_path = os.path.join(folder_path, file)
            break

    # If no audio file was found, return an error
    if not audio_path:
        return {"error": "Audio file not found"}, 404

    # Call your chord detection function (ensure async or run_in_executor)
    model, device = get_model("best_full.pt")

    logits, song_len, interval = await run_in_threadpool(
        run_inference,
        model,
        audio_path,
        device
    )

    chords = await run_in_threadpool(
        decode_chords_with_viterbi,
        logits,
        song_len,
        interval,
        self_transition_prob,
        chords_path
    )

    await run_in_threadpool(write_chords_to_midi, chords, midi_path)
    await run_in_threadpool(impose_midi_on_audio, midi_path, audio_path, sound_font_path, imposed_path)

    return {"status": "complete"}