from faster_whisper import WhisperModel

print("Starting transcription...")

audio_path = "data/audio/lecture.mp3"
output_path = "data/transcripts/lecture.txt"

model = WhisperModel("base", compute_type="int8")

segments, info = model.transcribe(audio_path)

with open(output_path, "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text.strip() + "\n")

print("Transcription completed successfully.")
print("Detected language:", info.language)
