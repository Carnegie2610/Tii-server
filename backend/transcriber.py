import os
import tempfile
import whisper
import yt_dlp
from fastapi import UploadFile

# Load the Whisper model. Using "base" is a good starting point.
# Other models: "tiny", "small", "medium", "large"
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded.")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Whisper.
    """
    print(f"Transcribing file: {file_path}")
    result = model.transcribe(file_path, fp16=False) # fp16=False for CPU-only inference
    print("Transcription complete.")
    return result["text"]

def process_youtube_url(url: str) -> str:
    """
    Downloads audio from a YouTube URL, transcribes it, and cleans up.
    """
    # Use a temporary directory to store the downloaded audio
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': False,
        }

        print(f"Downloading audio from URL: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # The downloaded file will be named 'audio.mp3' inside temp_dir
        audio_path = os.path.join(temp_dir, 'audio.mp3')
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Failed to download audio from the URL.")

        transcript = transcribe_audio(audio_path)

    return transcript

def process_audio_file(file: UploadFile) -> str:
    """
    Saves an uploaded audio file temporarily, transcribes it, and cleans up.
    """
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    try:
        transcript = transcribe_audio(temp_file_path)
    finally:
        # Ensure the temporary file is deleted
        os.unlink(temp_file_path)

    return transcript