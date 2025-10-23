from typing import Optional, Dict, Any, Union

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    status,
    Depends,
)
from fastapi.concurrency import run_in_threadpool

import transcriber

app = FastAPI(
    title="Transcription API",
    description="An API to transcribe audio from YouTube videos or uploaded files. Now fully testable from the docs UI!",
    version="1.2.0" # Version bump for the fix
)

async def validate_and_process_inputs(
    youtube_url: Optional[str] = Form(None, description="A YouTube video URL to transcribe."),
    # THE KEY FIX: Allow either UploadFile or str for the audio_file parameter
    audio_file: Union[UploadFile, str] = File(None, description="An audio file to transcribe.")
) -> Dict[str, Any]:
    """
    Dependency that validates the inputs and handles the Swagger UI empty string case.
    By accepting Union[UploadFile, str], we allow FastAPI's validation to pass,
    and then we perform the logical validation ourselves.
    """
    # If the browser sends an empty string for the file, treat it as None.
    if isinstance(audio_file, str):
        audio_file = None

    # Now perform the logical validation
    has_url = bool(youtube_url)
    has_file = bool(audio_file)

    if not has_url and not has_file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="You must provide either a 'youtube_url' or an 'audio_file'."
        )
    
    if has_url and has_file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please provide either a 'youtube_url' or an 'audio_file', not both."
        )

    return {"youtube_url": youtube_url, "audio_file": audio_file}


@app.get("/", tags=["General"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}


@app.post("/transcribe", tags=["Transcription"])
async def transcribe_endpoint(
    # Use the dependency to get clean, validated inputs
    inputs: Dict[str, Any] = Depends(validate_and_process_inputs)
):
    """
    Transcribes audio from either a YouTube URL or an uploaded audio file.
    
    This endpoint is now robust and can be tested directly from the interactive docs.
    """
    youtube_url = inputs.get("youtube_url")
    audio_file = inputs.get("audio_file")
    
    try:
        if youtube_url:
            transcript = await run_in_threadpool(transcriber.process_youtube_url, url=youtube_url)
        else: # We know from our dependency that if url is None, file must exist
            transcript = await run_in_threadpool(transcriber.process_audio_file, file=audio_file)

    except Exception as e:
        error_message = str(e)
        if "HTTP Error 403" in error_message or "unable to download" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to download audio from the provided YouTube URL. The video may be private or protected."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during transcription: {error_message}"
        )

    return {"transcript": transcript}