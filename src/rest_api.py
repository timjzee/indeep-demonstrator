"""Defines the RESTful API used by the Demonstrator to communicate between client and server."""

from __future__ import annotations
import os
import shutil
import time

import requests
from fastapi import FastAPI, UploadFile, Form, status
from fastapi.responses import FileResponse

import demonstrator

def send_user_speech_request(client: demonstrator.DemonstratorClient) -> tuple[float, str]:
    """Sends an HTTP response to the client containing the TTS-synthesized audio.

    Sends an HTTP response to the client containing the TTS-synthesized audio, which is attached as a file, alongside the audio's length in seconds and transcription in the response headers.

    Args:
        client (DemonstratorClient): The client to which to send the response.

    Returns:
        tuple[float, str]: The transcription and length in seconds of the synthesized audio.
    """
    with open(client.vad_model.path_to_temp_user_utterance, "rb") as user_utterance_stream:        
        response = requests.post(
            url=f"{client.api_url}/user-speech",
            files={"user_utterance": ("temp_user_utterance.mp3", user_utterance_stream, "audio/mpeg")},
            data={"read_intro": client.read_intro, 
                  "TTS_language": client.TTS_language},
        )
        
        with open(client.playback_module.path_to_temp_tts, "wb+") as temp_tts_stream:
            temp_tts_stream.write(response.content)
                    
        return float(response.headers["audio_length"]), response.headers["transcription"]
        
fast_api = FastAPI()
fast_api.demonstrator = None

@fast_api.get("/")
def _API_root() -> dict:
    """Home endpoint of the API.

    Home endpoint of the API. Returns a message when a GET request is received.

    Returns:
        dict: A response telling the client that a GET request has been performed successfully.
    """

    return {"message": "Successfully connected to the server. Hello World!"}

@fast_api.post("/user-speech")
def _API_user_speech(user_utterance: UploadFile, read_intro: bool = Form(...), TTS_language: str = Form(...)) -> FileResponse:
    """Endpoint of the API that receives user utterances.

    Endpoint of the API that receives user utterances.
    When a POST request is done to this endpoint with the user utterance attached, the `DemonstratorServer` will be instructed to transcribe the audio and to synthesize the transcription, which is sent back to the client.
    The API will wait to send a response until the `DemonstratorServer` has finished synthesizing.

    Args:
        user_utterance (UploadFile): The user utterance in the request sent by the client.

    Returns:
        FileResponse: The response from the server, which includes the synthesized audio.
    """

    with open(fast_api.demonstrator.asr_model.path_to_temp_user_utterance, "wb") as user_utterance_stream:
        shutil.copyfileobj(user_utterance.file, user_utterance_stream)

    fast_api.demonstrator.latest_user_utterance = fast_api.demonstrator.asr_model.path_to_temp_user_utterance
    fast_api.demonstrator.read_intro = read_intro
    fast_api.demonstrator.TTS_language = TTS_language
        
    while not fast_api.demonstrator.passed_server_response_barrier:
        time.sleep(.1)
    fast_api.demonstrator.passed_server_response_barrier = False
    
    os.remove(fast_api.demonstrator.asr_model.path_to_temp_user_utterance)
    
    return FileResponse(
        fast_api.demonstrator.tts_model.path_to_temp_tts,
        status_code=status.HTTP_201_CREATED,
        media_type="audio/mpeg",
        headers={
            "audio_length": str(fast_api.demonstrator.latest_tts_audio_length),
            "transcription": fast_api.demonstrator.latest_transcription
        }
    )  
