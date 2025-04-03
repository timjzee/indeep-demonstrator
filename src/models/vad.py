"""Defines the Voice Activity Detection models that can be used by a Demonstrator to detect the presence of speech."""

import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import pyaudio
import pydub

from models.model import AbstractModel

class VADModel(AbstractModel):
    """The abstract base VAD model class that all implementations inherit from.

    Attributes:
        path_to_resources: Path to the `resources` folder that contains various resources for inference, such as the Piper TTS model files and the audio used for warmup.
        path_to_temp_user_utterance: Path to the temporary file that contains the recorded end user speech.
    """

    def __init__(self) -> None:
        super().__init__()
        
        self.path_to_resources = Path(Path(__file__).parents[1], "resources")
        self.path_to_temp_user_utterance = Path(self.path_to_resources, "audio", f"temp_user_utterance.mp3")
    
    @abstractmethod
    def listen(self):
        """Listens for any detected speech input and records it if it is detected, so that it can be transcribed at a later stage.

        Raises:
            NotImplementedError: When called, since the `VADModel` is abstract.
        """

        raise NotImplementedError("VADModel class is abstract, please use an implementation.")
    
class SileroVAD(VADModel):
    """A `VADModel` implementation that uses Silero VAD to detect end user speech.

    A `VADModel` implementation that uses Silero VAD to detect end user speech.
    When the `listen()` method is called, it opens a PyAudio stream of audio and processes this stream in chunks.
    For every chunk of audio, the `SileroVAD` model checks the likelihood of end user speech being detected is above a set threshold.
    If it is, it assumes a user had started speaking and starts to record the audio chunks by saving them to a bytes array.
    From that point, for every audio chunk, it checks if the end user speech probability has decreased to below the threshold.
    If that is true for a set amount of sequential chunks, it is assumed that the speaker has stopped speaking.
    In that case, the `SileroVAD` instance will close the audio stream and save the bytes array as an audio file to the system, which is deleted after it has been transcribed.

    Attributes:
        AUDIO_STREAM_FORMAT: The PyAudio type of the stream chunks
        AUDIO_STREAM_CHANNELS: The number of channels that are recorded in the stream. 1 is mono, 2 is stereo.
        AUDIO_STREAM_SAMPLE_RATE: The number of frames per second of audio.
        AUDIO_STREAM_CHUNK_SIZE: The number of frames per audio chunk.
        MAX_SILENT_SECONDS: The amount of seconds that should pass below the end user speech confidence threshold before the audio stream is closed.
        MAX_SILENT_CHUNKS_COUNT: The amount of audio chunks that should pass below the end user speech confidence threshold before the audio stream is closed.
        SILENCE_CONFIDENCE_THRESHOLD: The percentile threshold that should be exceeded for an audio chunk to be considered as containing end user speech.
        model: The Silero VAD model for detecting user speech.
    """

    def __init__(self) -> None:
        super().__init__()
        
        self.AUDIO_STREAM_FORMAT = pyaudio.paInt16
        self.AUDIO_STREAM_CHANNELS = 1
        self.AUDIO_STREAM_SAMPLE_RATE = 16000
        self.AUDIO_STREAM_CHUNK_SIZE = 512
        
        SECONDS_PER_CHUNK_SIZE = self.AUDIO_STREAM_CHUNK_SIZE / self.AUDIO_STREAM_SAMPLE_RATE # Length of a chunk in seconds
        self.MAX_SILENT_SECONDS = 1.5
        self.MAX_SILENT_CHUNKS_COUNT = round(self.MAX_SILENT_SECONDS / SECONDS_PER_CHUNK_SIZE)
        self.SILENCE_CONFIDENCE_THRESHOLD = 0.8
        
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        
    def listen(self) -> tuple[str, float]:
        """Listens for any detected speech input and records it if it is detected, so that it can be transcribed at a later stage. 

        Returns:
            tuple[str, float]: The path to the audio file containing the recorded user speech and the length of the recorded speech in seconds.
        """

        user_is_speaking = False
        consecutive_silent_frames_count = 0
        user_utterance: bytes = b""
        
        audio_stream = pyaudio.PyAudio().open(
            format=self.AUDIO_STREAM_FORMAT,
            channels=self.AUDIO_STREAM_CHANNELS,
            rate=self.AUDIO_STREAM_SAMPLE_RATE,
            frames_per_buffer=self.AUDIO_STREAM_CHUNK_SIZE,
            input=True
        )
                
        while not user_is_speaking:
            user_spoke_confidence, _ = self._predicted_utterance_likelihood(audio_stream)
            
            if user_spoke_confidence > self.SILENCE_CONFIDENCE_THRESHOLD:
                user_is_speaking = True
        
        user_started_speaking_timestamp = time.time()
        while user_is_speaking:
            user_spoke_confidence, audio_chunk = self._predicted_utterance_likelihood(audio_stream)
            user_utterance += audio_chunk
            
            if user_spoke_confidence <= self.SILENCE_CONFIDENCE_THRESHOLD:
                consecutive_silent_frames_count +=1
                
                if consecutive_silent_frames_count >= self.MAX_SILENT_CHUNKS_COUNT:
                    user_is_speaking = False
                    
            if (consecutive_silent_frames_count > 0) & (user_spoke_confidence > self.SILENCE_CONFIDENCE_THRESHOLD):
                consecutive_silent_frames_count = 0
        audio_length = time.time() - user_started_speaking_timestamp - self.MAX_SILENT_SECONDS
        
        audio_stream.stop_stream()
        audio_stream.close()
        
        user_utterance = pydub.AudioSegment(
            user_utterance, 
            sample_width=2, 
            frame_rate=self.AUDIO_STREAM_SAMPLE_RATE, 
            channels=self.AUDIO_STREAM_CHANNELS
        )
        user_utterance.export(self.path_to_temp_user_utterance, format="mp3")
        
        return self.path_to_temp_user_utterance, audio_length
        
            
    def _predicted_utterance_likelihood(self, audio_stream: pyaudio.PyAudio) -> tuple[float, bytes]:
        """Returns the likelihood of speech being detected in an audio chunk by the Silero VAD model.

        Args:
            audio_stream (pyaudio.PyAudio): The audio chunk processed by the VAD model.

        Returns:
            tuple[float, bytes]: The likelihood of speech being detected in the audio chunk and the audio chunk itself.
        """

        audio_chunk = audio_stream.read(self.AUDIO_STREAM_CHUNK_SIZE, exception_on_overflow=False)
        processed_chunk = self._audio_stream_to_tensor(audio_chunk)
        
        user_spoke_confidence = self.model(processed_chunk, self.AUDIO_STREAM_SAMPLE_RATE).item()
        
        print(user_spoke_confidence)
        
        return user_spoke_confidence, audio_chunk
    
    def _audio_stream_to_tensor(self, audio_chunk: bytes) -> torch.Tensor:
        """Converts a PyAudio audio chunk to a PyTorch tensor.

        Args:
            audio_chunk (bytes): The audio chunk that should be converted to a tensor.

        Returns:
            torch.Tensor: The PyTorch tensor containing the bytes of the audio chunk.
        """

        audio_chunk = np.frombuffer(audio_chunk, np.int16)
        
        audio_chunk_abs_max = np.abs(audio_chunk).max()
        audio_chunk = audio_chunk.astype("float32")
        
        if audio_chunk_abs_max > 0:
            audio_chunk *= 1/32768
            
        audio_chunk.squeeze()
        
        return torch.from_numpy(audio_chunk)