"""A module for playing Demonstrator audio on a client/app."""

import os
import time
from pathlib import Path

from playsound import playsound

class PlaybackModule:
    """A module for playing back Demonstrator audio.

    Attributes:
        path_to_resources: Path to the `resources` folder, which contains the temporarily stored audio files of the user and TTS utterances.
        path_to_temp_tts: Path to the temporarily stored TTS utterance.
    """

    def __init__(self):
        self.path_to_resources = Path(Path(__file__).parents[0], "resources")
        self.path_to_temp_tts = Path(self.path_to_resources, "audio", "temp_tts.mp3")
    
    def playback(self, audio_length: float) -> None:
        """Plays back the temporarily stored TTS utterance file.

        Args:
            audio_length (float): The length of the temporarily stored TTS utterance in seconds.
        """

        playsound(str(self.path_to_temp_tts), block=False)
        time.sleep(audio_length)
        os.remove(self.path_to_temp_tts)
        