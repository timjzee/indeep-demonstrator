"""Defines the Automatic Speech Recognition models that can be used by a Demonstrator to transcribe audio."""

from abc import abstractmethod
from pathlib import Path

import faster_whisper.tokenizer
import torch
import faster_whisper

# Allows AbstractModel to be imported in both src and eval folders
try:
    from src.models.model import AbstractModel
except:
    from models.model import AbstractModel

class ASRModel(AbstractModel):
    """The abstract base ASR model class that all implementations inherit from.

    Attributes:
        path_to_resources: Path to the `resources` folder, which contains the temporarily stored audio files of the user and TTS utterances.
        path_to_warmup_utterance: Path to the audio file that is used in the `Warmup` state (see the `state` module) to load the ASR model into memory.
        path_to_temp_user_utterance: Path to the temporarily stored audio file of the user utterance.
        goodbye_transcription: If the transcription of the user utterance matches this string, the program will enter initiate shutdown.
    """

    def __init__(self) -> None:
        super().__init__()
        
        self.path_to_resources = Path(Path(__file__).parents[1], "resources")
        self.path_to_warmup_utterance = Path(self.path_to_resources, "audio", "asr_warmup.wav")
        self.path_to_temp_user_utterance = Path(self.path_to_resources, "audio", f"temp_user_utterance.mp3")
        
        self.goodbye_transcription = "demonstrator"
    
    @abstractmethod
    def transcribe(self, audio):
        """Transcribes the passed audio using the ASR model.

        Args:
            audio: The user utterance to be transcribed.

        Raises:
            NotImplementedError: When called, since the `ASRModel` is abstract and should not be used.
        """

        raise NotImplementedError("ASRModel class is abstract, please use an implementation.")
    
    @abstractmethod
    def warmup(self):
        """Loads the ASR model into memory so that it can be swiftly accessed during actual inference.

        Raises:
            NotImplementedError: When called, since the `ASRModel` is abstract and should not be used.
        """

        raise NotImplementedError("ASRModel class is abstract, please use an implementation.")
        
class FasterWhisper(ASRModel):
    """An `ASRModel` implementation that user faster-whisper for transcriptions.

    An `ASRModel` implementation that user faster-whisper for transcriptions.
    This implementation uses the `faster_whisper` library to instantiate a Whisper model of the size specified.
    The model is capable of multilingual and cross-lingual transcriptions.
    For example, it can transcribe both Dutch and English, but can also transcribe speech of one into the other.
    This implementation encourages the Whisper model to transcribe numbers as words, so that it is easier to synthesize TTS audio for the transcriptions.

    Attributes:
        model_size: The size of the FasterWhisper model, in ["tiny", "base", "small", "medium", "large-v2"].
        language: The language faster-whisper will transcribe in, and thus the language that faster-whisper will expect to receive an utterance in. If another language is spoken in an user utterance, it will be translated to the language specified.
        model: The actual faster-whisper model from the `faster_whisper` library.
        number_tokens: The Whisper tokens that correspond to the numerals [0-9]. It is used to encourage the model to transcribe numbers as words.
    
    NOTE: Findings about the model's capabilities
    - Is capable of transcribing nonsense words ("Tradiculeren", "neoëuner"), words pronounced erroneously ("diarysation" instead of "diarization"), laughter ("GELACH"), and swear words.
    - When transcribing non-words such as laughter, filled pauses ("ehm/uhm/hmm"), and repair initiators ("huh?"), processing times may increase significantly.
    """
    def __init__(self, device: str, model_size: str, language: str = "nl") -> None:
        super().__init__()
        
        self.model_size = model_size
        self.language = language
        
        if device[-1].isdigit():
            self.model = faster_whisper.WhisperModel(self.model_size, device="cuda", device_index=[int(device[-1])], compute_type="float32")
        else:
            self.model = faster_whisper.WhisperModel(self.model_size, device=device, compute_type="float32")
        
        if self.language:
            tokenizer = faster_whisper.tokenizer.Tokenizer(tokenizer=self.model.hf_tokenizer, task="transcribe", language=self.language, multilingual=True)

            # Encourages FasterWhisper to transcribe numbers as words, so that the TTS can read them out loud
            self.number_tokens = [i for i in range(tokenizer.eot) if all(char in "0123456789" for char in tokenizer.decode([i]).removeprefix(" "))]
    
    def transcribe(self, audio: torch.Tensor | Path | str, print_transcription: bool = True) -> tuple[str, float]:
        """Transcribes a user utterance using the FasterWhisper model.

        Args:
            audio (torch.Tensor | Path | str): The user utterance to be transcribed. If of type torch.Tensor, this parameter is the utterance represented as a tensor. If of type Path or str, this parameter is the absolute path to the file recording of the user utterance.
            print_transcription (bool, optional): Whether or not to print the transcription to the console. Defaults to True.

        Returns:
            tuple[str, float]: The transcription and length of the audio that was transcribed in seconds.
        """

        with torch.no_grad():
            if self.language:
                transcription_segments, transcription_info = self.model.transcribe(audio, language=self.language, suppress_tokens=[-1]+self.number_tokens)
            else:
                transcription_segments, transcription_info = self.model.transcribe(audio, language=self.language)
                print(f"Detected language: {transcription_info.language}")
            transcription = ""
        
        for segment in transcription_segments:
            transcription += segment.text
        
        transcription = transcription.strip()
        
        if print_transcription:
            print(transcription)
        
        return (transcription, transcription_info.duration, transcription_info.language)
    
    def warmup(self, print_transcription=True):
        """Loads the FasterWhisper model into memory before inference.

        Args:
            print_transcription (bool, optional): Whether or not to print the transcription to the console. Defaults to True.
        """

        self.transcribe(self.path_to_warmup_utterance, print_transcription)