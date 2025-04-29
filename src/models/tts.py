"""Defines the Text-to-Speech models that can be used by a Demonstrator to synthesize audio."""

import os
import random
import wave
import json
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import pydub
import torch
import torchaudio
import transformers

from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf

try:
    import piper
except ModuleNotFoundError:
    print("WARNING: Module 'piper' could not be imported. This is likely because you are using a Windows machine, and the Piper TTS does not run on Windows.")

from models.model import AbstractModel

class TTSModel(AbstractModel):
    """The abstract base TTS model class that all implementations inherit from.

    Attributes:
        device: The device on which the model will be run.
        path_to_resources: Path to the `resources` folder that contains various resources for inference, such as the Piper TTS model files and the audio used for warmup.
        path_to_temp_tts: Path to the temporary file containing the synthesized TTS audio.
        path_to_messages: Path to the JSON file containing the standard messages for the Demonstrator for a specific language.
        empty_transcription_message: The message synthesized by the Demonstrator when a transcription is empty and cannot be parroted.
        fully_unpronounceable_message: The message synthesized by the Demonstrator when a transcription is fully unpronounceable for the TTS system.
        partially_unpronounceable_message: The message synthesized by the Demonstrator when a transcription is partially unpronounceable for the TTS system. The synthesized audio will be the part of the transcription that can be pronounced, preceded by this message.
        goodbye_texts: The set of messages that can be synthesized by the Demonstrator when it enters the GoodbyeState. One of these is selected at random.
    """

    def __init__(self, device: str, language: str = "nl") -> None:
        super().__init__()
        
        self.device = device
        
        self.path_to_resources = Path(Path(__file__).parents[1], "resources")
        self.path_to_temp_tts = Path(self.path_to_resources, "audio", "temp_tts.mp3")
        self.path_to_messages = Path(Path(__file__).parents[2], "data", "tts_messages", f"{language}.json")

        with open(self.path_to_messages, encoding="utf-8") as stream:
            messages = json.loads(stream.read())
        
            self.empty_transcription_message = messages["empty_transcription_message"]
            self.fully_unpronounceable_message = messages["fully_unpronounceable_message"]
            self.partially_unpronounceable_message = messages["partially_unpronounceable_message"]
            self.goodbye_texts = messages["goodbye_texts"]
    
    @abstractmethod
    def synthesize(self, text: str, tone: str) -> float:
        """Synthesizes the passed text using a Text-to-Speech module.

        Args:
            text (str): The text for which audio should be synthesized.
            tone (str): The tone in which the text should be synthesized.

        Raises:
            NotImplementedError: When called, since the `TTSModel` is abstract.
        """

        raise NotImplementedError("TTSModel class is abstract, please use an implementation.")
    
    def say_goodbye(self) -> float:
        """Randomly selects a goodbye message and synthesizes audio for it.

        Returns:
            float: The length of the synthesized audio in seconds.
        """

        return self.synthesize(random.choice(self.goodbye_texts))
        
class MMS(TTSModel):
    """A `TTSModel` implementation that uses Facebook's MMS-TTS model to synthesize audio.

    A `TTSModel` implementation that uses Facebook's MMS-TTS model to synthesize audio.
    The MMS-TTS model is an E2E neural model that takes a string and synthesizes speech for it.
    The synthesis cannot be controlled, however, so the synthesized voice may fluctuate between, for example, perceived speaker sex.
    It supports many different languages, given the correct language code (which consists of 3 letters and thus is different from the codes used for the Demonstrator!)

    Attributes:
        model: The MMS-TTS model, a type of VitsModel.
        tokenizer: The MMS-TTS model's tokenizer.
    """
    
    def __init__(self, device: str, language: str = "nl") -> None:
        super().__init__(device)

        mms_tts_language_key = self._get_mms_tts_language_key(language)
        model_name = f"facebook/mms-tts-{mms_tts_language_key}"
        
        self.model = transformers.VitsModel.from_pretrained(model_name).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
    def synthesize(self, text: str, tone: str) -> float:
        """Synthesizes the passed text as speech using the MMS-TTS module.

        Args:
            text (str): The text that should be synthesized to speech.

        Returns:
            float: The length of the synthesized audio in seconds.
        """

        if not text:
            text = self.empty_transcription_message
            
        if "..." in text:
            if text.strip() == "...":
                text = self.fully_unpronounceable_message
            else:
                text = self.partially_unpronounceable_message + text.replace("...", "")

        tokenized_input = self.tokenizer(text, return_tensors="pt").to(self.device)
                
        with torch.no_grad():
            speech = self.model(**tokenized_input).waveform.cpu()
        
        torchaudio.save(str(self.path_to_temp_tts), speech, self.model.config.sampling_rate)
        
        temp_tts_info = torchaudio.info(str(self.path_to_temp_tts))
        audio_length = temp_tts_info.num_frames / temp_tts_info.sample_rate
        
        return audio_length
    
    def _get_mms_tts_language_key(self, language: str) -> str:
        """Returns the language key used by MMS-TTS to determine the language that should be synthesized for, given the language key used by the Demonstrator.

        Args:
            language (str): The Demonstrator language key of the language speech should be synthesized for.

        Raises:
            ValueError: When a non-supported language key is passed as an argument.

        Returns:
            str: The MMS-TTS language key for the provided Demonstrator language key.
        """

        if language == "nl":
            return "nld"
        elif language == "en":
            return "eng"
        else:
            raise ValueError(f"Language code {language} is not a supported language for the Demonstrator.")

class Piper(TTSModel):
    """A `TTSModel` implementation that uses Rhasspy's Piper TTS to synthesize audio.

    A `TTSModel` implementation that uses Rhasspy's Piper TTS to synthesize audio.
    The Piper TTS module uses relatively small neural networks to synthesize speech.
    A Piper model is stored as a model file alongside its configuration file, and every model maps to a specific speaker voice in a specific language.
    Different models are thus used across different languages and different speakers.
    These files are stored in the `resources` folder (`src/resources/models/`).

    Attributes:
        model_name: The name of the Piper model used.
        voice_id: The voice ID of the selected Piper model, which is sometimes used to select a specific speaker profile.
        model: The loaded Piper TTS model.
    """

    def __init__(self, device: str, language: str = "nl") -> None:
        super().__init__(device)
        
        self.model_name, self.voice_id = self._get_model_name_by_language(language)
        
        path_to_model = Path(self.path_to_resources, "models", f"{self.model_name}.onnx")
        self.model = piper.voice.PiperVoice.load(path_to_model)
        
    def synthesize(self, text: str, tone: str) -> float:
        """Synthesizes the passed text as speech using the Piper TTS module.

        Args:
            text (str): The text that should be synthesized to speech.

        Returns:
            float: The length of the synthesized audio in seconds.
        """

        if not text:
            text = self.empty_transcription_message
        
        path_to_temp_wav = str(Path(self.path_to_temp_tts.parents[0], "temp_tts.wav"))
        
        with wave.open(path_to_temp_wav, mode="wb") as temp_tts_wav:
            self.model.synthesize(text, temp_tts_wav, self.voice_id)
        
        pydub.AudioSegment.from_wav(path_to_temp_wav).export(self.path_to_temp_tts, format="mp3")
        os.remove(path_to_temp_wav)
        
        temp_tts_info = torchaudio.info(self.path_to_temp_tts)
        audio_length = temp_tts_info.num_frames / temp_tts_info.sample_rate
        
        return audio_length
    
    def _get_model_name_by_language(self, language: str) -> tuple[str, Optional[int]]:
        """Returns the Piper TTS model name given the language speech should be synthesized for.

        Args:
            language (str): The language in which speech will be synthesized.

        Raises:
            ValueError: When a non-supported language code is passed.

        Returns:
            tuple[str, Optional[int]]: The name of the Piper TTS model used for the language specified and the optionally the voice ID if the model supports it.
        """

        if language == "nl":
            return "nl_BE-rdh-medium", None
        elif language == "en":
            return "en_US-ryan-high", None
        else:
            raise ValueError(f"Language code {language} is not a supported language for the Demonstrator.")

class Parler(TTSModel):
    """A `TTSModel` implementation that uses Parler's TTS to synthesize audio.

    A `TTSModel` implementation that uses Parler's TTS to synthesize audio.
    The Parler TTS module uses relatively small neural networks to synthesize speech.
    A Parler model is stored as a model file alongside its configuration file, and every model maps to a specific speaker voice in a specific language.
    Different models are thus used across different languages and different speakers.
    These files are stored in the `resources` folder (`src/resources/models/`).

    Attributes:
        model_name: The name of the Parler model used.
        voice_id: The voice ID of the selected Parler model, which is sometimes used to select a specific speaker profile.
        model: The loaded Parler TTS model.
    """

    def __init__(self, device: str, language: str = "nl") -> None:
        super().__init__(device)
        self._handle_language(language)
        self.model_name = "parler-tts/parler-tts-mini-expresso"
        self.voice_id = "Jerry"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_name, attn_implementation="eager").to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def synthesize(self, text: str, tone: str) -> float:
        """Synthesizes the passed text as speech using the Piper TTS module.

        Args:
            text (str): The text that should be synthesized to speech.
            tone (str): The tone in which the text should be synthesized.

        Returns:
            float: The length of the synthesized audio in seconds.
        """

        if not text:
            text = self.empty_transcription_message

        description = self.voice_id + " speaks in a " + tone + " tone with clear articulation."

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, bos_token_id=1025, decoder_start_token_id=1025, do_sample=True, eos_token_id=1024, max_new_tokens=2580, min_new_tokens=10, pad_token_id=1024)

        path_to_temp_wav = str(Path(self.path_to_temp_tts.parents[0], "temp_tts.wav"))
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(path_to_temp_wav, audio_arr, self.model.config.sampling_rate)

        audio = pydub.AudioSegment.from_wav(path_to_temp_wav)

        if os.path.exists(self.path_to_temp_tts):
            audio_pre = pydub.AudioSegment.from_mp3(str(self.path_to_temp_tts))
            audio = audio_pre + audio

        audio.export(self.path_to_temp_tts, format="mp3")
        os.remove(path_to_temp_wav)
        
        temp_tts_info = torchaudio.info(str(self.path_to_temp_tts))
        audio_length = temp_tts_info.num_frames / temp_tts_info.sample_rate
        
        return audio_length

    def _handle_language(self, language: str) -> None:
        """Handles the language code.

        Args:
            language (str): The language in which speech will be synthesized.
        """

        if language == "nl":
            raise ValueError(f"Language code {language} is not a supported language for the Parler TTS Model.")