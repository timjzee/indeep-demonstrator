"""Defines the Speech Emption Recognition models that can be used by a Demonstrator to recognize the emotion of utterances."""

from abc import abstractmethod
from pathlib import Path

import torchaudio
import torch
from transformers import pipeline
import requests
import os
from num2words import num2words

# Allows AbstractModel to be imported in both src and eval folders
try:
    from src.models.model import AbstractModel
except:
    from models.model import AbstractModel


class SERModel(AbstractModel):
    """The abstract base SER model class that all implementations inherit from.

    Attributes:
        path_to_resources: Path to the `resources` folder, which contains the temporarily stored audio files of the user and TTS utterances.
        path_to_warmup_utterance: Path to the audio file that is used in the `Warmup` state (see the `state` module) to load the SER model into memory.
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
    def recognize(self, audio):
        """Transcribes the passed audio using the SER model.

        Args:
            audio: The user utterance to be recocgnized.

        Raises:
            NotImplementedError: When called, since the `SERModel` is abstract and should not be used.
        """

        raise NotImplementedError("SERModel class is abstract, please use an implementation.")
    
    @abstractmethod
    def warmup(self):
        """Loads the SER model into memory so that it can be swiftly accessed during actual inference.

        Raises:
            NotImplementedError: When called, since the `SERModel` is abstract and should not be used.
        """

        raise NotImplementedError("SERModel class is abstract, please use an implementation.")

class RAVDESS(SERModel):
    """An `SERModel` implementation that uses wav2vec2 finetuned on the RAVDESS dataset for Emotion Recognition."""
    def __init__(self, device: str) -> None:
        super().__init__()

        url = "https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition/resolve/main/pytorch_model.bin"
        model_file_path = str(self.path_to_resources) + "/models/pytorch_model.bin"

        if os.path.exists(model_file_path):
            print("RAVDESS model already downloaded.")
        else:
            print("Downloading RAVDESS model...")
            os.makedirs(str(self.path_to_resources) + "/models", exist_ok=True)
            r = requests.get(url)
            with open(model_file_path, 'wb') as f:
                f.write(r.content)

        device_int = 0 if device == "cuda" else int(device[-1]) if device[-1].isdigit() else -1

        self.classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device_int)
        self.classifier.model.projector = torch.nn.Linear(1024, 1024, bias=True)
        self.classifier.model.classifier = torch.nn.Linear(1024, 8, bias=True)
        torch_state_dict = torch.load(model_file_path, map_location=torch.device(device))
        self.classifier.model.projector.weight.data = torch_state_dict['classifier.dense.weight']
        self.classifier.model.projector.bias.data = torch_state_dict['classifier.dense.bias']
        self.classifier.model.classifier.weight.data = torch_state_dict['classifier.output.weight']
        self.classifier.model.classifier.bias.data = torch_state_dict['classifier.output.bias']

    def recognize(self, audio):
        """Transcribes the passed audio using the SER model.

        Args:
            audio: The user utterance to be recognized.
        """

        waveform, sample_rate = torchaudio.load(str(audio))
        np_waveform = waveform.squeeze().numpy()

        result = self.classifier(np_waveform, top_k=8)
        print(result)
        label = result[0]['label']
        score = int(result[0]['score'] * 100)
        last_label = "disgusted" if result[-1]['label'] == "disgust" else result[-1]['label']
        return label, num2words(score), last_label

    def warmup(self):
        """Loads the SER model into memory so that it can be swiftly accessed during actual inference."""
        
        self.recognize(self.path_to_warmup_utterance)

