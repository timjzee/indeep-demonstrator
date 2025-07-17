"""Defines the states a Demonstrator instance can be in."""

from __future__ import annotations
import sys
import os
import random
import string
import time
import copy
from abc import ABC, abstractmethod

import torch

import demonstrator
import rest_api
from num2words import num2words
from pydub import AudioSegment
from pathlib import Path

emo_dict = {
    "neutral": "neutraal",
    "happy": "blij",
    "sad": "verdrietig",
    "angry": "boos",
    "fearful": "bang",
    "disgusted": "walgend",
    "surprised": "verbaasd",
    "calm": "kalm",
}

class AbstractState(ABC):
    """The abstract base state all other states inherit from.

    Raises:
        NotImplementedError: When called, since the class is abstract and should not be called.
    """

    @abstractmethod
    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        raise NotImplementedError("The AbstractState is not a valid state for a client to be in. Please call an implementation of the AbstractState.")
    
class Idle(AbstractState):
    """Makes the demonstrator instance wait for manual input like a keypress."""

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm idle...")

        input_text = input("Press Enter to continue...")
        
        if input_text == "n":
            context.TTS_language = "nl"
            context.emo_list = list(emo_dict.values())
            context.state = Intro()
        elif input_text == "e":
            context.TTS_language = "en"
            context.emo_list = list(emo_dict.keys())
            context.state = Intro()
        else:
            context.state = Listen()
            context.read_intro = False

class Intro(AbstractState):
    """Introduces the demonstrator."""

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic."""

        # Create dummy audio. then serverside skip asr and tts intro text based on read_intro flag.
        path_to_resources = Path(__file__).resolve().parent / "resources"
        path_to_temp_user_utterance = Path(path_to_resources, "audio", f"temp_user_utterance.mp3")
        silent = AudioSegment.silent(duration=50)
        os.makedirs(os.path.dirname(path_to_temp_user_utterance), exist_ok=True)
        silent.export(path_to_temp_user_utterance, format="mp3", bitrate="32k")

        context.latest_user_utterance = path_to_temp_user_utterance
        context.read_intro = True

        if isinstance(context, demonstrator.DemonstratorClient):
            context.state = RESTRequest()
        
        if isinstance(context, demonstrator.DemonstratorApp):
            if context.TTS_language == "nl":
                intro_text = "Hallo, ik ben de InDiep demo! Zeg iets met emotie, en ik probeer te raden welke emotie het was."
            elif context.TTS_language == "en":
                intro_text = "Hi, I am the InDeep demo! Please say something with emotion, and I'll try to guess which emotion it was."
            
            audio_length = context.tts_model.synthesize(intro_text, "neutral", context.TTS_language)
            context.state = Speak()


class Wakeup(AbstractState):
    """The initial state all Demonstrator instances start in, which assigns the first state with logic to the instance."""

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm waking up...")
        
        if isinstance(context, (demonstrator.DemonstratorApp, demonstrator.DemonstratorServer)):
            context.state = Warmup()
        
        if isinstance(context, demonstrator.DemonstratorClient):
            if context.activation == "auto":
                context.state = Listen()
            else:   # if it is "input"
                context.state = Idle()
    
class Warmup(AbstractState):
    """Prepares the Demonstrator instance's ASR model for use by loading it in memory and running it once."""
    
    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm warming up...")
        
        context.asr_model.warmup()
        context.ser_model.warmup()
        
        if isinstance(context, demonstrator.DemonstratorServer):           
            context.state = RESTAwait()
        
        if isinstance(context, demonstrator.DemonstratorApp):
            if context.activation == "auto":
                context.state = Listen()
            else:   # if it is "input"
                context.state = Idle()
        
class Listen(AbstractState):
    """Makes the Demonstrator instance listen to input audio and record any utterances.
    
    Makes the Demonstrator instance listen to input audio and record any utterances.
    The Voice Activity Detection model's Real-Time Factor is calculated here."""

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm listening...")
        
        starting_timestamp = time.time()
        user_utterance, audio_length = context.vad_model.listen()
        ending_timestamp = time.time()
        
        context.vad_model.metric_tracker.calculate_rtf(starting_timestamp, ending_timestamp, audio_length)
        context.latest_user_utterance = user_utterance
        
        if isinstance(context, demonstrator.DemonstratorClient):
            context.state = RESTRequest()
        
        if isinstance(context, demonstrator.DemonstratorApp):
            context.state = Transcribe()
        
class Transcribe(AbstractState):
    """Instructs the Demonstrator instance to transcribe a user utterance.

    Instructs the Demonstrator instance to transcribe a user utterance.
    The Automatic Speech Recognition model's Real-Time Factor is recorded here.
    A copy of the transcription is also preprocessed and compared to the string that activates the `SayGoodbye` state, and if they are the same, the program will move to that state.
    Otherwise, it will start to synthesize the transcription.
    """

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm transcribing...")
        print("Intro: ", context.read_intro)

        if context.read_intro:  # if intro, we skip the ASR and TTS intro text
            # remove previous tts audio
            if os.path.exists(context.tts_model.path_to_temp_tts):
                os.remove(context.tts_model.path_to_temp_tts)
            # Set model language for future transcriptions
            context.asr_model.language = context.TTS_language
            context.read_intro = False
            if context.TTS_language == "nl":
                intro_text = "Hallo, ik ben de InDiep demo! Zeg iets met emotie, en ik probeer te raden welke emotie het was."
            elif context.TTS_language == "en":
                intro_text = "Hi, I am the InDeep demo! Please say something with emotion, and I'll try to guess which emotion it was."
            
            audio_length = context.tts_model.synthesize(intro_text, "neutral", context.TTS_language)
            context.latest_tts_audio_length = audio_length
            context.latest_transcription = intro_text

            if isinstance(context, demonstrator.DemonstratorServer):
                context.state = RESTResponse()
            
            if isinstance(context, demonstrator.DemonstratorApp):
                context.state = Speak()
        
        else:
            starting_timestamp = time.time()
            transcription, audio_length, recog_lang = context.asr_model.transcribe(context.latest_user_utterance)
            ending_timestamp = time.time()
            
            context.asr_model.metric_tracker.calculate_rtf(starting_timestamp, ending_timestamp, audio_length)        
            context.latest_transcription = transcription
            context.TTS_language = recog_lang
            
            preprocessed_transcription = copy.copy(transcription)
            preprocessed_transcription = preprocessed_transcription.strip().lower()
            preprocessed_transcription = preprocessed_transcription.translate(str.maketrans("", "", string.punctuation))
            
            # here we remove temporary tts file so the tts module knows whether to concatenate synthesized audio to the existing one or not
            if os.path.exists(context.tts_model.path_to_temp_tts):
                os.remove(context.tts_model.path_to_temp_tts)

            if preprocessed_transcription == context.asr_model.goodbye_transcription:
                context.state = SayGoodbye()
            else:
                context.state = RecognizeEmo()

class RecognizeEmo(AbstractState):
    """Instructs the Demonstrator instance to recognize the emotion of a user utterance.

    Instructs the Demonstrator instance to recognize the emotion of a user utterance.
    """

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm recognizing the emotion...")
        
        emo_label, emo_score, oth_label = context.ser_model.recognize(context.latest_user_utterance)
        
        context.latest_emo_label = emo_label
        context.latest_emo_score = num2words(emo_score)
        context.latest_other_label = oth_label if context.tts_model.name == "parler" else "calm" if emo_label == "neutral" else "neutral"
        if context.TTS_language == "nl":
            context.latest_emo_score = num2words(emo_score, lang="nl")
            context.latest_emo_label = emo_dict[emo_label]
            context.latest_other_label = emo_dict[context.latest_other_label]
            context.latest_text_to_synthesize = "Op basis van je stem ben ik {} procent zeker dat je {} klonk. Als je in plaats daarvan {} was, zou je zo hebben geklonken:".format(context.latest_emo_score, context.latest_emo_label, context.latest_other_label)
        else:
            context.latest_text_to_synthesize = "Based on your tone of voice, I am {} percent certain that you sounded {}. If you were {} instead, you would have sounded like this:".format(context.latest_emo_score, context.latest_emo_label, context.latest_other_label)

        context.state = Synthesize()

class Synthesize(AbstractState):
    """Instructs the Demonstrator instance to synthesize the transcription provided to it.

    Instructs the Demonstrator instance to synthesize the transcription provided to it.
    The Text-to-Speech model's Real-Time Factor is also calculated here.
    """

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm synthesizing speech...")
        
        starting_timestamp = time.time()
        audio_length = context.tts_model.synthesize(context.latest_text_to_synthesize, "neutral", context.TTS_language) # could be changed to a faster tts mode, context.fast_tts_model
        audio_length = context.tts_model.synthesize(context.latest_transcription, context.latest_other_label, context.TTS_language)

        # select emo suggestion
        context.emo_list.remove(context.latest_emo_label) if context.latest_emo_label in context.emo_list else None
        if len(context.emo_list) == 0:
            context.emo_list = list(emo_dict.keys()) if context.TTS_language == "en" else list(emo_dict.values())
            context.emo_list.remove(context.latest_emo_label)

        print("Emo list: ", context.emo_list)
        print(context.TTS_language)
        emo_sug = random.choice(context.emo_list)
        context.emo_list.remove(emo_sug)
        if context.TTS_language == "nl":
            if emo_sug == "walgend":
                audio_length = context.tts_model.synthesize("Probeer nu eens vol walging te klinken.", "neutral", context.TTS_language)
            else:
                audio_length = context.tts_model.synthesize("Probeer nu eens heel {} te klinken.".format(emo_sug), "neutral", context.TTS_language)
        else:
             audio_length = context.tts_model.synthesize("Now try to sound very {}.".format(emo_sug), "neutral", context.TTS_language)

        ending_timestamp = time.time()
        
        context.tts_model.metric_tracker.calculate_rtf(starting_timestamp, ending_timestamp, audio_length)
        context.latest_tts_audio_length = audio_length
        
        if isinstance(context, demonstrator.DemonstratorServer):
            context.state = RESTResponse()
            
        if isinstance(context, demonstrator.DemonstratorApp):
            context.state = Speak()
        
class Speak(AbstractState):
    """Instructs the Demonstrator instance to play back the synthesized audio to the user."""

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm speaking...")
        
        context.playback_module.playback(context.latest_tts_audio_length)

        #if context.latest_text_to_synthesize != context.latest_transcription:
        #    context.latest_text_to_synthesize = context.latest_transcription
        #    context.state = Synthesize()
        #else:
        if context.activation == "auto":
            context.state = Listen()
        else:   # if it is "input"
            context.state = Idle()
    
class SayGoodbye(AbstractState):
    """Instructs the Demonstrator instance to tell the end user goodbye and to then shut down the program.

    Instructs the Demonstrator instance to tell the end user goodbye and to then shut down the program.
    A `DemonstratorApp` instance will first play back a goodbye message.
    Then, the instance prints the RTFs of its models tracked so far, alongside the lengths of the user utterances.
    It will then quit the program.
    This also shuts down the API, which ensures that the client is also aware of the shutdown, which will shut down itself in turn.
    """

    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm saying goodbye...")
        
        if isinstance(context, demonstrator.DemonstratorApp):
            audio_length = context.tts_model.say_goodbye()
            context.playback_module.playback(audio_length)
        
        print("Metrics:")
        
        if isinstance(context, demonstrator.DemonstratorApp):
            print(f" VAD: {context.vad_model.metric_tracker.rtfs}")
            print(f"    {context.vad_model.metric_tracker.audio_lengths}")
            
        print(f" ASR: {context.asr_model.metric_tracker.rtfs}")
        print(f"    {context.asr_model.metric_tracker.audio_lengths}")
        print(f" TTS: {context.tts_model.metric_tracker.rtfs}")
        print(f"    {context.tts_model.metric_tracker.audio_lengths}")
            
        sys.exit(0)

class RESTRequest(AbstractState):
    """Instructs a client Demonstrator to prepare and send a REST request with the user utterance attached to the server."""

    def handle(self, context: demonstrator.DemonstratorClient):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm sending a REST request to the server...")
        
        print("Intro: ")
        print(context.read_intro)
        audio_length, transcription = rest_api.send_user_speech_request(context)
        context.latest_tts_audio_length = audio_length
        print(f"{transcription}")
        
        context.state = Speak()
    
class RESTAwait(AbstractState):
    """Instructs a Demonstrator server to wait for an incoming REST request from a client."""

    def handle(self, context: demonstrator.DemonstratorServer):       
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """
            
        print("I'm awaiting a request from the client...")
        
        while context.latest_user_utterance is None:
            time.sleep(.1)

        print(" Request received!")
        
        context.state = Transcribe()

class RESTResponse(AbstractState):
    """Instructs a Demonstrator server to send a REST response to the client.
    
    Instructs a Demonstrator server to send a REST response to the client.
    Also cleans the Demonstrator server up by emptying the CUDA cache and emptying the latest user utterance."""

    def handle(self, context: demonstrator.DemonstratorServer):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm sending a REST response to the client...")
        
        context.passed_server_response_barrier = True
        context.latest_user_utterance = None
        torch.cuda.empty_cache()
        
        context.state = RESTAwait()
