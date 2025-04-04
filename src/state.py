"""Defines the states a Demonstrator instance can be in."""

from __future__ import annotations
import sys
import string
import time
import copy
from abc import ABC, abstractmethod

import torch

import demonstrator
import rest_api

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

        input("Press Enter to continue...")
        
        context.state = Listen()

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
            context.state = Listen()
    
class Warmup(AbstractState):
    """Prepares the Demonstrator instance's ASR model for use by loading it in memory and running it once."""
    
    def handle(self, context: demonstrator.Demonstrator):
        """Runs the state's logic.

        Args:
            context (demonstrator.Demonstrator): The Demonstrator instance to which the state is assigned.
        """

        print("I'm warming up...")
        
        context.asr_model.warmup()
        
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
        
        starting_timestamp = time.time()
        transcription, audio_length = context.asr_model.transcribe(context.latest_user_utterance)
        ending_timestamp = time.time()
        
        context.asr_model.metric_tracker.calculate_rtf(starting_timestamp, ending_timestamp, audio_length)        
        context.latest_transcription = transcription
        
        preprocessed_transcription = copy.copy(transcription)
        preprocessed_transcription = preprocessed_transcription.strip().lower()
        preprocessed_transcription = preprocessed_transcription.translate(str.maketrans("", "", string.punctuation))
        
        if preprocessed_transcription == context.asr_model.goodbye_transcription:
            context.state = SayGoodbye()
        else:
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
        audio_length = context.tts_model.synthesize(context.latest_transcription)
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
