import jiwer

class MetricTracker:
    """Tracks the metrics of a Demonstrator model.

    The MetricTracker calculates the running metrics of any of the Demonstrator's models (VAD, ASR, TTS, ...).
    It tracks the Real-Time Factor (RTF) for all models during runtime and evaluation, as well as the Word Error Rate (WER) for ASR models in an evaluation setup.

    Attributes:
        rtfs: A list of the RTF values recorded for a model.
        wers: A list of the WER values recorded for an ASR model.
        current_predicted_text: In an evaluation setup, the transcription made by the ASR model of the last processed utterance.
        current_target_text: In an evaluation setup, the the ground truth transcription of the last processed utterance.
        audio_lengths: The length of all user utterances encountered so far. Used for calculating the Real-Time Factor.
    """

    def __init__(self) -> None:
        self.rtfs = []
        self.wers = []

        self.current_predicted_text = None
        self.current_target_text = None
        self.audio_lengths = []
        
    def calculate_rtf(self, starting_timestamp: float, ending_timestamp: float, audio_length: float) -> float:
        """Calculates the Real-Time Factor for a single utterance.

        Args:
            starting_timestamp (float): The UTC timestamp recorded before the start of an utterance.
            ending_timestamp (float):  The UTC timestamp recorded after the end of an utterance.
            audio_length (float): The length of the recorded utterance.

        Returns:
            float: The Real-Time Factor for the latest user utterance.
        """

        current_rtf = real_time_factor(
            processing_time=(ending_timestamp - starting_timestamp), 
            audio_length=audio_length
        )
        self.rtfs.append(current_rtf)
        self.audio_lengths.append(audio_length)
        
        return current_rtf
    
    def calculate_wer(self) -> float:
        """Calculates the Word Error Rate for a single utterance.

        Returns:
            float: The Word Error Rate for the latest user utterance.
        """

        current_wer = word_error_rate(self.current_predicted_text, self.current_target_text)
        self.wers.append(current_wer)
        
        return current_wer
    
    def get_mean_rtf(self) -> float:
        """Returns the mean of all RTF values recorded in the current Demonstrator run.

        Raises:
            ValueError: When no RTF values have been recorded yet.

        Returns:
            float: The mean RTF.
        """

        if len(self.rtfs) == 0:
            raise ValueError("No metrics have been recorded yet, so a mean cannot be calculated.")
        else:
            return sum(self.rtfs)/len(self.rtfs)
        
    def get_mean_wer(self) -> float:
        """Returns the mean of all WER values recorded in the current Demonstrator run.

        Raises:
            ValueError: When no WER values have been recorded yet.

        Returns:
            float: The mean WER.
        """

        if len(self.wers) == 0:
            raise ValueError("No metrics have been recorded yet, so a mean cannot be calculated.")
        else:
            return sum(self.wers)/len(self.wers)

def word_error_rate(predicted_text: str, target_text: str) -> float:
    """Preprocesses texts for calculating the Word Error Rate, then calculates and returns the WER.

    Args:
        predicted_text (str): The text that is compared to the ground truth text, for which the WER is calculated.
        target_text (str): The ground truth text.

    Returns:
        float: The Word Error Rate between the two texts.
    """

    wer_string_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ])
    
    return jiwer.wer(target_text, predicted_text, reference_transform=wer_string_transforms, hypothesis_transform=wer_string_transforms)

def real_time_factor(processing_time: float, audio_length: float) -> float:
    """Calculates the Real-Time Factor.

    Args:
        processing_time (float): The amount of time it took a model to process an utterance in seconds.
        audio_length (float): The duration of the utterance in seconds.

    Returns:
        float: The Real-Time Factor.
    """

    return processing_time / audio_length