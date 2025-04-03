"""Defines the possible evaluation methods for the Demonstrator and runs one of them."""

import sys
import os
import time
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parents[1]
HF_CACHE_PATH = Path(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = HF_CACHE_PATH

# This is a bodge/malpractice but the imports will not work without it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.runtime
import src.metrics
from src.models.asr import FasterWhisper

SEED = 3131
EVAL_CONFIG_PATH = Path(PROJECT_ROOT, "configs", "eval_configs.yaml")
EVAL_RESULTS_PATH = Path(PROJECT_ROOT, "eval", "results")
EVAL_DATA_ROOT = Path(PROJECT_ROOT, "data", "audio", "eval")

def asr_evaluation(eval_config: dict, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A function that runs a full evaluation setup.
    
    A function that runs a full evaluation setup on a set of FasterWhisper models.
    It takes the model sizes defined in `eval_config`, creates FasterWhisper models of the sizes specified and measures their transcription performance.
    The models' WER and RTFs are measured on the ground truths found in /data/manifests/asr_eval.json.
    The metrics of all evaluated model sizes are stored in two dataframes.

    Args:
        eval_config (dict): A YAML file loaded as a dict that contains the configurations for the evaluation setup. The YAML file itself can be found in /configs/eval_configs.yaml and defines the model sizes evaluated, the dataset with ground truths, and whether to include edge case recordings. 
        device (str): The device that the FasterWhisper models will be run on, in ["cpu", "cuda", "cuda:0", "cuda:1", ...]

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The pandas dataframes containing the measured metrics for the FasterWhisper models. The first dataframe contains the measured metric per audio file, the second dataframe the average per model size.
    """

    results = []
    mean_results = []
    skip_edge_cases = False 
    
    path_to_manifest = Path(PROJECT_ROOT, "data", "manifests", eval_config["manifest"])
    with open(path_to_manifest, encoding="utf-8") as manifest_stream:
        manifest = json.loads(manifest_stream.read())
        
    if "exclude" in eval_config:
        if eval_config["exclude"]["type"] == "edge_case":
            skip_edge_cases = True
            
    for model_size in eval_config["model_sizes"]:
        print(model_size)
        
        asr_model = FasterWhisper(device, model_size)
        asr_model.warmup(print_transcription=False)
        
        for audio_metadata in tqdm(manifest):
            if skip_edge_cases and audio_metadata["type"] == "edge_case":
                continue
            
            audio_path = Path(EVAL_DATA_ROOT, audio_metadata["filename"])
            starting_timestamp = time.time()
            
            prediction, audio_length = asr_model.transcribe(audio_path, print_transcription=False)
            
            ending_timestamp = time.time()
            real_time_factor = src.metrics.real_time_factor(
                processing_time=ending_timestamp-starting_timestamp,
                audio_length=audio_length
            )
            asr_model.metric_tracker.rtfs.append(real_time_factor)
            
            word_error_rate = src.metrics.word_error_rate(
                predicted_text=prediction,
                target_text=audio_metadata["transcription"]
            )
            asr_model.metric_tracker.wers.append(word_error_rate)
            
            results.append({
                "Model Size": model_size,
                "Audio": audio_metadata["filename"],
                "Target": audio_metadata["transcription"],
                "Prediction": prediction,
                "WER": word_error_rate,
                "RTF": real_time_factor,
                "Audio Type": audio_metadata["type"],
                "Audio Notes": audio_metadata["notes"] if "notes" in audio_metadata else ""
            })
            
        mean_results.append({
            "Model Size": model_size,
            "Mean WER": asr_model.metric_tracker.get_mean_wer(),
            "Mean RTF": asr_model.metric_tracker.get_mean_rtf()
        })
    
    return pd.DataFrame(results), pd.DataFrame(mean_results)


src.runtime.set_universal_seed(SEED)
device = src.runtime.get_cuda_device()

with open(EVAL_CONFIG_PATH) as eval_config_stream:
    eval_config = yaml.safe_load(eval_config_stream)

asr_eval_results, asr_eval_mean_results = asr_evaluation(
    eval_config["asr_eval"], 
    device
)

asr_eval_results.to_parquet(Path(EVAL_RESULTS_PATH, "asr_eval.parquet"))
asr_eval_mean_results.to_parquet(Path(EVAL_RESULTS_PATH, "asr_eval_mean.parquet"))