"""Defines functions used for controlling the runtime of the Demonstrator program."""

import random

import torch
import transformers

def set_universal_seed(seed: int) -> None:
    """Sets a given seed across different deep learning libraries to improve consistency of results.

    Args:
        seed (int): The seed passed to the various libraries.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        
def set_universal_max_threads(thread_count: int) -> None:
    """Sets the number of threads that can be used by certain libraries.

    Args:
        thread_count (int): The number of threads that can be used.
    """

    torch.set_num_threads(thread_count)
        
def get_cuda_device() -> str:
    """Returns the CUDA device to run the Demonstrator's models on.

    Returns the CUDA device to run the Demonstrator's models on.
    If no CUDA device is available, returns "cpu".
    If one CUDA device is available, returns "cuda".
    If more than one CUDA device is available, returns the device that has the most free space left.

    Returns:
        str: The CUDA device the Demonstrator's models will be loaded on.
    """

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = _get_cuda_device_with_most_free_memory()
        else:
            device = "cuda"
    else:
        device = "cpu"
    
    print(torch.device(device))

    if "cuda" in device:
        print(torch.cuda.get_device_name(), "\n")
        
    return device

def _get_cuda_device_with_most_free_memory() -> str:
    """Returns the CUDA device with the most free memory.

    Returns:
        str: The CUDA device with the most free memory.
    """

    device = ""
    device_memory = 0
            
    for i in range(torch.cuda.device_count()):
        current_device = f"cuda:{i}"
        current_device_memory = torch.cuda.mem_get_info(current_device)[0]
                
        if current_device_memory >= device_memory:
            device = current_device
            device_memory = current_device_memory
            
    return device
