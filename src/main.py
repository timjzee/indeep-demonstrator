"""Runs the Demonstrator program.

Raises:
    ValueError: When one of the constants in the `.env` file does not match one of the accepted values.
"""

import os
import threading
from pathlib import Path

import yaml
import dotenv
import transformers
import uvicorn

import runtime
import rest_api
from demonstrator import DemonstratorFactory, DemonstratorServer, DemonstratorClient
import ssl

PROJECT_ROOT = Path(__file__).parents[1]
DOT_ENV_PATH = Path(PROJECT_ROOT, ".env")
CONFIG_PATH = Path(PROJECT_ROOT, "configs", "demonstrator_configs.yaml")
HF_CACHE_PATH = Path(PROJECT_ROOT, ".cache", "huggingface")

if __name__ == "__main__":   
    transformers.logging.set_verbosity_error()
    os.environ["HF_HOME"] = str(HF_CACHE_PATH)
    dotenv.load_dotenv()
        
    mode = os.getenv("DEMONSTRATOR_MODE").strip()
    profile = os.getenv("DEMONSTRATOR_PROFILE").strip()

    if os.name == 'nt':
        # Disable SSL verification on Windows to avoid certificate issues
        ssl._create_default_https_context = ssl._create_unverified_context
    
    print(f"Starting Demonstrator in {mode} mode using {profile} profile.")
    
    if mode is None:
        raise ValueError("Could not find an .env file with the 'DEMONSTRATOR_MODE' variable. Please create a .env file in the project root, then add the variable 'DEMONSTRATOR_MODE' in it, and assign it a value of 'app' (for a full standalone Demonstrator), 'server' (for a Demonstrator server), or 'client' (for a Demonstrator client).")
    
    if profile is None:
        raise ValueError("Could not find an .env file with the 'DEMONSTRATOR_PROFILE' variable. Please create a .env file in the project root, then add the variable 'DEMONSTRATOR_PROFILE' in it. Its value should be one of the profiles that exist in the Demonstrator Config YAML for the selected 'DEMONSTRATOR_MODE', likely 'default'.")
    
    with open(CONFIG_PATH) as demonstrator_config_stream:
        config = yaml.safe_load(demonstrator_config_stream)
        
        host_ip = config["host_ip"]
        host_port = config["host_port"]        
        runtime.set_universal_seed(config["seed"])
        runtime.set_universal_max_threads(config["threads"])
        
        model_config = config[mode][profile]
        device = runtime.get_cuda_device()
        
        factory = DemonstratorFactory(mode, model_config, device)
        demonstrator_instance = factory.create_demonstrator()
        
        if isinstance(demonstrator_instance, DemonstratorServer):           
            rest_api.fast_api.demonstrator = demonstrator_instance
            
            # Run the RESTful API on a separate thread
            api_thread = threading.Thread(
                target=uvicorn.run,
                daemon=True,
                kwargs={
                    "app": rest_api.fast_api,
                    "host": host_ip,
                    "port": host_port
                }
            )
            api_thread.start()
            
        if isinstance(demonstrator_instance, DemonstratorClient):
            demonstrator_instance.api_url = f"http://{host_ip}:{host_port}"
        
        demonstrator_instance.run()

        # TODO
        # - Write a dockerfile
    