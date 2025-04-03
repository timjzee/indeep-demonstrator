1. Download codebase from GitHub
2. Install Python 3.10.xx (if not installed)
3. Create (`py -3.10 -m venv env`) and activate (`source env/bin/activate` for Linux, `source env/Scripts/activate` for Windows) a Python venv in the root of the project called "env"
4. Install libraries from requirements.txt to venv (`pip install -r requirements.txt`)
5. Create ".env" file in project root with following contents: 
```py 
PYTHONPATH=.
DEMONSTRATOR_MODE="client"
DEMONSTRATOR_PROFILE="default"
```
6. Test connection to server:
   1. Dig SSH tunnel to Ponyland on IP and port defined in configs (`ssh -L 8031:localhost:8031 -N username@[thunderlane|lightning].science.ru.nl`)
   2. Connect to Ponyland to monitor server behavior
   3. Start the server
   4. Start the client
      1. In a terminal, go to the root of the Demonstrator codebase
      2. Activate the local Python env (`source env/bin/activate` for Linux, `source env/Scripts/activate` for Windows)
      3. Run main.py (`python src/main.py`)
   5. Do a round of parroting