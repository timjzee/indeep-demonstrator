# InDeep Demonstrator 🦜

## To Do
- Base SER implementation:
  - create alternative mode in which the client idles (is not listening) until a button is pressed
  - add explanation: "I think you said:" + recognized text
  - add SER functionality
    - use either [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
    - or [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
  - print out result server side
  - add SER probability to explanation: "And I'm PERCENTAGE sure you sounded MOST_PROBABLE_EMOTION."
- Bonus TTS + emotion control
  - add [Parler-TTS Mini: Expresso](https://huggingface.co/parler-tts/parler-tts-mini-expresso) as a TTS model
  - control emotion based on SER
  - Different response: "I'm PERCENTAGE sure you did not sound LEAST_PROBABLE_EMOTION. Because then you would have said: + emotional TTS of recognized text"


[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive)

This repository contains all code and documentation relevant to the Demonstrator software that can be made public, and is the end product of a Research Internship performed by [Daan Brugmans](https://github.com/daanbrugmans) at the [Radboud University](https://www.ru.nl/en)'s [Centre of Language and Speech Technology](https://www.ru.nl/en/cls/clst) department.

## Description
The Demonstrator is a Proof of Concept software application.
It is created as part of the [InDeep Project](https://projects.illc.uva.nl/indeep/), a collaboration between researchers of varying Dutch universities that work in the domain of interpretability and explainability of deep learning models that consume and produce text, speech and/or music.
As part of the InDeep Project, researchers organize "outreach activities" where focus groups outside the data science field, both academic as wel as general audiences, can be educated on deep learning models that consume and/or produce text, speech and/or music.
This Demonstrator is an application that can be used for such outreach activities as a way for target audiences to interact with a speech-only neural system.

This Demonstrator implementation is a Proof of Concept (PoC) and is capable of "parroting" an end user's Dutch speech.
It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcribing audio and either [MMS-TTS](https://huggingface.co/facebook/mms-tts) or [Piper TTS](https://rhasspy.github.io/piper-samples/) for synthesizing speech.
It is primarily used in a Client-Server setup using the Radboud CLST team's [Ponyland Server](https://ponyland.science.ru.nl/), but can also be run entirely locally.

## Installing & Running
Instructions for installing the Demonstrator as a client or standalone app can be found [here](/docs/installation_client.md).

It is recommended that you use an NVIDIA GPU that is powerful enough to infer a faster-whisper model.
Both Windows and Linux are supported, although Windows does not support the Piper TTS.

The [`/configs/demonstrator_configs.yaml`](/configs/demonstrator_configs.yaml) folder contains settings for the Demonstrator that can be changed.
Please refer to the `.env` file for setting up your systems mode (client/server/app) and used model profile.

## Architecture
The architecture and flow of the Demonstrator is explained concisely in the following diagram:

![Diagram of the Demonstrator's Client-Server architecture](/docs/diagrams/deployment.png)

### State Machine Diagrams
This implementation of the Demonstrator works using state logic.
The following diagrams show the state machines of the Demonstrator Client, Server, and App respectively.

#### Client
![State Machine Diagram of a Demonstrator Client](/docs/diagrams/state_client.png)

#### Server
![State Machine Diagram of a Demonstrator Server](/docs/diagrams/state_server.png)

#### App
![State Machine Diagram of a Demonstrator App](/docs/diagrams/state_app.png)
