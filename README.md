# Voice_Assistant

The Voice Assistant Demo allows the users to speak something/query and obtain its Answer in the form of audio.

Process :
    The speech given as an input is saved as an Audio file which is then converted into the text using the Speech Recognition Library and that text is fed to the 'Falcon-7B-Instruct' LLM Model which generates the Answer Text for the input provided.

The Text generated from the  'Falcon-7B-Instruct' LLM Model of Hugging face as an answer is then fed to the 'microsoft/speecht5_tts' LLM Model which is used for converting the Text to Speech and gives the Output as the Audio file.
