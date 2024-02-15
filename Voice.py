import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import pyaudio
import wave
import speech_recognition as sr

load_dotenv()


# Page title
st.set_page_config(page_title='🦜🔗 Voice Assistant Demo')
st.title('🦜🔗 Voice Assistant Demo')

# Define the parameters for the audio file
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_FILE = "sample.wav"

# Create a pyaudio object
audio = pyaudio.PyAudio()

# Initialize button state in session state
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = True

st.write(" ")

# Start and stop recording buttons in a single line
col1, col2 = st.columns([1, 1])
with col1:
    start_button = st.button('Start Recording')
with col2:
    stop_button = st.button('Stop Recording')



st.session_state.is_recording = False

if start_button:
    st.session_state.is_recording = True
    # Open a stream to record the audio
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    st.session_state.stream = stream
    st.session_state.frames = []
    st.success("Recording started")

if stop_button:
    st.session_state.is_recording = False
    if 'stream' in st.session_state:
        # Stop recording
        st.session_state.stream.stop_stream()
        st.session_state.stream.close()
        
        # Terminate the pyaudio object
        audio.terminate()

        # Save the audio frames as a wave file
        with wave.open(WAVE_FILE, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(st.session_state.frames))
        
        st.success(f"Recording stopped. Audio saved as '{WAVE_FILE}'")

# Recording logic
while st.session_state.is_recording:
    if 'stream' in st.session_state:
        data = st.session_state.stream.read(CHUNK)
        st.session_state.frames.append(data)

if stop_button:
    # audio_file= open("sample.wav", "rb")
    # transcript = client.audio.translations.create(
    # model="whisper-1", 
    # file=audio_file
    # )

    r = sr.Recognizer()

    # open the file
    with sr.AudioFile("sample.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        print(text)

    transcript_data = text
    st.session_state.transcript_data = transcript_data
    st.header('Text generated of your speech :')
    st.success(st.session_state.transcript_data) 
    
  
submit = st.button('Generate Text')

if submit :
    
    if 'transcript_data' in st.session_state :
        # ------------------------------Text Generation--------------------------------------
        import requests

        API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers = {"Authorization": f"Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        input = f"Provide the answer in 2 to 3 lines: {st.session_state.transcript_data}"
        print("INPUT : ",input)
                
        output = query({
            "inputs": input,
            
        })

        # if 'error' in output :
        #     output = query({
        #     "inputs": f"Provide the answer in 2 to 3 lines: {st.session_state.transcript_data}" })
        
        # print(output)

        # output_data = output[0]['generated_text'].split('\n', 1)[1]
        # print(output_data)

        if 'error' in output:
            # Handle the error here
            print(f"An error occurred: {output['error']}")
        else:
            output_data = output[0]['generated_text'].split('\n', 1)[1]
            print(output_data)
    
        #--------------------------------Text Generated to Speech------------------------------
        from transformers import pipeline
        import torch
        from datasets import load_dataset
        import soundfile as sf

        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        speech = synthesiser(output_data, forward_params={"speaker_embeddings": speaker_embedding})

        sf.write("Testing.wav", speech["audio"], samplerate=speech["sampling_rate"])

        st.audio("Testing.wav")
