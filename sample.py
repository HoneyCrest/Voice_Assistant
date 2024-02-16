
from transformers import pipeline
import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import pyaudio
import wave
import speech_recognition as sr
# from transformers import pipeline
import torch
from datasets import load_dataset
import soundfile as sf
from transformers import pipeline
import pyaudio
from st_audiorec import st_audiorec
import streamlit as st
import wavio as wv



# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')

# for i in range(0, numdevices):
#     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

load_dotenv()

hugging_face_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Page title
st.set_page_config(page_title='🦜🔗 Voice Assistant Demo')
st.title('🦜🔗 Voice Assistant Demo')

print(sd.query_devices())

import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 10  # Duration of recording (changed to 10 seconds)

record = st.button("Start Record")
if record :
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('voice_output.wav', fs, myrecording)  # Save as WAV file
    # Convert the NumPy array to audio file
    wv.write("recording1.wav", myrecording, fs, sampwidth=2)

    
    try:
        r = sr.Recognizer()

        # open the file
        with sr.AudioFile("recording1.wav") as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print(text)
    


            transcript_data = text
            st.session_state.transcript_data = transcript_data
            

                
            
            # submit = st.button('Generate text')

        # if submit :
            
            # if 'transcript_data' in st.session_state :
                # ------------------------------Text Generation--------------------------------------
            import requests

            API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
            headers = {"Authorization": f"Bearer {hugging_face_token}"}

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
            
            input = f"Provide the answer in 2 to 3 lines: {transcript_data}"
            print("INPUT : ",input)
                    
            output = query({
                "inputs": input,
            })

            if 'error' in output:
                # Handle the error here
                print(f"An error occurred: {output['error']}")
                output = query({
                "inputs": input,
                
            })
                output_data = output[0]['generated_text'].split('\n', 1)[1]
                print(output_data)
            else:
                output_data = output[0]['generated_text'].split('\n', 1)[1]
                print(output_data)
        
            #--------------------------------Text Generated to Speech------------------------------
            

            synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

            speech = synthesiser(output_data, forward_params={"speaker_embeddings": speaker_embedding})

            sf.write("Testing.wav", speech["audio"], samplerate=speech["sampling_rate"])

            st.audio("Testing.wav")

    except sr.UnknownValueError:
        
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
