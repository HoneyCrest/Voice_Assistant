from transformers import pipeline
import streamlit as st
import pandas as pd
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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from aiortc.contrib.media import MediaRecorder

load_dotenv()

hugging_face_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def recorder_factory():
    return MediaRecorder("record.wav")

webrtc_streamer(
    key="sendonly-audio",
    mode=WebRtcMode.SENDONLY,
    in_recorder_factory=recorder_factory,
    client_settings=ClientSettings(
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "audio": True,
            "video": False,
        },
    ),
)




try:
        r = sr.Recognizer()

        # open the file
        with sr.AudioFile("record.wav") as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print(text)

            # # Delete the record.wav file
            # os.remove("record.wav")
    


        transcript_data = text
        st.session_state.transcript_data = transcript_data
        st.spinner('Getting the response')
        # st.header('Text generated of your speech :')
        # st.success(st.session_state.transcript_data) 

        
    
        
        if 'transcript_data' in st.session_state :
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
                st.error(f"An error occurred: {output['error']}")
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

                # # Use HTML audio tags to autoplay the audio
                st.audio('Testing.wav')

                import sounddevice as sd
                import soundfile as sf

                filename = 'Testing.wav'
                # Extract data and sampling rate from file
                data, fs = sf.read(filename, dtype='float32')  
                sd.play(data, fs)
                status = sd.wait() 

                # # Use HTML audio tags to autoplay the audio
                st.audio('Testing.wav')

                # Delete the record.wav file
                os.remove("record.wav")

except sr.UnknownValueError:
    print("Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
except Exception as e:
    print(f"An error occurred: {e}")



