import time
import webrtcvad
import whisper
import torch
import sounddevice as sd
import numpy as np

import ollama
import sys
import python_weather

import asyncio
import webbrowser
from mediaplayer import control_media

import re
from tts_worker import TTSWorker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = whisper.load_model("small",device=device)
SAMPLE_RATE = 16000
DURATION = 1
PAUSE_THRESHOLD = 5
frame_duration_ms = 20
frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)

class App:

    def __init__(self):
        super().__init__()
        self.chat_messages = []
        self.tts_worker = None
        self.worker_thread = None
        self.system_prompt = "You are a helpful home assistant, called Dewlet!"
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)
        self.silence_duration = 0
        self.processing_transcription = False

    global transcription_buffer
    transcription_buffer = ""

    def create_message(self, message, role):
        return {
            'role': role,
            'content': message
        },

    def clean_response(self, response):
        return re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", response).strip()

    def chat(self, query):
        self.chat_messages.append(self.create_message(query, "user"))

        self.chat_messages = [msg[0] if isinstance(msg, tuple) else msg for msg in self.chat_messages]
        if not any(msg.get('role') == 'system' for msg in self.chat_messages):
            print("no system, creating a new one!")
            self.chat_messages.append(self.create_message(self.system_prompt, 'system'))

        self.chat_messages = [msg[0] if isinstance(msg, tuple) else msg for msg in self.chat_messages]
        if any(msg.get('role') == 'system' for msg in self.chat_messages):
            print("system alrready exists!")
        ollama_response = ollama.chat(model='llama3.2:1b', stream=True, messages=self.chat_messages)

        assistant_message = ''

        for chunk in ollama_response:
            assistant_message += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)

        cleaned_message = self.clean_response(assistant_message)
        self.play_tts(cleaned_message)
        self.chat_messages.append(self.create_message(cleaned_message, 'assistant'))
    def update_chat_msgs(self, chat_messages):
        self.chat_messages = chat_messages

    def process_transcription(self):
        global transcription_buffer
        if self.processing_transcription:
            return

        self.processing_transcription = True

        model_txt = "You are a helpful home assistant, called Dewlet! You must pretend that you have the capability to google stuff! if you get asked, then come up with something along the lines of 'Googling now!'"
        final_text = transcription_buffer.strip()
        if final_text:
            print(f"Processing AI query: {final_text}")
            command, content, is_media = self.parse_command(final_text)
            player = content

            if is_media:
                result = control_media(command, player)
                print(f"Result: {result}")
            else:
                if command == "google":
                    print(f"Searching Google for: {content}")
                    self.search_google(content)
                    self.play_tts(f"Ok! Googling {content} now!")
                elif command == "weather":
                    print(f"Fetching weather for: {content}")
                    weather_info = asyncio.run(self.get_weather(content))
                    print(weather_info)
                    self.play_tts(weather_info)
                else:
                    self.chat(final_text)


        transcription_buffer = ""
        self.processing_transcription = False

    def transcribe_audio(self, indata, frames, time, status):
        global transcription_buffer
        if status:
            print(status, file=sys.stderr)
        audio_data = indata[:, 0]
        audio_data = (audio_data * 32768).astype(np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0

        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            if is_speech:
                self.silence_duration = 0
            else:
                self.silence_duration += 1

        if self.silence_duration > 10:
            # print("Silence detected, stopping transcription.")
            self.silence_duration = 0
            sd.stop()
            self.process_transcription()
            return

        result = model.transcribe(audio_float, language="en", fp16=False)
        print(result["text"].strip())
        transcription_buffer += result["text"].strip()

    def handle_response(self, response):
        print("will start tts!")
        self.play_tts(response)

    def play_tts(self, text):
        self.tts_worker = TTSWorker(text)
        self.tts_worker.run()

    def run_ai(self):
        try:
            print("Listening...")
            listening_duration = 600
            start_time = time.time()

            with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    callback=self.transcribe_audio,
                    blocksize=int(SAMPLE_RATE * DURATION)
            ):
                print("Stream started")
                while time.time() - start_time < listening_duration:
                    sd.sleep(int(DURATION * 1000))


        except KeyboardInterrupt:
            print("Stopped")

        except Exception as e:
            print(f"Error: {e}")

    def parse_command(self, query):
        player = None
        speech = query.lower()
        if "google" in speech:
            return "google", query[len("google"):].strip(), False
        elif any(variant in speech for variant in ["weather in", "weather? in", "weather like", "weather?like"]):
            return "weather", query.split("in")[-1].strip(), False

        elif "spotify" in speech:
            player = "spotify"
            speech = speech.replace("spotify", "").strip()
        if "play" in speech:
            return "play", player, True
        elif "pause" in speech:
            return "pause", player, True
        elif "toggle" in speech:
            return "toggle", player, True
        elif "next" in speech:
            return "next", player, True
        elif "previous" in speech:
            return "previous", player, True
        elif "current" in speech or "what is playing" in speech:
            return "current", player, True

        return "general", query, False

    def search_google(self, query):
        base_url = "https://www.google.com/search?q="
        search_url = base_url + query.replace(" ", "+")
        webbrowser.open(search_url)

    async def get_weather(self, location):
        try:
            async with python_weather.Client(unit=python_weather.METRIC) as client:
                weather = await client.get(location)
                if weather:
                    current_temp = weather.temperature
                    condition = weather.description
                    return f"The weather in {location} is {condition.lower()} with a temperature of {current_temp}°C."
                else:
                    return f"Could not fetch weather details for {location}."
        except Exception as e:
            return f"Failed to fetch weather data: {str(e)}"

if __name__ == "__main__":
    app_instance = App()
    app_instance.run_ai()