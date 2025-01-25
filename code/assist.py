import time

import webrtcvad
import whisper
import sounddevice as sd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import ollama
import threading
import sys
import os


import re

from piper.voice import PiperVoice
import librosa

model = whisper.load_model("base")
SAMPLE_RATE = 16000
DURATION = 1
PAUSE_THRESHOLD = 5
frame_duration_ms = 20
frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)


class TTSWorker(QThread):

    def __init__(self, input, parent=None):
        super().__init__(parent)
        self.text = input
        self.audio_played_event = threading.Event()

    def get_resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def run(self):
        temp_file_path = None
        try:
            model = self.get_resource_path("src/lessac/en_US-lessac-medium.onnx")
            #print(f"Looking for ONNX model at: {model}")

            voice = PiperVoice.load(model)

            audio_segment = AudioSegment.empty()

            for audio_bytes in voice.synthesize_stream_raw(self.text):
                int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                segment = AudioSegment(
                    int_data.tobytes(),
                    frame_rate=voice.config.sample_rate,
                    sample_width=2,
                    channels=1
                )
                audio_segment += segment

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file_path = temp_file.name
            print(f"Temporary MP3 file path: {temp_file_path}")
            if not os.path.exists(temp_file_path):
                print("MP3 file not found!")
            audio_segment.export(temp_file_path, format="mp3")
            temp_file.close()

            duration = librosa.get_duration(path=temp_file_path)
            print(f"Duration: {duration}")

            self.play_audio_blocking(temp_file_path)

        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def play_audio_blocking(self, audio_file_path):
        audio = AudioSegment.from_file(audio_file_path)

        playback_thread = threading.Thread(target=self.play_audio, args=(audio,))
        playback_thread.start()

        self.audio_played_event.wait()
        print("Audio playback finished. Resuming speech detection.")

        App.resume_speech_detection

    def play_audio(self, audio):
        print("Playing audio...")
        play(audio)
        self.audio_played_event.set()

class App(QObject):

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

    def resume_speech_detection(self):
        print("resuming now!")
        self.processing_transcription = False
        self.run_ai()


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
            print("systtem alrready exists!")
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

        model_txt = "You are a helpful home assistant, called Dewlet!"
        final_text = transcription_buffer.strip()
        if final_text:
            print(f"Processing AI query: {final_text}")
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
            #print("Silence detected, stopping transcription.")
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
        self.tts_worker = TTSWorker(text, self)
        self.tts_worker.start()

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

    def cleanup(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        if self.tts_worker:
            self.tts_worker.quit()
            self.tts_worker.wait()

if __name__ == "__main__":
    app_instance = App()
    app_instance.run_ai()
    app_instance.cleanup()