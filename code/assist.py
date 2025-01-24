import whisper
import sounddevice as sd
import numpy as np
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import ollama

import sys
import os


import re

from piper.voice import PiperVoice

model = whisper.load_model("base")
SAMPLE_RATE = 16000
DURATION = 1
PAUSE_THRESHOLD = 5

class WorkerThread():
    def create_message(self, message, role):
        return {
            'role': role,
            'content': message
        },

    def clean_response(self, response):
        cleaned_response = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", response)
        return cleaned_response.strip()

    def chat(self):
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
        self.result_ready.emit(cleaned_message)
        self.chat_messages.append(self.create_message(cleaned_message, 'assistant'))
        self.chat_msg_history.emit(self.chat_messages)

    def run(self):
        self.chat_messages.append(
            self.create_message(self.query, 'user')
        )
        print(f'\n\n--{self.query}--\n\n')
        self.chat()

class TTSWorker():

    def __init__(self, input, parent=None):
        super().__init__(parent)
        self.text = input

    def get_resource_path(self, relative_path):
        """Get the absolute path to a resource, works for dev and PyInstaller."""
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def run(self):
        temp_file_path = None
        try:
            model = self.get_resource_path("src/lessac/en_US-lessac-medium.onnx")
            print(f"Looking for ONNX model at: {model}")

            voice = PiperVoice.load(model)

            audio_segment = AudioSegment.empty()

            for audio_bytes in voice.synthesize_stream_raw(self.text):
                int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                segment = AudioSegment(
                    int_data.tobytes(),
                    frame_rate=voice.config.sample_rate,
                    sample_width=2,  # int16 = 2 bytes
                    channels=1
                )
                audio_segment += segment

            # Save the audio as a temporary MP3 file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file_path = temp_file.name
            audio_segment.export(temp_file_path, format="mp3")
            temp_file.close()

            # Play the MP3 file
            play(AudioSegment.from_file(temp_file_path))

        except Exception as e:
            #self.tts_error.emit(f"TTS Error: {str(e)}")
            print(f"TTS Error: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

def process_transcription(self):
    global transcription_buffer
    final_text = " ".join(transcription_buffer).strip()
    transcription_buffer = []
    if final_text:
        print(f"Processing AI query: {final_text}")
        self.worker = WorkerThread(model_txt="llama3.2:1b", chat_messages=[], query=final_text)
        self.worker.result_ready.connect(
            lambda response: self.handle_response(response),
        )
        self.worker_thread.error_occurred.connect(
            lambda error: self.handle_response(f"Error: {error}", False)
        )
        self.worker_thread.start()


def callback(indata, frames, time, status):
    global last_speech_time, transcription_buffer
    if status:
        print(f"Status: {status}")

    audio_data = indata[:, 0]
    audio_data = (audio_data * 32768).astype(np.int16)
    audio_float = audio_data.astype(np.float32) / 32768.0

    result = model.transcribe(audio_float, language="en", fp16=False)
    text = result["text"].strip()

    if text:
        print(f"Detected Speech: {text}")
        transcription_buffer.append(text)
        last_speech_time = time.time()

    else:
        if time.time() - last_speech_time > PAUSE_THRESHOLD and transcription_buffer:
            process_transcription()

def handle_response(self, response):
    self.play_tts(response)


def play_tts
try:
    print("Listening...")
    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=callback,
            blocksize=int(SAMPLE_RATE * DURATION)
    ):
        sd.sleep(int(DURATION * 1000) * 60)

except KeyboardInterrupt:
    print("Stopped")

except Exception as e:
    print(f"Error: {e}")