import os
import sys
import threading

from pydub import AudioSegment
from pydub.playback import play
import tempfile

from piper.voice import PiperVoice
import librosa
import numpy as np

class TTSWorker:

    def __init__(self, input):
        super().__init__()
        self.text = input
        self.audio_played_event = threading.Event()

    def get_resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def run(self):
        temp_file_path = None
        try:
            model = self.get_resource_path("src/micah/large/en_US-micah-large.onnx")
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

    def play_audio(self, audio):
        print("Playing audio...")
        play(audio)
        self.audio_played_event.set()
