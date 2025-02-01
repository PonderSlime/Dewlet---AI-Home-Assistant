### **DISCLAIMER: I COULDN'T GET A DEMO HOSTED ANYWHERE, DUE TO THE PROJECT SIZE AND HOSTING REQUIREMENTS. ALSO, I COULDN'T GET A REQUIREMENTS.TXT TO GENERATE PROPERLY. IF YOU WANT TO TRY IT YOURSELF, FEEL FREE TO EITHER TRY MANUALLY INSTALLING DEPENDENCIES, OR OPEN AN ISSUE AND I CAN HELP YOU GET IT UP AND RUNNING!**

# Dewlet - AI Home Assistant
This is my AI home assistant project! It includes a custom pcb hat design (Very WIP), which handles audio amplification, microphone input, and buttons! 

## Hardware
All I have right now is the schematic, nothing else

## Software
The assistant itself is very comples, featuring a Speech to Text model (OpenAI Whisper), LLM (Ollama), and TTS (My own voice model, courtsey of Piper-TTS!). Together, these three models work together to get almost anything done!

Some of the features include:
- Basic conversations with persistint memory for 10 minutes! So you can have extended debates, if wanted!
- Weather Data! It can retrieve weather data from online, as long as it picks up the location properly!
- Googling! You can use it to open a new tab in browser, complete with a web search! Not really sure why though, since it takes a bit to process. Much faster to just do it like normal! 🥹
- Media control!!! From Spotify, to web browsers, to VLC, you can use it to perform every basic media-related task!!
- And the best part?! IT RUNS COMPLETELY OFFLINE! Aside from the tasks that require the internet, specifically weather, and google, It runs on your local machine!

One of the things that I don't have in here (for privacy reasons) is the TTS model I trained myself! This was by far the most tedious part of this process (also because I couldn't record time for it).
I spent around 4 hours perfecting over 275 audio files with my voice, and an additional 8 hours to train it into something that sounds like me!! Overall, I'm happy with the results!
