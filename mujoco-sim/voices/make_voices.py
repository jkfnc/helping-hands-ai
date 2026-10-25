"""Margaret's lines for the demo video, voiced by Gemini TTS as an older woman (16 kHz mono WAV)."""
import os, subprocess, sys, wave
HERE = os.path.dirname(os.path.abspath(__file__))
for l in open(os.path.join(HERE, "..", "..", ".env")):
    if "=" in l: k, v = l.strip().split("=", 1); os.environ.setdefault(k, v)
from google import genai
from google.genai import types

MODEL, VOICE = "gemini-2.5-flash-preview-tts", "Gacrux"     # Gacrux: Google's "mature" voice
DIRECTION = "Say slowly, softly and warmly, like a frail woman in her eighties"
LINES = {
    "1_apple": "Hello, Robo. My hands aren't what they used to be. Could you put the apple in the bowl for me?",
    "2_orange": "Oh, lovely. Now could you put everything back, and then put the orange in the bowl?",
    "3_banana": "And the banana on the plate, please.",
    "stop_4": "Wait, stop! Stop, please.",
    "5_water": "That's alright, dear. I'm a bit thirsty. Could you bring me some water?",
}
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for name, text in LINES.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    r = client.models.generate_content(model=MODEL, contents=f"{DIRECTION}: {text}",
        config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)))))
    raw = os.path.join(HERE, f"{name}.pcm")
    open(raw, "wb").write(r.candidates[0].content.parts[0].inline_data.data)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", raw,
                    "-ar", "16000", "-ac", "1", os.path.join(HERE, f"{name}.wav")], check=True)
    os.remove(raw)
    with wave.open(os.path.join(HERE, f"{name}.wav")) as w:
        print(f"{name}: {w.getnframes() / w.getframerate():.1f}s")
