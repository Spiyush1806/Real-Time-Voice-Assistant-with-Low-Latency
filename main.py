import sounddevice as sd
import numpy as np
import collections
import time
from faster_whisper import WhisperModel
import webrtcvad
import soundfile as sf
import re
import asyncio
import nest_asyncio
import requests
import json
from TTS.api import TTS
import threading
from torch.serialization import add_safe_globals, load as torch_load
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig

# allow both config classes for unpickling
add_safe_globals([XttsConfig, XttsAudioConfig])


last_transcribe_time = time.time()
transcribe_interval = 1 
samplerate = 16000
channels = 1
frame_duration_ms = 30
frame_size = int(samplerate * frame_duration_ms / 1000)
vad = webrtcvad.Vad(2) 
vad_interval = webrtcvad.Vad(1) 
ring_buffer = collections.deque(maxlen=int(1500/ frame_duration_ms))
# bigger_ring_buffer = collections.deque(maxlen=int(1500 / frame_duration_ms))  # ~1.5 sec
model = WhisperModel("medium", device="cpu")
triggered = False
recording = []
ring_buffer_interupt = collections.deque(maxlen=int(1000 / frame_duration_ms)) 

OLLAMA_MODEL = 'gemma3:4b'
OLLAMA_URL = 'http://localhost:11434/api/generate'

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SPEAKER_WAV = r"C:\Users\spiyu\OneDrive - Indian Institute of Science\5 semester\DS246 GEN AI\Project\en_sample.wav"
TTS_LANGUAGE = "en"
TTS_PLAYBACK_SPEED = 1.0

tts = TTS(TTS_MODEL).to("cpu")

interupt_value = False 

print("Starting Live Transcription with Interval... Press Ctrl+C to stop.")

def int16_audio(audio):
    """Convert float32 audio to int16"""
    return (audio * 32767).astype(np.int16)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # dingbats
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def is_low_energy(audio, threshold=1e-4):
    """Check if audio frame is too quiet (likely not speech)."""
    energy = np.sum(np.square(audio)) / len(audio)
    return energy < threshold

def interupt(audio_queue: asyncio.Queue, stop_event: threading.Event):
    global vad_interval, interupt_value, samplerate, channels, frame_size, ring_buffer_interupt

    print("[VAD] Starting VAD interrupt thread...")
    window_size = 20
    speech_threshold = 10

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32',
                            blocksize=frame_size, latency='low') as stream:
            frame_count = 0
            while not stop_event.is_set():
                frame, overflowed = stream.read(frame_size)
                if overflowed:
                    print("[VAD] Warning: Input audio overflowed!")

                audio_float32 = frame.flatten()
                audio_int16 = int16_audio(audio_float32)

                
                if is_low_energy(audio_float32):
                    is_speech = False
                else:
                    try:
                        is_speech = vad.is_speech(audio_int16.tobytes(), samplerate)
                    except Exception as e:
                        print(f"[VAD] Error in vad.is_speech: {e}")
                        is_speech = False

                ring_buffer_interupt.append((audio_float32, is_speech))

                # Analyze only the last few frames
                recent = list(ring_buffer_interupt)[-window_size:]
                num_voiced = sum(1 for _, speech in recent if speech)

                if frame_count % 10 == 0:
                    energy = np.sum(np.square(audio_float32)) / len(audio_float32)
                    # print(f"[VAD] Frame {frame_count}, energy={energy:.6f}, voiced_frames={num_voiced}/{window_size}")

                if num_voiced >= speech_threshold:
                    interupt_value = True
                    print("[VAD] 🔇 Voice interruption detected — stopping playback.")
                    sd.stop()
                    time.sleep(2)
                    sd.stop() 
                    ring_buffer_interupt.clear()  
                   
                    drained_count = 0
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                            audio_queue.task_done()
                            drained_count += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained_count > 0:
                        print(f"[VAD] Cleared {drained_count} audio items from queue.")

                    time.sleep(0.5)
                    break

                frame_count += 1

    except Exception as e:
        print(f"[VAD] Error in VAD interrupt thread: {e}")
    finally:
        print("[VAD] Interrupt thread exiting.")

def blocking_play_audio(audio_data, samplerate, speed):
    # effective_samplerate = int(samplerate * speed)
    print("[Playback] Playing audio at samplerate:", samplerate, "with speed:", speed)
    sd.play(audio_data, samplerate=samplerate)
    # while sd.get_stream().active:
    #     time.sleep(0.05)
    sd.wait()  
    print("[Playback] Finished")
    
async def audio_worker(audio_queue):
    while True:
        try:
            item = await audio_queue.get()
            audio_data, original_samplerate, play_speed = item
            print(f"[AudioPlayer] Playing audio (samplerate: {original_samplerate}, speed: {play_speed})...")
            await asyncio.to_thread(blocking_play_audio, audio_data, original_samplerate, play_speed)

            print("[AudioPlayer] Playback finished.")
            
            audio_queue.task_done()
        except Exception as e:
            print(f"[AudioPlayer] Error: {e}")
           
            if audio_queue.empty() and item is None: 
                 audio_queue.task_done()
                 break
            
async def synthesize_and_queue_audio(text, audio_queue):
    global TTS_MODEL, TTS_SPEAKER_WAV, TTS_LANGUAGE, TTS_PLAYBACK_SPEED, tts, interupt_value
    sentence = remove_emojis(text.strip())
    if not sentence:
        print("[TTS] Empty sentence, skipping synthesis.")
        return
    print(f"[TTS] Synthesizing: '{sentence}'")
    print(interupt_value, "palced before tts")
    if interupt_value == False:
        wav = await asyncio.to_thread(
            tts.tts,
            text=sentence,
            speaker_wav=TTS_SPEAKER_WAV,
            language=TTS_LANGUAGE,
        )
        original_xtts_samplerate = 24000
        await audio_queue.put((np.array(wav), original_xtts_samplerate, TTS_PLAYBACK_SPEED))

async def stream_ollama_tts(prompt, audio_queue):
    global interupt_value, OLLAMA_URL, OLLAMA_MODEL
    sentence_buffer = ""
    llm_start_time = asyncio.get_event_loop().time()
    response = await asyncio.to_thread(
                requests.post,
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "num_predict": 50},
                stream=True,
            )
    response.raise_for_status() 

    audio_processing_active = asyncio.Event() 
    audio_processing_active.set()

    print("[Ollama] Streaming response: ", end="", flush=True)
    for line in response.iter_lines():
        if not interupt_value:
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("response", "")
                    chunk = chunk.replace("*", " ")
                    print(chunk, end="", flush=True)
                    sentence_buffer += chunk
                    while any(p in sentence_buffer for p in ".!?:"): 
                        match = re.search(r"([^.!?:]*[.!?:])", sentence_buffer)
                        if match:
                            sentence = match.group(1).strip()
                    
                            sentence_buffer = sentence_buffer[len(match.group(0)):].lstrip()
                            if sentence:
                                await synthesize_and_queue_audio(sentence, audio_queue)
                        else:
                            break 
                except json.JSONDecodeError:
                    print(f"\n[Ollama] Warning: Could not decode JSON line: {line}")
                except Exception as e:
                    print(f"\n[Ollama] Error processing chunk: {e}")
                    break # Stop processing if a chunk fails badly
        else:
            return

    if sentence_buffer.strip():
        print(f"\n[Ollama] Queuing remaining: '{sentence_buffer.strip()}'")
        await synthesize_and_queue_audio(sentence_buffer.strip(), audio_queue)


    llm_end_time = asyncio.get_event_loop().time()
    print(f"[Ollama] Response processing took {llm_end_time - llm_start_time:.2f} seconds.")
    await audio_queue.join() 
    print("[Ollama] All TTS for this response played.")
    audio_processing_active.clear()
        

text = ""
live_chunk = []
seen_segments = set()

async def record_utternace():
    global ring_buffer, recording, triggered, model, vad, samplerate, channels, frame_size, interupt_value
    audio_queue = asyncio.Queue()
    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32',
                            blocksize=frame_size, latency='low') as stream:
            while True:
                frame, _ = stream.read(frame_size)
                audio = frame.flatten()
                audio_int16 = int16_audio(audio) 
                is_speech = vad.is_speech(audio_int16.tobytes(), samplerate)

                if not triggered:
                    ring_buffer.append((audio, is_speech))
                    # bigger_ring_buffer.append((audio, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        recording.extend([f for f, _ in ring_buffer])
                        ring_buffer.clear()
                else:
                    recording.append(audio)
                    ring_buffer.append((audio, is_speech))
                    # bigger_ring_buffer.append((audio, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    # bigger_num_unvoiced = len([f for f, speech in bigger_ring_buffer if not speech])

                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        print("\nVoice ended. Transcribing...")
                        triggered = False
                        ring_buffer.clear()

                        # Concatenate recorded audio
                        recorded_audio = np.concatenate(recording)
                        recorded_audio /= np.max(np.abs(recorded_audio) + 1e-6)
                        sf.write("debug_vad.wav", recorded_audio, samplerate)

                        segments, _ = model.transcribe(recorded_audio, language="en")
                        text_list = []
                        for segment in segments:
                            print(segment.text.strip())
                            text_list.append(segment.text.strip())
                        # print(recording)
                        recording = []

                        transcription_text = " ".join(text_list)
                        print("Transcription:", transcription_text)

                        audio_task = asyncio.create_task(audio_worker(audio_queue))
                        # Generate response using Ollama
                        vad_stop_event = threading.Event()
                        print(interupt_value)
                        # Start VAD interrupt thread
                        interrupt_thread = threading.Thread(
                            target=interupt,
                            args=(audio_queue, vad_stop_event),
                            daemon=True 
                        )
                        interrupt_thread.start()
                        print("[Main] VAD interrupt thread started.")
                        print("[Main] Starting Ollama TTS stream...")
                        await stream_ollama_tts(transcription_text, audio_queue)
                        print(interupt_value, "palced after ollama stream")
                        

                        await audio_queue.join()  
                        await audio_queue.put(None) 
                        await audio_task 
                        # interrupt.start()
                        vad_stop_event.set()
                        interrupt_thread.join(timeout=2)
                        interupt_value = False
                        if interrupt_thread.is_alive():
                            print("[Main] VAD thread did not stop in time.")
                        else:
                            print("[Main] VAD interrupt thread finished.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  
    asyncio.run(record_utternace()) 
