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
import http.client
import threading
from dotenv import load_dotenv
import os
import trafilatura
from urllib.parse import urlparse
from typing import Dict, Any
import logging
import aiohttp
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

SERPER_API = "<your_api_key_here>"

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
model = WhisperModel("medium", device="cuda", compute_type="float16")
triggered = False
recording = []
ring_buffer_interupt = collections.deque(maxlen=int(1000 / frame_duration_ms)) 

OLLAMA_MODEL = 'gemma3'
OLLAMA_URL = 'http://localhost:11434/api/generate'

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SPEAKER_WAV = r"C:\Users\yashk\OneDrive - Indian Institute of Science\final\en_sample.wav" 
TTS_LANGUAGE = "en"
TTS_PLAYBACK_SPEED = 1.0
HOST = "localhost"
conn = http.client.HTTPSConnection("google.serper.dev")
headers = {
'X-API-KEY': SERPER_API,
'Content-Type': 'application/json'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def get_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=30)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    return _aiohttp_session


async def query_gemma_async(query: str) -> str:
    """Async Gemma call using shared aiohttp session."""
    try:
        session = await get_session()
        url = f"http://{HOST}:11434/api/generate"
        payload = {"model": OLLAMA_MODEL, "prompt": query, "stream": False}
        # simple retry loop
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, timeout=30) as resp:
                    resp.raise_for_status()
                    body = await resp.json()
                    return body.get("response", "")
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(1 + attempt)
    except Exception as e:
        logger.error("Gemma query failed: %s", e)
        return f"Error generating response: {e}"
_executor = ThreadPoolExecutor(max_workers=6)
_aiohttp_session: aiohttp.ClientSession | None = None
tts = TTS(TTS_MODEL).to("cuda")

interupt_value = False 

print("Starting Live Transcription with Interval... Press Ctrl+C to stop.")

def classify_link(url):
    domain = urlparse(url).netloc.lower()

    # Category dictionaries
    VIDEO_SITES = ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]
    SOCIAL_SITES = ["reddit.com", "facebook.com", "twitter.com", "x.com", "quora.com", "instagram.com", "linkedin.com"]
    NEWS_SITES = ["bbc.com", "cnn.com", "nytimes.com", "theguardian.com"]
    FORUM_SITES = ["stackexchange.com", "stackoverflow.com", "github.com", "medium.com"]

    if any(s in domain for s in VIDEO_SITES):
        return "video"
    elif any(s in domain for s in SOCIAL_SITES):
        return "social"
    elif any(s in domain for s in NEWS_SITES):
        return "news"
    elif any(s in domain for s in FORUM_SITES):
        return "forum"
    else:
        return "web"

async def get_webpage_text_async(url: str) -> str | None:
    """Run trafilatura in thread to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    try:
        def fetch():
            downloaded = trafilatura.fetch_url(url)
            return trafilatura.extract(downloaded) if downloaded else None
        return await loop.run_in_executor(_executor, fetch)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

async def website_search_sse(query: str) -> Dict[str, Any]:
    """Fixed async web search using Serper API"""
    try:
        print(f"[Search] 🔍 Starting search for: '{query}'")
        
        # Create a fresh connection each time
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        payload = json.dumps({"q": query})
        
        headers_local = {
            'X-API-KEY': SERPER_API,
            'Content-Type': 'application/json'
        }
        
        # Make the request
        conn.request("POST", "/search", payload, headers_local)
        res = conn.getresponse()
        
        print(f"[Search] Response status: {res.status}")
        
        if res.status != 200:
            error_body = res.read().decode('utf-8')
            print(f"[Search] ❌ API Error ({res.status}): {error_body}")
            return {"error": f"API returned {res.status}", "links": [], "snippets": [], "peopleAlsoAsk": []}
        
        data = res.read()
        result = json.loads(data.decode('utf-8'))
        
        # Close connection
        conn.close()
        
        # Extract results
        organic = result.get('organic', [])
        links = [item.get('link', '') for item in organic]
        snippets = [item.get('snippet', '') for item in organic]
        titles = [item.get('title', '') for item in organic]
        peopleAlsoAsk = [
            {
                "question": item.get('question', ''),
                "snippet": item.get('snippet', ''),
                "link": item.get('link', '')
            } 
            for item in result.get('peopleAlsoAsk', [])
        ]
        
        print(f"[Search] ✅ Found {len(links)} results")
        print(f"[Search] First 3 links: {links[:3]}")
        
        return {
            "links": links,
            "snippets": snippets,
            "titles": titles,
            "peopleAlsoAsk": peopleAlsoAsk
        }
        
    except HttpError as e:
        logger.error(f"[Search] ❌ Serper API error: {e}")
        return {"error": f"Serper API error: {e}", "links": [], "snippets": [], "peopleAlsoAsk": []}
    except json.JSONDecodeError as e:
        logger.error(f"[Search] ❌ JSON decode error: {e}")
        return {"error": f"Invalid JSON response: {e}", "links": [], "snippets": [], "peopleAlsoAsk": []}
    except Exception as e:
        logger.error(f"[Search] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {e}", "links": [], "snippets": [], "peopleAlsoAsk": []}



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

                        # refine the prompt such that we can do google search and get better results
                        
                        search_prompt = f"""You are a search query optimizer. Your task is to transform conversational user queries into precise, effective search queries that will return the most relevant results.

                        USER'S ORIGINAL QUERY:
                        "{transcription_text}"

                        INSTRUCTIONS:
                        1. Extract the core intent and key concepts from the user's query
                        2. Remove filler words, typos, and conversational elements
                        3. Use clear, searchable keywords and phrases
                        4. Focus on actionable terms (e.g., "how to", "troubleshoot", "fix", "guide")
                        5. Keep the refined query between 5-12 words
                        6. Return ONLY the refined query - no explanation, no quotes, no extra text

                        EXAMPLES:

                        Input: "i ahve my vehicle's engine light on what to do and how to fix it"
                        Output: How to diagnose and fix vehicle engine light

                        Input: "my internet is not working properly and i keep getting disconencted"
                        Output: Troubleshooting frequent internet disconnection issues

                        Input: "what's the best way to learn python programming for beginners like me"
                        Output: Best Python programming tutorials for beginners

                        Input: "my laptop won't turn on after i spilled water on it yesterday"
                        Output: Fix water damaged laptop won't power on

                        Input: "can you tell me about climate change effects on polar bears"
                        Output: Climate change impact on polar bear populations

                        Input: "I need to know how much protein I should eat if I'm trying to build muscle"
                        Output: Daily protein intake for muscle building

                        Now generate the refined search query for the user's query above:"""

                        refined_query = await query_gemma_async(search_prompt)
                        print(f"[Main] Refined search query: {refined_query}")

                        website_content = await website_search_sse(refined_query)
                        print(f"[Main] Website content fetched: {website_content}")
                        links = website_content.get("links", [])
                        snippets = website_content.get("snippets", [])
                        peopleAlsoAsk = website_content.get("peopleAlsoAsk", [])

                        context = ""

                        if links is not None:
                            for link in links:
                                logging.info(f"Fetching content from link: {link}")
                                cat = classify_link(link)
                                logging.info(f"Classified link category: {cat}")
                                if cat == "web":
                                    text = await get_webpage_text_async(link)
                                    if text:
                                        context += f"Content from webpage {link}:\n{text}\n\n"
                                else:
                                    continue
                        
                        print(context)

                        format_prompt = f"""
                        You are an expert assistant designed to help users w    ith their queries by providing detailed, accurate, and helpful responses. Your goal is to assist the user in understanding and resolving their issue effectively.

                        The user asked:
                        "{transcription_text}"

                        Below is some context information from reliable websites (already summarized for you):
                        ---
                        {context}
                        ---
                        Use this context as background knowledge when crafting your response. If you refer to any facts or steps from the websites, clearly mention the corresponding source links at the end of your explanation. Also check if the website is promotional or not some websites doesn't provide any info they are just selling thier services so avoid such websites. ignore that info while answering.
                        Your response should be clear, concise, and directly address the user's question. If the user asks for specific steps or solutions, provide them in a logical order. If you need to suggest any tools or parts, include clickable Amazon links for easy access.
                        Your task:
                        1. Provide a clear, conversational, and step-by-step explanation that directly answers the user's question.
                        2. Integrate key information from the website context naturally into your response.
                        3. Include the website links you used as references under a section titled **"Sources"**.
                        4. At the end, include a section titled **"Recommended YouTube Tutorials"** listing the most relevant videos, with a short reason for each (why it’s useful).
                        6. give response in markdown format for better readability and don't loose the details. let the response go beyond 100 words if needed.
                        Output Format:
                        -----------------
                        **Response:**
                        (Your helpful, natural-language explanation here.)


                        **Sources:**
                        // In the website sources if you have youtube links please exclude them.
                        - [Website Name](https://example.com)
                        - [Website Name](https://example2.com)
                        -----------------
                        """
                        

                        await stream_ollama_tts(format_prompt, audio_queue)
                        print(interupt_value, "placed after ollama stream")

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

