# Real-Time Voice Agent with Low Latency

This real-time voice agent provides a truly conversational experience for any service requiring an offline voice interface. Our current implementation focuses on achieving minimal latency, ensuring a seamless back-and-forth interaction.

* **Voice Activity Detection (VAD):** We utilize a VAD module to intelligently detect when a user begins speaking. This activates the recording process.
* **FasterWhisper (Medium Model):** Once the user finishes speaking, the recorded audio is immediately transcribed into text using the **FasterWhisper (medium model)**. This provides fast and accurate speech-to-text conversion.
* **Gemma 3(4B) LLM:** The transcribed text is then fed into the **Gemma 3 Large Language Model (LLM)**. Gemma 3 processes the input and streams its response.
* **Coqui-XTTSv2 for Text-to-Speech (TTS):** As Gemma 3 streams its response, we segment the output based on punctuation marks. Each complete sentence is then passed to  **Coqui-XTTSv2 TTS model**, which generates the corresponding audio data.
* **Concurrent Audio Streaming:** we employ a concurrent audio streaming mechanism. Generated audio segments are added to an **audio queue**. An **audio worker** operates in parallel, continuously checking the queue and streaming audio to the user as it becomes available. This ensures a continuous flow of audio even while new sentences are being processed and added to the queue.
* **Seamless Loop:** Once the audio queue is empty, the system seamlessly transitions back to listening for user input, ready for the next interaction.

We've implemented a crucial **interrupt** feature. If the user begins speaking while the TTS is still generating its response, the system immediately recognizes the interruption. It stops all ongoing processes and promptly returns to listening mode, prioritizing the user's input.



### Tutorial 
- for cpu
```bash
conda create -n asr python==3.11 
conda activate asr
```
```bash
pip install torch==2.5.1 torchaudio==2.5.1 faster_whisper webrtcvad sounddevice soundfile asyncio TTS
```
```bash
python .\main.py
```
- for GPU
  ```bash
conda create -n asr python==3.11 
conda activate asr
```
```bash
pip install torch==2.5.1 torchaudio==2.5.1 / from cuda
pip install faster_whisper webrtcvad sounddevice soundfile asyncio TTS
```
```bash
python .\main.py
```