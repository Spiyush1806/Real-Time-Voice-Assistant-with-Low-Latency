import requests
## for a given query tell ollama to generate a response from the model

response = requests.post("http://localhost:11434/v1/chat/completions", json={
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})
print(response.json())