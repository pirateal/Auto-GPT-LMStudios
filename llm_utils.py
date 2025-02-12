import requests
import time
from colorama import Fore
from config import Config

cfg = Config()

# Define the base URL for your localhost server
base_url = "http://192.168.1.101:1234/v1"

def create_chat_completion(messages, model=None, temperature=cfg.temperature, max_tokens=None) -> str:
    """Create a chat completion using your localhost server."""
    # Filter out any messages that have empty or whitespace-only content
    valid_messages = [m for m in messages if m.get("content", "").strip()]

    response = None
    num_retries = 5
    for attempt in range(num_retries):
        try:
            url = f"{base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": valid_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                break
            elif response.status_code == 429:
                if cfg.debug_mode:
                    print(Fore.RED + "Error: API Rate Limit Reached. Waiting 20 seconds..." + Fore.RESET)
                time.sleep(20)
            elif response.status_code == 502:
                if cfg.debug_mode:
                    print(Fore.RED + "Error: API Bad gateway. Waiting 20 seconds..." + Fore.RESET)
                time.sleep(20)
            else:
                raise RuntimeError(f"Received unexpected status code: {response.status_code}")
        except Exception as e:
            if attempt == num_retries - 1:
                raise
            else:
                print(Fore.RED + "Error:", str(e) + Fore.RESET)
                time.sleep(20)

    if response is None:
        raise RuntimeError("Failed to get response after 5 retries")

    content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
    return content

# Example usage:
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": "Introduce yourself."}
    ]
    completion = create_chat_completion(messages, model="local-model")
    print(completion)
