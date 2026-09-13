"""
Thin wrapper around Google Gemini API for all LLM + embedding calls.
Handles rate limiting, retries, and structured output parsing.
"""
import time
import json
import re
import google.generativeai as genai
from config import (
    GEMINI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    API_CALLS_PER_MINUTE, API_RETRY_ATTEMPTS, API_RETRY_DELAY
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Rate limiting state
_last_call_time = 0.0
_min_interval = 60.0 / API_CALLS_PER_MINUTE


def _rate_limit():
    """Simple rate limiter — waits if calling too fast."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_call_time = time.time()


def call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
             system_instruction: str = None) -> str:
    """
    Call Gemini LLM with retry logic.
    Returns the raw text response.
    """
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            _rate_limit()
            model = genai.GenerativeModel(
                model_name=LLM_MODEL,
                system_instruction=system_instruction,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < API_RETRY_ATTEMPTS - 1:
                wait = API_RETRY_DELAY * (2 ** attempt)
                print(f"  [LLM retry {attempt+1}/{API_RETRY_ATTEMPTS}] {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM call failed after {API_RETRY_ATTEMPTS} attempts: {e}")


def call_llm_json(prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
                  system_instruction: str = None) -> dict:
    """
    Call Gemini and parse the response as JSON.
    Handles markdown code fences that Gemini sometimes wraps around JSON.
    """
    raw = call_llm(prompt, temperature, max_tokens, system_instruction)
    # Strip markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """
    Embed a list of texts using Gemini's embedding model.
    Returns list of embedding vectors.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _rate_limit()
        try:
            result = genai.embed_content(
                model=f"models/{EMBEDDING_MODEL}",
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            print(f"  [Embed error at batch {i}] {e}")
            # Return zero vectors for failed batch
            all_embeddings.extend([[0.0] * 768] * len(batch))
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query text for retrieval."""
    _rate_limit()
    result = genai.embed_content(
        model=f"models/{EMBEDDING_MODEL}",
        content=text,
        task_type="RETRIEVAL_QUERY"
    )
    return result['embedding']


def test_api():
    """Quick smoke test for the Gemini API connection."""
    try:
        response = call_llm("Say 'API working' and nothing else.", temperature=0.0, max_tokens=20)
        print(f"  Gemini API test passed: {response.strip()}")
        return True
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False


if __name__ == "__main__":
    test_api()
