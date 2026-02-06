"""
Text sentence chunker for streaming TTS.

Splits text into sentence chunks as they arrive, enabling TTS to start
on the first sentence while the rest is still being generated.
"""

import re
from typing import Iterable, Iterator


# Sentence boundary pattern
_SENT_SPLIT = re.compile(r"([.!?]+[\)\]\"']*\s+)")


def sentence_chunks(
    stream: Iterable[str],
    min_chars: int = 24,
    max_chars: int = 220
) -> Iterator[str]:
    """
    Convert an incremental text stream into sentence chunks.
    
    Yields sentence-ish chunks as soon as punctuation is detected,
    allowing TTS to start immediately on the first sentence.
    
    Args:
        stream: Iterable of incremental text pieces (tokens or partial strings)
        min_chars: Minimum characters before yielding a chunk
        max_chars: Maximum buffer size before forcing a yield
    
    Yields:
        Sentence chunks suitable for TTS
    
    Example:
        # With LLM streaming tokens
        for chunk in sentence_chunks(llm.stream(prompt)):
            audio = tts.synthesize(chunk)
            play(audio)
    """
    buf = ""
    
    for part in stream:
        if not part:
            continue
        buf += part
        
        # Emit at sentence boundary
        while True:
            m = _SENT_SPLIT.search(buf)
            if not m:
                break
            cut = m.end()
            chunk = buf[:cut].strip()
            buf = buf[cut:]
            if len(chunk) >= min_chars:
                yield chunk
        
        # Force yield if buffer grows too large (no punctuation)
        if len(buf) > max_chars:
            chunk = buf[:max_chars].strip()
            buf = buf[max_chars:]
            if chunk:
                yield chunk
    
    # Yield remaining buffer
    tail = buf.strip()
    if tail:
        yield tail


def split_text_to_sentences(text: str, min_chars: int = 24) -> list[str]:
    """
    Split complete text into sentence chunks.
    
    Convenience wrapper for non-streaming use.
    
    Args:
        text: Complete text to split
        min_chars: Minimum characters per chunk
    
    Returns:
        List of sentence chunks
    """
    return list(sentence_chunks([text], min_chars=min_chars))
