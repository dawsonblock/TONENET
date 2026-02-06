"""
JSONL ledger for deterministic replay and debugging.

Records all events in the duplex pipeline:
- Utterance segments (audio hashes, VAD stats)
- STT results (text, latency, model params)
- Agent responses
- TTS chunks and timing
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Any


def sha256_bytes(b: bytes) -> str:
    """SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


class JsonlLedger:
    """
    Append-only JSONL ledger for event logging.
    
    Enables deterministic replay by recording all pipeline events
    with content hashes and timestamps.
    
    Example:
        ledger = JsonlLedger("duplex.jsonl")
        ledger.append({"type": "stt", "text": "Hello", "latency_sec": 0.5})
    """
    
    def __init__(self, path: str | Path):
        """
        Args:
            path: Path to JSONL file (created if doesn't exist)
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, record: dict[str, Any]):
        """
        Append record to ledger.
        
        Args:
            record: Dict to log (ts added automatically if missing)
        """
        record = dict(record)
        record.setdefault("ts", time.time())
        
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def read_all(self) -> list[dict[str, Any]]:
        """Read all records from ledger."""
        if not self.path.exists():
            return []
        
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    
    def filter_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Get all records of a specific type."""
        return [r for r in self.read_all() if r.get("type") == event_type]
    
    def get_conversation(self) -> list[dict[str, str]]:
        """
        Extract conversation history from ledger.
        
        Returns:
            List of {"role": "user"|"assistant", "content": text}
        """
        conversation = []
        for r in self.read_all():
            if r.get("type") == "stt" and r.get("text"):
                conversation.append({
                    "role": "user",
                    "content": r["text"]
                })
            elif r.get("type") == "agent" and r.get("agent_text"):
                conversation.append({
                    "role": "assistant",
                    "content": r["agent_text"]
                })
        return conversation
    
    def clear(self):
        """Clear the ledger (dangerous!)."""
        if self.path.exists():
            self.path.unlink()


def replay_print(ledger_path: str):
    """
    Print conversation from ledger file.
    
    Args:
        ledger_path: Path to JSONL ledger
    """
    ledger = JsonlLedger(ledger_path)
    
    for r in ledger.read_all():
        t = r.get("type")
        
        if t == "speech_start":
            print("\n[LISTENING...]")
        elif t == "stt":
            print(f"\n[USER] {r.get('text', '')}")
        elif t == "agent":
            print(f"[ASSISTANT] {r.get('agent_text', '')}")
        elif t == "tts_abort":
            print(f"[INTERRUPTED] {r.get('reason', '')}")
        elif t == "drop":
            print(f"[DROP] {r.get('reason', '')}")
