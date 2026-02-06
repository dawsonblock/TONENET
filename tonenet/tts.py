"""
Text-to-Speech (TTS) integration using edge-tts.

Provides async and sync speech synthesis for the duplex agent.

Install: pip install edge-tts
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import torch


@dataclass
class TTSVoice:
    """TTS voice configuration."""
    name: str
    language: str
    gender: str


# Common voices
VOICES = {
    "guy": "en-US-GuyNeural",
    "jenny": "en-US-JennyNeural", 
    "aria": "en-US-AriaNeural",
    "davis": "en-US-DavisNeural",
    "tony": "en-US-TonyNeural",
    "sara": "en-US-SaraNeural",
}


class StreamingTTS:
    """
    Text-to-speech using edge-tts (Microsoft Edge TTS).
    
    Free, no API key required.
    
    Example:
        tts = StreamingTTS()
        audio = tts.speak("Hello world")
    """
    
    def __init__(
        self,
        voice: str = "en-US-GuyNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 24000
    ):
        """
        Args:
            voice: Voice name (e.g., "en-US-GuyNeural")
            rate: Speaking rate (e.g., "+10%", "-20%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")
            sample_rate: Target sample rate for output
        """
        # Handle shorthand names
        self.voice = VOICES.get(voice.lower(), voice)
        self.rate = rate
        self.pitch = pitch
        self.sample_rate = sample_rate
        self._edge_tts = None
    
    def _check_edge_tts(self):
        """Check if edge-tts is available."""
        try:
            import edge_tts
            self._edge_tts = edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Run: pip install edge-tts"
            )
    
    async def speak_async(self, text: str) -> torch.Tensor:
        """
        Async speech synthesis.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio tensor [1, 1, T] at self.sample_rate
        """
        self._check_edge_tts()
        
        # Create communicate instance
        communicate = self._edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Collect audio data
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        
        # Convert MP3 to tensor
        return self._mp3_to_tensor(audio_data)
    
    def speak(self, text: str) -> torch.Tensor:
        """
        Synchronous speech synthesis.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio tensor [1, 1, T] at self.sample_rate
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create new loop
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.speak_async(text))
    
    def speak_to_file(self, text: str, path: str):
        """
        Synthesize speech to file.
        
        Args:
            text: Text to synthesize
            path: Output file path (.mp3 or .wav)
        """
        self._check_edge_tts()
        
        async def _save():
            communicate = self._edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                pitch=self.pitch
            )
            await communicate.save(path)
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_save())
    
    def _mp3_to_tensor(self, mp3_data: io.BytesIO) -> torch.Tensor:
        """Convert MP3 bytes to tensor."""
        try:
            import torchaudio
            
            # Save to temp file (torchaudio needs file path for mp3)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(mp3_data.read())
                temp_path = f.name
            
            # Load and resample
            audio, sr = torchaudio.load(temp_path)
            
            # Clean up
            Path(temp_path).unlink()
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            # Ensure shape is [1, 1, T]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2:
                audio = audio.unsqueeze(0)
            
            # Mix to mono if stereo
            if audio.shape[1] > 1:
                audio = audio.mean(dim=1, keepdim=True)
            
            return audio
            
        except ImportError:
            raise ImportError(
                "torchaudio required for audio processing. Run: pip install torchaudio"
            )
    
    @staticmethod
    async def list_voices(language: str = "en") -> List[TTSVoice]:
        """List available voices for a language."""
        try:
            import edge_tts
        except ImportError:
            return []
        
        voices = await edge_tts.list_voices()
        result = []
        for v in voices:
            if v["Locale"].startswith(language):
                result.append(TTSVoice(
                    name=v["ShortName"],
                    language=v["Locale"],
                    gender=v["Gender"]
                ))
        return result


class MockTTS:
    """
    Mock TTS for testing without edge-tts installed.
    """
    
    def __init__(self, sample_rate: int = 24000, **kwargs):
        self.sample_rate = sample_rate
        self.call_count = 0
    
    def speak(self, text: str) -> torch.Tensor:
        self.call_count += 1
        # Return silence with length proportional to text
        duration_samples = len(text) * 100
        return torch.zeros(1, 1, duration_samples)
    
    async def speak_async(self, text: str) -> torch.Tensor:
        return self.speak(text)


def get_tts(voice: str = "guy", mock: bool = False) -> StreamingTTS:
    """
    Factory function to get TTS instance.
    
    Args:
        voice: Voice name or shorthand (guy, jenny, aria, etc.)
        mock: If True, return mock TTS (for testing)
    
    Returns:
        TTS instance
    """
    if mock:
        return MockTTS()
    return StreamingTTS(voice=voice)
