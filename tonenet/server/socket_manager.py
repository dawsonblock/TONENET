import asyncio
import base64
import json
import logging
import queue
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..pipeline.audio_io import AudioIOConfig

logger = logging.getLogger(__name__)


class NetworkMicStream:
    """Adapts WebSocket input to the blocking iterator interface expected by RealtimeDuplex."""

    def __init__(self, config: AudioIOConfig):
        self.config = config
        self.q = queue.Queue()
        self.active = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.active:
            raise StopIteration

        # Blocking get, compatible with mic_thread loop
        return self.q.get()

    def push_chunk(self, chunk: np.ndarray):
        """Called by websocket handler to inject audio."""
        if self.active:
            self.q.put(chunk)

    def close(self):
        self.active = False
        self.q.put(np.zeros(160, dtype=np.float32))  # Unblock


class NetworkAudioOutput:
    """Adapts RealtimeDuplex audio output to WebSocket messages."""

    def __init__(self, socket_manager):
        self.manager = socket_manager
        self.is_playing = False
        self._rate = 24000

    def stop(self):
        self.is_playing = False
        # Send stop signal to UI
        asyncio.create_task(self.manager.broadcast({"type": "audio_stop"}))

    def play(
        self, audio: np.ndarray, sample_rate: int | None = None, blocking: bool = True
    ):
        self.is_playing = True
        sr = sample_rate or self._rate

        # Convert to base64 float32 or int16
        # To save bandwidth, let's use int16
        audio_int16 = (audio * 32767).astype(np.int16)
        b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        # We need to run this in the event loop, but we might be called from a thread
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                asyncio.create_task,
                self.manager.broadcast({"type": "audio_chunk", "data": b64, "sr": sr}),
            )
        except RuntimeError:
            # If no loop (e.g. testing), just skip
            pass

        if blocking:
            # Simulate playback time to throttle the loop
            dur = len(audio) / sr
            time.sleep(dur)

        self.is_playing = False


class SocketManager:
    """Manages the WebSocket connection and duplex agent lifecycle."""

    def __init__(self):
        self.active_connection: WebSocket | None = None
        self.duplex = None
        self.mic_stream = None
        self.thread = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        logger.info("Client connected")

    def disconnect(self):
        self.active_connection = None
        if self.duplex:
            self.duplex.stop()
        if self.mic_stream:
            self.mic_stream.close()
        logger.info("Client disconnected")

    async def broadcast(self, message: dict):
        if self.active_connection:
            try:
                await self.active_connection.send_json(message)
            except Exception as e:
                logger.error(f"Send failed: {e}")
                self.active_connection = None

    async def handle_message(self, data: dict):
        action = data.get("action")

        if action == "start":
            await self.start_agent(data.get("config", {}))

        elif action == "stop":
            if self.duplex:
                self.duplex.stop()
                self.duplex = None
            await self.broadcast({"type": "status", "status": "stopped"})

        elif action == "audio_input":
            if self.mic_stream:
                # Expect base64 float32 or int16
                raw = base64.b64decode(data["data"])
                # Assume chunks come as int16 from browser for bandwidth
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.mic_stream.push_chunk(chunk)

    async def start_agent(self, config: dict):
        """Initialize and start the RealtimeDuplex agent in a thread."""
        from tonenet.pipeline.realtime_duplex import RealtimeDuplex, DuplexConfig
        from threading import Thread

        # Create networking adapters
        self.mic_stream = NetworkMicStream(AudioIOConfig(sample_rate=16000))
        audio_out = NetworkAudioOutput(self)

        # Mock mode?
        mock = config.get("mock", False)

        # Define callback to broadcast transcripts
        def on_transcript(role, text):
            asyncio.create_task(
                self.broadcast({"type": "transcript", "role": role, "text": text})
            )

        self.duplex = RealtimeDuplex(
            mock=mock,
            audio_output=audio_out,
            cfg=DuplexConfig(),
            on_transcript=on_transcript,
        )

        # Start the duplex loop in a separate thread because it has blocking loops
        self.thread = Thread(
            target=self.duplex.run, kwargs={"mic_stream": self.mic_stream}
        )
        self.thread.daemon = True
        self.thread.start()

        await self.broadcast({"type": "status", "status": "running"})
