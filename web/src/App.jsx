import React, { useState, useEffect, useRef, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { Mic, Square, Activity, Send } from 'lucide-react';

// Audio config
const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 4096;

function App() {
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('Disconnected');
  const [isRunning, setIsRunning] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false); // Agent speaking

  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  // WebSocket
  const { sendMessage, lastMessage, readyState } = useWebSocket('ws://localhost:8000/ws', {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
  });

  // Handle incoming messages
  useEffect(() => {
    if (lastMessage !== null) {
      const data = JSON.parse(lastMessage.data);

      if (data.type === 'status') {
        setStatus(data.status === 'running' ? 'Active' : 'Stopped');
        setIsRunning(data.status === 'running');
      } else if (data.type === 'transcript') {
        addMessage(data.role, data.text);
      } else if (data.type === 'audio_chunk') {
        playAudioChunk(data.data);
      } else if (data.type === 'audio_stop') {
        setIsSpeaking(false);
      }
    }
  }, [lastMessage]);

  const addMessage = (role, text) => {
    setMessages(prev => [...prev, { role, text, id: Date.now() }]);
  };

  // Audio Playback Queue
  const audioQueue = useRef([]);
  const isPlayingRef = useRef(false);

  const playAudioChunk = async (base64Data) => {
    // Decode base64 to Int16
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    // Enqueue
    audioQueue.current.push(float32);
    processAudioQueue();
  };

  const processAudioQueue = async () => {
    if (isPlayingRef.current || audioQueue.current.length === 0) return;
    isPlayingRef.current = true;
    setIsSpeaking(true);

    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }

    const chunk = audioQueue.current.shift();
    const buffer = audioContextRef.current.createBuffer(1, chunk.length, 24000); // Server sends 24k
    buffer.getChannelData(0).set(chunk);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    source.onended = () => {
      isPlayingRef.current = false;
      if (audioQueue.current.length === 0) setIsSpeaking(false);
      processAudioQueue();
    };
    source.start();
  };

  // Audio Capture
  const startRecording = async () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = ctx; // Reuse context?

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const source = ctx.createMediaStreamSource(stream);
      // Deprecated but works everywhere for raw access
      const processor = ctx.createScriptProcessor(BUFFER_SIZE, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!isRunning) return;

        const inputData = e.inputBuffer.getChannelData(0);

        // Downsample/Convert to Int16
        // NOTE: We asked for 16k ctx, so buffer should be 16k.
        // Convert Float32 -> Int16
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          let s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Base64 encode
        // Quick buffer to base64
        const binary = String.fromCharCode(...new Uint8Array(int16Data.buffer));
        const b64 = btoa(binary);

        sendMessage(JSON.stringify({
          action: 'audio_input',
          data: b64
        }));
      };

      source.connect(processor);
      processor.connect(ctx.destination); // Needed to keep processor alive
      processorRef.current = processor;

      sendMessage(JSON.stringify({ action: 'start', config: { mock: true } }));
      setIsRunning(true);

    } catch (err) {
      console.error("Mic error:", err);
      alert("Microphone access denied");
    }
  };

  const stopRecording = () => {
    sendMessage(JSON.stringify({ action: 'stop' }));

    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (processorRef.current) processorRef.current.disconnect();
    // Don't close context to keep playback alive? Or close it.
    // audioContextRef.current?.close(); 

    setIsRunning(false);
    setStatus('Stopped');
  };

  const toggleRun = () => {
    if (isRunning) stopRecording();
    else startRecording();
  };

  return (
    <div className="container">
      <h1>ToneNet</h1>

      <div className="orb-container">
        <div className={`orb ${isRunning ? 'active' : ''} ${isSpeaking ? 'speaking' : ''}`} />
      </div>

      <div className="status">
        {status} • {readyState === ReadyState.OPEN ? 'Connected' : 'Connecting...'}
      </div>

      <div className="chat-window">
        {messages.length === 0 && (
          <div className="message agent" style={{ textAlign: 'center', color: '#888' }}>
            Start the agent to begin conversation.
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.text}
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', paddingBottom: '20px' }}>
        <button
          className={`primary ${isRunning ? 'stop' : 'start'}`}
          onClick={toggleRun}
          style={{ width: '64px', height: '64px', borderRadius: '50%', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: isRunning ? '#ff3b30' : '#0071e3' }}
        >
          {isRunning ? <Square size={24} fill="currentColor" /> : <Mic size={28} />}
        </button>
      </div>
    </div>
  );
}

export default App;
