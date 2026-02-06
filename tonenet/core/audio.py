"""
Audio file compression and reconstruction utilities.

Simple interface for file-based operations:
- compress_audio(): Load audio file → ToneNet codes → binary file
- decompress_audio(): Binary file → audio file
- AudioCodec: High-level wrapper for batch processing
"""

import struct
import wave
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    torchaudio = None
    HAS_TORCHAUDIO = False

import torch
from .codec import ToneNetCodec


class AudioCodec:
    """
    High-level audio file compression/decompression.
    
    Example:
        >>> codec = AudioCodec()
        >>> codec.compress("input.wav", "compressed.tnc")
        >>> codec.decompress("compressed.tnc", "output.wav")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        sample_rate: int = 24000,
        n_quantizers: int = 8
    ):
        """
        Args:
            model_path: Path to trained model checkpoint (None = untrained)
            device: 'cpu' or 'cuda'
            sample_rate: Target sample rate
            n_quantizers: Number of quantizers (1-8, affects bitrate)
        """
        self.device = device
        self.sample_rate = sample_rate
        self.n_quantizers = n_quantizers
        
        # Load or create model
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint.get('config', {})
            self.model = ToneNetCodec(
                sample_rate=config.get('sample_rate', sample_rate),
                num_quantizers=config.get('num_quantizers', 8),
                codebook_size=config.get('codebook_size', 1024)
            ).to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = ToneNetCodec(sample_rate=sample_rate).to(device)
        
        self.model.eval()
    
    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file to numpy array."""
        path = Path(path)
        
        if HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(str(path))
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze().numpy(), self.sample_rate
        else:
            # Fallback to wave module (WAV only)
            with wave.open(str(path), 'rb') as f:
                sr = f.getframerate()
                n_frames = f.getnframes()
                n_channels = f.getnchannels()
                audio = np.frombuffer(f.readframes(n_frames), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)
            return audio, sr
    
    def save_audio(self, audio: np.ndarray, path: str, sample_rate: Optional[int] = None):
        """Save numpy array to audio file."""
        sr = sample_rate or self.sample_rate
        path = Path(path)
        
        if HAS_TORCHAUDIO:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            torchaudio.save(str(path), waveform, sr)
        else:
            # Fallback to wave module
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(str(path), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes(audio_int16.tobytes())
    
    def compress(
        self,
        input_path: str,
        output_path: str,
        n_quantizers: Optional[int] = None
    ) -> dict:
        """
        Compress audio file to ToneNet format.
        
        Args:
            input_path: Input audio file (WAV, MP3, FLAC, etc.)
            output_path: Output compressed file (.tnc)
            n_quantizers: Override default quantizer count
        
        Returns:
            dict with compression statistics
        """
        n_q = n_quantizers or self.n_quantizers
        
        # Load audio
        audio, original_sr = self.load_audio(input_path)
        original_samples = len(audio)
        
        # Pad to frame boundary
        hop = 320  # 75 Hz frame rate at 24kHz
        pad_len = (hop - len(audio) % hop) % hop
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))
        
        # Encode
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            codes = self.model.encode(audio_t, n_quantizers=n_q)
        
        num_frames = codes[0].shape[-1]
        
        # Write compressed format
        # Header: [magic][version][n_quantizers][codebook_size][num_frames][sample_rate][original_samples]
        # Body: [codes for each quantizer as uint16]
        with open(output_path, 'wb') as f:
            f.write(b'TNC2')  # Magic number
            f.write(struct.pack('B', 2))  # Version
            f.write(struct.pack('B', len(codes)))  # n_quantizers
            f.write(struct.pack('H', self.model.quantizer.codebook_size))  # codebook_size
            f.write(struct.pack('I', num_frames))  # num_frames
            f.write(struct.pack('I', self.sample_rate))  # sample_rate
            f.write(struct.pack('I', original_samples))  # original_samples
            
            for code in codes:
                code_np = code.cpu().numpy().squeeze().astype(np.uint16)
                f.write(code_np.tobytes())
        
        # Statistics
        original_bits = original_samples * 16
        compressed_bits = Path(output_path).stat().st_size * 8
        bitrate_info = self.model.get_bitrate(n_quantizers=n_q)
        
        return {
            'input': input_path,
            'output': output_path,
            'original_samples': original_samples,
            'original_duration_s': original_samples / self.sample_rate,
            'num_frames': num_frames,
            'n_quantizers': len(codes),
            'compression_ratio': original_bits / compressed_bits,
            'bitrate_kbps': bitrate_info['bitrate_kbps'],
            'file_size_bytes': Path(output_path).stat().st_size
        }
    
    def decompress(
        self,
        input_path: str,
        output_path: str
    ) -> dict:
        """
        Decompress ToneNet file to audio.
        
        Args:
            input_path: Compressed .tnc file
            output_path: Output audio file (WAV)
        
        Returns:
            dict with decompression statistics
        """
        # Read compressed format
        with open(input_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'TNC2':
                raise ValueError(f"Invalid ToneNet file (magic: {magic})")
            
            version = struct.unpack('B', f.read(1))[0]
            n_quantizers = struct.unpack('B', f.read(1))[0]
            codebook_size = struct.unpack('H', f.read(2))[0]
            num_frames = struct.unpack('I', f.read(4))[0]
            sample_rate = struct.unpack('I', f.read(4))[0]
            original_samples = struct.unpack('I', f.read(4))[0]
            
            codes = []
            for _ in range(n_quantizers):
                code_bytes = f.read(num_frames * 2)
                code = np.frombuffer(code_bytes, dtype=np.uint16).astype(np.int64)
                codes.append(torch.from_numpy(code).unsqueeze(0).to(self.device))
        
        # Decode
        with torch.no_grad():
            audio_t = self.model.decode(codes)
        
        audio = audio_t.cpu().numpy().squeeze()
        
        # Trim to original length
        audio = audio[:original_samples]
        
        # Save
        self.save_audio(audio, output_path, sample_rate)
        
        return {
            'input': input_path,
            'output': output_path,
            'duration_s': original_samples / sample_rate,
            'sample_rate': sample_rate,
            'n_quantizers': n_quantizers
        }
    
    def roundtrip(
        self,
        input_path: str,
        output_path: str,
        n_quantizers: Optional[int] = None
    ) -> dict:
        """
        Compress and immediately decompress (for testing quality).
        
        Args:
            input_path: Input audio file
            output_path: Output audio file
            n_quantizers: Number of quantizers
        
        Returns:
            dict with statistics
        """
        n_q = n_quantizers or self.n_quantizers
        
        # Load
        audio, _ = self.load_audio(input_path)
        original_samples = len(audio)
        
        # Pad
        hop = 320
        pad_len = (hop - len(audio) % hop) % hop
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))
        
        # Forward pass
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstructed, outputs = self.model(audio_t, n_quantizers=n_q)
        
        recon = reconstructed.cpu().numpy().squeeze()[:original_samples]
        
        # Save
        self.save_audio(recon, output_path)
        
        # SNR
        from .metrics import compute_snr
        snr = compute_snr(audio[:original_samples], recon)
        
        return {
            'input': input_path,
            'output': output_path,
            'duration_s': original_samples / self.sample_rate,
            'n_quantizers': n_q,
            'bitrate_kbps': self.model.get_bitrate(n_q)['bitrate_kbps'],
            'snr_db': snr,
            'vq_loss': outputs['vq_loss'].item()
        }


# Convenience functions
def compress_audio(
    input_path: str,
    output_path: str,
    n_quantizers: int = 8,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> dict:
    """
    Compress audio file to ToneNet format.
    
    Args:
        input_path: Input audio file
        output_path: Output .tnc file
        n_quantizers: 1-8 (lower = smaller file, worse quality)
        model_path: Path to trained model (None = untrained)
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with compression statistics
    """
    codec = AudioCodec(model_path=model_path, device=device, n_quantizers=n_quantizers)
    return codec.compress(input_path, output_path)


def decompress_audio(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> dict:
    """
    Decompress ToneNet file to audio.
    
    Args:
        input_path: Compressed .tnc file
        output_path: Output audio file
        model_path: Path to trained model
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with decompression statistics
    """
    codec = AudioCodec(model_path=model_path, device=device)
    return codec.decompress(input_path, output_path)


def reconstruct_audio(
    input_path: str,
    output_path: str,
    n_quantizers: int = 8,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> dict:
    """
    Compress and immediately reconstruct audio (quality test).
    
    Args:
        input_path: Input audio file
        output_path: Output reconstructed audio file
        n_quantizers: 1-8
        model_path: Path to trained model
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with SNR and other statistics
    """
    codec = AudioCodec(model_path=model_path, device=device, n_quantizers=n_quantizers)
    return codec.roundtrip(input_path, output_path, n_quantizers)
