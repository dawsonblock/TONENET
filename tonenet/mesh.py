"""
Multi-agent audio mesh for networked voice coordination.

Enables:
- Token stream exchange between agents
- Distributed voice synthesis
- Coordinated multi-speaker scenarios
- Mesh topology with routing
"""

import json
import socket
import pickle
import threading
import queue
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import torch


@dataclass
class MeshMessage:
    """Message exchanged in audio mesh."""
    sender_id: str
    message_type: str  # "tokens", "control", "sync", "heartbeat"
    payload: Any
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class MeshPeer:
    """Peer node in the mesh."""
    peer_id: str
    host: str
    port: int
    connected: bool = False
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioMeshNode:
    """
    Single node in multi-agent audio mesh.
    
    Features:
    - Token stream exchange (not raw audio)
    - Automatic peer discovery
    - Message routing
    - Synchronization
    """
    
    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 7700,
        max_peers: int = 10
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.max_peers = max_peers
        
        # Peer registry
        self.peers: Dict[str, MeshPeer] = {}
        
        # Message queues
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        
        # State
        self.running = False
        self.sequence = 0
        
        # Callbacks
        self.handlers: Dict[str, Callable] = {
            "tokens": self._handle_tokens,
            "control": self._handle_control,
            "sync": self._handle_sync,
            "heartbeat": self._handle_heartbeat,
        }
        
        # Server socket
        self.server_socket: Optional[socket.socket] = None
        self.threads: List[threading.Thread] = []
    
    def start(self):
        """Start mesh node."""
        self.running = True
        
        # Start server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_peers)
        self.server_socket.settimeout(1.0)
        
        # Server thread
        t = threading.Thread(target=self._server_loop, daemon=True)
        t.start()
        self.threads.append(t)
        
        # Sender thread
        t = threading.Thread(target=self._sender_loop, daemon=True)
        t.start()
        self.threads.append(t)
    
    def stop(self):
        """Stop mesh node."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def connect_peer(self, peer_id: str, host: str, port: int) -> bool:
        """Connect to a peer node."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.close()
            
            self.peers[peer_id] = MeshPeer(
                peer_id=peer_id,
                host=host,
                port=port,
                connected=True,
                last_seen=time.time()
            )
            return True
        except Exception:
            return False
    
    def send_tokens(
        self,
        tokens: torch.Tensor,
        target_id: Optional[str] = None
    ):
        """
        Send tokens to peer(s).
        
        Args:
            tokens: Token tensor to send
            target_id: Specific peer or None for broadcast
        """
        msg = MeshMessage(
            sender_id=self.node_id,
            message_type="tokens",
            payload=tokens.cpu().numpy().tolist(),
            sequence=self._next_seq()
        )
        
        if target_id:
            self.outbox.put((target_id, msg))
        else:
            # Broadcast
            for peer_id in self.peers:
                self.outbox.put((peer_id, msg))
    
    def send_control(
        self,
        command: str,
        target_id: str,
        **params
    ):
        """Send control message."""
        msg = MeshMessage(
            sender_id=self.node_id,
            message_type="control",
            payload={"command": command, **params},
            sequence=self._next_seq()
        )
        self.outbox.put((target_id, msg))
    
    def receive(self, timeout: float = 0.1) -> Optional[MeshMessage]:
        """Receive next message from inbox."""
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _next_seq(self) -> int:
        """Get next sequence number."""
        self.sequence += 1
        return self.sequence
    
    def _server_loop(self):
        """Accept incoming connections."""
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                t = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr),
                    daemon=True
                )
                t.start()
            except socket.timeout:
                continue
            except Exception:
                break
    
    def _handle_connection(self, conn: socket.socket, addr: tuple):
        """Handle incoming connection."""
        try:
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            
            if data:
                msg = pickle.loads(data)
                self._process_message(msg)
        except Exception:
            pass
        finally:
            conn.close()
    
    def _sender_loop(self):
        """Send queued messages."""
        while self.running:
            try:
                target_id, msg = self.outbox.get(timeout=0.1)
                peer = self.peers.get(target_id)
                if peer and peer.connected:
                    self._send_to_peer(peer, msg)
            except queue.Empty:
                continue
    
    def _send_to_peer(self, peer: MeshPeer, msg: MeshMessage):
        """Send message to specific peer."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((peer.host, peer.port))
            sock.sendall(pickle.dumps(msg))
            sock.close()
            peer.last_seen = time.time()
        except Exception:
            peer.connected = False
    
    def _process_message(self, msg: MeshMessage):
        """Process received message."""
        handler = self.handlers.get(msg.message_type)
        if handler:
            handler(msg)
        self.inbox.put(msg)
    
    def _handle_tokens(self, msg: MeshMessage):
        """Handle token message."""
        pass
    
    def _handle_control(self, msg: MeshMessage):
        """Handle control message."""
        pass
    
    def _handle_sync(self, msg: MeshMessage):
        """Handle sync message."""
        pass
    
    def _handle_heartbeat(self, msg: MeshMessage):
        """Handle heartbeat."""
        peer = self.peers.get(msg.sender_id)
        if peer:
            peer.last_seen = time.time()
            peer.connected = True


class MeshCoordinator:
    """
    Coordinator for multi-agent voice scenarios.
    
    Manages:
    - Turn-taking
    - Voice blending
    - Synchronized playback
    - Conflict resolution
    """
    
    def __init__(self, node: AudioMeshNode):
        self.node = node
        self.turn_queue: List[str] = []
        self.current_speaker: Optional[str] = None
        self.blend_weights: Dict[str, float] = {}
    
    def request_turn(self) -> bool:
        """Request speaking turn."""
        if self.current_speaker is None:
            self.current_speaker = self.node.node_id
            return True
        
        self.turn_queue.append(self.node.node_id)
        return False
    
    def release_turn(self):
        """Release speaking turn."""
        if self.current_speaker == self.node.node_id:
            if self.turn_queue:
                self.current_speaker = self.turn_queue.pop(0)
            else:
                self.current_speaker = None
    
    def set_blend_weight(self, peer_id: str, weight: float):
        """Set voice blend weight for peer."""
        self.blend_weights[peer_id] = max(0.0, min(1.0, weight))
    
    def blend_tokens(
        self,
        local_tokens: torch.Tensor,
        peer_tokens: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Blend tokens from multiple sources.
        
        Simple weighted average in token space.
        """
        total_weight = 1.0
        result = local_tokens.float()
        
        for peer_id, tokens in peer_tokens.items():
            weight = self.blend_weights.get(peer_id, 0.0)
            if weight > 0 and tokens.shape == local_tokens.shape:
                result = result + weight * tokens.float()
                total_weight += weight
        
        result = result / total_weight
        return result.long()
