"""
Real-time semantic memory graph for voice agent.

Uses sentence-transformers for meaningful text embeddings and retrieval.

Features:
- Store text/transcript memories with real embeddings
- Fast similarity search
- Temporal context linking
- Cross-modal associations (audio ↔ text ↔ speaker)

Install: pip install sentence-transformers
"""

import time
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class MemoryNode:
    """Single memory node in the graph."""
    id: str
    content: str  # Text content (transcript, note, etc.)
    embedding: torch.Tensor
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)  # IDs of linked nodes


class SentenceEmbedder:
    """
    Real semantic embeddings using sentence-transformers.
    
    Uses a pretrained model (all-MiniLM-L6-v2) for fast, meaningful embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.embed_dim = 384  # MiniLM output dim
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: str | list[str]) -> torch.Tensor:
        """
        Encode text(s) to embedding vectors.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Embeddings tensor [N, embed_dim]
        """
        model = self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        return embeddings


class MockSentenceEmbedder:
    """Mock embedder for testing without sentence-transformers."""
    
    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
    
    def encode(self, texts: str | list[str]) -> torch.Tensor:
        """Generate deterministic embeddings from text content."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Use text hash for deterministic embedding
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:4], 'big')
            torch.manual_seed(seed)
            emb = torch.randn(self.embed_dim)
            embeddings.append(F.normalize(emb, dim=0))
        
        return torch.stack(embeddings)


def get_embedder(use_pretrained: bool = True, device: str = "cpu"):
    """Factory to get sentence embedder."""
    if use_pretrained:
        try:
            return SentenceEmbedder(device=device)
        except ImportError:
            print("Warning: sentence-transformers not available, using mock")
            return MockSentenceEmbedder()
    return MockSentenceEmbedder()


class SemanticMemoryGraph:
    """
    Graph-based semantic memory with real embeddings.
    
    Features:
    - Text storage with pretrained embeddings
    - Fast similarity search
    - Temporal linking
    - Persistence
    
    Example:
        memory = SemanticMemoryGraph(use_pretrained=True)
        memory.store("User asked about weather", speaker="user1")
        results = memory.search("What's the temperature?", top_k=3)
    """
    
    def __init__(
        self,
        use_pretrained: bool = True,
        device: str = "cpu",
        max_nodes: int = 10000,
        similarity_threshold: float = 0.5
    ):
        self.embedder = get_embedder(use_pretrained, device)
        self.embed_dim = self.embedder.embed_dim
        self.max_nodes = max_nodes
        self.similarity_threshold = similarity_threshold
        
        self.nodes: dict[str, MemoryNode] = {}
        
        # Matrix for fast search
        self._embeddings: torch.Tensor | None = None
        self._node_ids: list[str] = []
    
    def _generate_id(self, content: str, timestamp: float) -> str:
        """Generate unique ID."""
        h = hashlib.sha256(f"{content}{timestamp}".encode())
        return h.hexdigest()[:16]
    
    def _rebuild_index(self):
        """Rebuild the embedding index for fast search."""
        if not self.nodes:
            self._embeddings = None
            self._node_ids = []
            return
        
        self._node_ids = list(self.nodes.keys())
        embeddings = [self.nodes[nid].embedding for nid in self._node_ids]
        self._embeddings = torch.stack(embeddings)
    
    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        auto_link: bool = True
    ) -> str:
        """
        Store content in memory graph.
        
        Args:
            content: Text content to store
            metadata: Optional metadata (speaker, turn_id, etc.)
            auto_link: Automatically link to similar nodes
        
        Returns:
            Node ID
        """
        timestamp = time.time()
        node_id = self._generate_id(content, timestamp)
        embedding = self.embedder.encode(content).squeeze(0)
        
        # Find similar nodes for linking
        links = []
        if auto_link and self._embeddings is not None:
            similar = self.search(content, top_k=3, threshold=self.similarity_threshold)
            links = [nid for nid, _, _ in similar if nid != node_id]
        
        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            timestamp=timestamp,
            metadata=metadata or {},
            links=links
        )
        
        # Evict oldest if at capacity
        if len(self.nodes) >= self.max_nodes:
            oldest_id = min(self.nodes.keys(), key=lambda k: self.nodes[k].timestamp)
            del self.nodes[oldest_id]
        
        self.nodes[node_id] = node
        self._rebuild_index()
        
        return node_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
        filter_speaker: str | None = None
    ) -> list[tuple[str, float, MemoryNode]]:
        """
        Search for similar memories.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity (default: self.similarity_threshold)
            filter_speaker: Only return results from this speaker
        
        Returns:
            List of (node_id, similarity, node) tuples
        """
        if not self.nodes or self._embeddings is None:
            return []
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Encode query
        query_emb = self.embedder.encode(query).squeeze(0)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0),
            self._embeddings
        )
        
        # Filter and sort
        results = []
        for i, (node_id, sim) in enumerate(zip(self._node_ids, similarities)):
            if sim < threshold:
                continue
            
            node = self.nodes[node_id]
            
            # Apply speaker filter
            if filter_speaker and node.metadata.get("speaker") != filter_speaker:
                continue
            
            results.append((node_id, sim.item(), node))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_recent(self, n: int = 10) -> list[MemoryNode]:
        """Get n most recent memories."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_nodes[:n]
    
    def get_by_speaker(self, speaker: str) -> list[MemoryNode]:
        """Get all memories from a speaker."""
        return [
            node for node in self.nodes.values()
            if node.metadata.get("speaker") == speaker
        ]
    
    def clear(self):
        """Clear all memories."""
        self.nodes.clear()
        self._embeddings = None
        self._node_ids = []
    
    def save(self, path: str | Path):
        """Save memory graph to file."""
        import json
        
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "embedding": node.embedding.tolist(),
                    "timestamp": node.timestamp,
                    "metadata": node.metadata,
                    "links": node.links
                }
                for node in self.nodes.values()
            ]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str | Path):
        """Load memory graph from file."""
        import json
        
        with open(path) as f:
            data = json.load(f)
        
        self.nodes.clear()
        for node_data in data["nodes"]:
            node = MemoryNode(
                id=node_data["id"],
                content=node_data["content"],
                embedding=torch.tensor(node_data["embedding"]),
                timestamp=node_data["timestamp"],
                metadata=node_data["metadata"],
                links=node_data["links"]
            )
            self.nodes[node.id] = node
        
        self._rebuild_index()


class CrossModalMemory:
    """
    Cross-modal memory linking audio, text, and speaker.
    
    Associates transcript memories with speaker identities
    for contextual retrieval.
    """
    
    def __init__(self, use_pretrained: bool = True, device: str = "cpu"):
        self.text_memory = SemanticMemoryGraph(use_pretrained, device)
        self.speaker_turns: dict[str, list[str]] = {}  # speaker_id -> node_ids
    
    def store_turn(
        self,
        transcript: str,
        speaker_id: str,
        turn_id: str | None = None,
        audio_tokens: torch.Tensor | None = None
    ) -> str:
        """
        Store a conversation turn.
        
        Args:
            transcript: Text transcript
            speaker_id: Speaker identifier
            turn_id: Optional turn identifier
            audio_tokens: Optional associated audio tokens
        
        Returns:
            Memory node ID
        """
        metadata = {
            "speaker": speaker_id,
            "turn_id": turn_id,
            "has_audio": audio_tokens is not None
        }
        
        node_id = self.text_memory.store(transcript, metadata)
        
        # Track by speaker
        if speaker_id not in self.speaker_turns:
            self.speaker_turns[speaker_id] = []
        self.speaker_turns[speaker_id].append(node_id)
        
        return node_id
    
    def search_context(
        self,
        query: str,
        speaker_id: str | None = None,
        top_k: int = 5
    ) -> list[tuple[str, float, MemoryNode]]:
        """
        Search for relevant context.
        
        Args:
            query: Search query
            speaker_id: Optional speaker filter
            top_k: Number of results
        
        Returns:
            Similar memories
        """
        return self.text_memory.search(
            query,
            top_k=top_k,
            filter_speaker=speaker_id
        )
    
    def get_conversation_history(
        self,
        n_turns: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent conversation history."""
        recent = self.text_memory.get_recent(n_turns)
        return [
            {
                "speaker": node.metadata.get("speaker", "unknown"),
                "content": node.content,
                "timestamp": node.timestamp
            }
            for node in recent
        ]
