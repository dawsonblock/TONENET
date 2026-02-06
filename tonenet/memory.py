"""
Real-time semantic memory graph for audio tokens.

Stores, indexes, and retrieves audio token patterns with:
- Semantic embedding search
- Temporal context linking
- Cross-modal associations (audio ↔ text ↔ speaker)
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class MemoryNode:
    """Single memory node in the graph."""
    id: str
    tokens: torch.Tensor
    embedding: torch.Tensor
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)  # IDs of linked nodes


class SemanticMemoryGraph:
    """
    Graph-based semantic memory for audio tokens.
    
    Features:
    - Token storage with embeddings
    - Similarity search
    - Temporal linking
    - Cross-modal associations
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        max_nodes: int = 10000,
        similarity_threshold: float = 0.7
    ):
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.similarity_threshold = similarity_threshold
        
        self.nodes: Dict[str, MemoryNode] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self.node_ids: List[str] = []
        
        # Simple embedding projection
        self.embed_proj = torch.nn.Linear(1024, embed_dim)
    
    def _generate_id(self, tokens: torch.Tensor) -> str:
        """Generate unique ID from token content."""
        h = hashlib.sha256(tokens.cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    
    def _compute_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute semantic embedding from tokens."""
        # Simple: mean pooling over token sequence
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        # One-hot encode and project
        one_hot = F.one_hot(tokens.long() % 1024, num_classes=1024).float()
        pooled = one_hot.mean(dim=1)  # [B, 1024]
        
        with torch.no_grad():
            embedding = self.embed_proj(pooled)
        
        return F.normalize(embedding, dim=-1)
    
    def store(
        self,
        tokens: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        auto_link: bool = True
    ) -> str:
        """
        Store tokens in memory graph.
        
        Args:
            tokens: Token tensor
            metadata: Optional metadata (speaker, text, etc.)
            auto_link: Automatically link to similar nodes
        
        Returns:
            Node ID
        """
        node_id = self._generate_id(tokens)
        embedding = self._compute_embedding(tokens)
        
        # Find similar nodes for linking
        links = []
        if auto_link and self.nodes:
            similar = self.search(tokens, top_k=3)
            links = [s[0] for s in similar if s[1] > self.similarity_threshold]
        
        node = MemoryNode(
            id=node_id,
            tokens=tokens.cpu(),
            embedding=embedding.cpu(),
            timestamp=time.time(),
            metadata=metadata or {},
            links=links
        )
        
        self.nodes[node_id] = node
        self.node_ids.append(node_id)
        
        # Update embedding matrix
        self._rebuild_embeddings()
        
        # Enforce max size
        while len(self.nodes) > self.max_nodes:
            oldest_id = self.node_ids.pop(0)
            del self.nodes[oldest_id]
            self._rebuild_embeddings()
        
        return node_id
    
    def _rebuild_embeddings(self):
        """Rebuild embedding matrix for fast search."""
        if not self.nodes:
            self.embeddings = None
            return
        
        embs = []
        for nid in self.node_ids:
            if nid in self.nodes:
                emb = self.nodes[nid].embedding
                # Squeeze to [embed_dim] if needed
                if emb.dim() > 1:
                    emb = emb.squeeze(0)
                embs.append(emb)
        
        if embs:
            self.embeddings = torch.stack(embs, dim=0)  # [N, embed_dim]
    
    def search(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar token patterns.
        
        Args:
            query: Query tokens
            top_k: Number of results
        
        Returns:
            List of (node_id, similarity_score)
        """
        if self.embeddings is None or len(self.nodes) == 0:
            return []
        
        query_emb = self._compute_embedding(query)
        
        # Cosine similarity
        sims = F.cosine_similarity(
            query_emb.unsqueeze(1),
            self.embeddings.unsqueeze(0),
            dim=-1
        ).squeeze(0)
        
        # Top-k
        k = min(top_k, len(self.node_ids))
        values, indices = torch.topk(sims, k)
        
        results = []
        indices_list = indices.tolist()
        values_list = values.tolist()
        
        # Handle scalar case when k=1
        if not isinstance(indices_list, list):
            indices_list = [indices_list]
            values_list = [values_list]
        
        for i, idx in enumerate(indices_list):
            if idx < len(self.node_ids):
                results.append((self.node_ids[idx], float(values_list[i])))
        
        return results
    
    def get(self, node_id: str) -> Optional[MemoryNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_linked(self, node_id: str, depth: int = 1) -> List[MemoryNode]:
        """Get linked nodes up to depth."""
        if node_id not in self.nodes:
            return []
        
        visited = set()
        frontier = [node_id]
        result = []
        
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                
                node = self.nodes.get(nid)
                if node:
                    result.append(node)
                    next_frontier.extend(node.links)
            
            frontier = next_frontier
        
        return result
    
    def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 10
    ) -> List[MemoryNode]:
        """Query nodes by metadata field."""
        results = []
        for node in self.nodes.values():
            if node.metadata.get(key) == value:
                results.append(node)
                if len(results) >= limit:
                    break
        return results
    
    def save(self, path: str):
        """Save memory graph to file."""
        data = {
            "nodes": {
                nid: {
                    "tokens": node.tokens.tolist(),
                    "embedding": node.embedding.tolist(),
                    "timestamp": node.timestamp,
                    "metadata": node.metadata,
                    "links": node.links
                }
                for nid, node in self.nodes.items()
            },
            "node_ids": self.node_ids
        }
        torch.save(data, path)
    
    def load(self, path: str):
        """Load memory graph from file."""
        data = torch.load(path)
        
        self.nodes = {}
        self.node_ids = data["node_ids"]
        
        for nid, ndata in data["nodes"].items():
            self.nodes[nid] = MemoryNode(
                id=nid,
                tokens=torch.tensor(ndata["tokens"]),
                embedding=torch.tensor(ndata["embedding"]),
                timestamp=ndata["timestamp"],
                metadata=ndata["metadata"],
                links=ndata["links"]
            )
        
        self._rebuild_embeddings()


class CrossModalMemory(SemanticMemoryGraph):
    """
    Extended memory with cross-modal associations.
    
    Links audio tokens to:
    - Text transcriptions
    - Speaker identities
    - Emotional markers
    - Temporal events
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Cross-modal indices
        self.text_index: Dict[str, List[str]] = {}  # text → node_ids
        self.speaker_index: Dict[str, List[str]] = {}  # speaker → node_ids
    
    def store_with_text(
        self,
        tokens: torch.Tensor,
        text: str,
        speaker_id: Optional[str] = None,
        **metadata
    ) -> str:
        """Store with text association."""
        metadata["text"] = text
        if speaker_id:
            metadata["speaker_id"] = speaker_id
        
        node_id = self.store(tokens, metadata)
        
        # Update text index
        text_key = text.lower().strip()
        if text_key not in self.text_index:
            self.text_index[text_key] = []
        self.text_index[text_key].append(node_id)
        
        # Update speaker index
        if speaker_id:
            if speaker_id not in self.speaker_index:
                self.speaker_index[speaker_id] = []
            self.speaker_index[speaker_id].append(node_id)
        
        return node_id
    
    def search_by_text(self, text: str) -> List[MemoryNode]:
        """Search by text content."""
        text_key = text.lower().strip()
        node_ids = self.text_index.get(text_key, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_by_speaker(self, speaker_id: str) -> List[MemoryNode]:
        """Get all nodes for speaker."""
        node_ids = self.speaker_index.get(speaker_id, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
