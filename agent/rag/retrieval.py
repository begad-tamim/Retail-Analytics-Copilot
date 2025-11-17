"""
RAG retrieval module using TF-IDF/BM25 for document retrieval.
Loads markdown documents and provides similarity-based search.
"""
import os
import re
from dataclasses import dataclass
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class DocChunk:
    """Represents a document chunk with metadata."""
    id: str
    source: str
    content: str
    score: float = 0.0


class RAGRetriever:
    """Simple TF-IDF based retrieval system for markdown documents."""
    
    def __init__(self, docs_dir: str = "docs/"):
        """Initialize the retriever and load documents.
        
        Args:
            docs_dir: Directory containing markdown documents.
        """
        self.docs_dir = docs_dir
        self.chunks: List[DocChunk] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        
        # Load and index documents
        self._load_documents()
        
    def _load_documents(self):
        """Load markdown documents and split into chunks."""
        markdown_files = [
            'marketing_calendar.md',
            'kpi_definitions.md',
            'catalog.md',
            'product_policy.md'
        ]
        
        for filename in markdown_files:
            filepath = os.path.join(self.docs_dir, filename)
            if not os.path.exists(filepath):
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraph-level chunks
            chunks = self._split_into_chunks(content, filename)
            self.chunks.extend(chunks)
        
        # Build TF-IDF index
        if self.chunks:
            chunk_texts = [chunk.content for chunk in self.chunks]
            self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
    
    def _split_into_chunks(self, content: str, source: str) -> List[DocChunk]:
        """Split document content into paragraph-level chunks.
        
        Args:
            content: Document text content.
            source: Source filename.
            
        Returns:
            List of DocChunk objects.
        """
        # Split on double newlines (paragraphs) or headers
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        chunk_counter = 1
        base_name = source.replace('.md', '')
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:  # Skip very short paragraphs
                continue
            
            chunk_id = f"{base_name}::chunk{chunk_counter}"
            chunk = DocChunk(
                id=chunk_id,
                source=source,
                content=para,
                score=0.0
            )
            chunks.append(chunk)
            chunk_counter += 1
        
        return chunks
    
    def retrieve(self, question: str, k: int = 4) -> List[DocChunk]:
        """Retrieve top-k most relevant document chunks.
        
        Args:
            question: User query.
            k: Number of chunks to retrieve.
            
        Returns:
            List of top-k DocChunk objects with similarity scores.
        """
        if not self.chunks or self.chunk_vectors is None:
            return []
        
        # Vectorize the question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_vector, self.chunk_vectors)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Create result chunks with scores
        results = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            # Create a new chunk with the score
            result_chunk = DocChunk(
                id=chunk.id,
                source=chunk.source,
                content=chunk.content,
                score=float(similarities[idx])
            )
            results.append(result_chunk)
        
        return results


# Global instance
_rag_retriever = None


def get_rag_retriever(docs_dir: str = "docs/") -> RAGRetriever:
    """Get or create a singleton RAG retriever instance.
    
    Args:
        docs_dir: Directory containing markdown documents.
        
    Returns:
        RAGRetriever instance.
    """
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever(docs_dir)
    return _rag_retriever
