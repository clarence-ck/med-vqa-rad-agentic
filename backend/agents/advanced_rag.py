"""
Advanced RAG Retriever for Medical VQA
Implements: query rewriting, hybrid search (dense + BM25), 
metadata filtering, cross-encoder reranking, self-reflection validation
Uses FAISS for efficient vector similarity search.
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

import torch

# FAISS is required for vector search
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class RetrievalResult:
    """Result from advanced RAG retrieval"""
    question: str
    answer: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'dense', 'sparse', or 'reranked'


class AdvancedRAGRetriever:
    """
    Advanced RAG with:
    - Query rewriting/expansion
    - Hybrid retrieval (dense embeddings + BM25)
    - Metadata pre-filtering (modality, body_region)
    - Cross-encoder reranking
    - Self-reflection validation
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        top_k_final: int = 3,
        data_dir: str = None
    ):
        self.embedding_model_name = embedding_model
        self.cross_encoder_name = cross_encoder_model
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_final = top_k_final
        
        # Data directory
        if data_dir is None:
            self.data_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vqa-rad')
            )
        else:
            self.data_dir = data_dir
        
        # Models (loaded lazily)
        self._embedding_model: Optional[SentenceTransformer] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        self._openai_client: Optional[OpenAI] = None
        
        # Index storage
        self._train_examples: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        
        # Metadata indices for filtering
        self._modality_index: Dict[str, List[int]] = defaultdict(list)
        self._body_region_index: Dict[str, List[int]] = defaultdict(list)
        self._last_query_rewritten: bool = False
        self._device = self._resolve_device()
        print(f"[AdvancedRAG] Using device: {self._device}")
        
        self._is_indexed = False

    def _shorten_text(self, text: str, limit: int = 120) -> str:
        clean_text = " ".join(str(text or "").split())
        if len(clean_text) <= limit:
            return clean_text
        return clean_text[: max(0, limit - 1)].rstrip() + "…"

    def _log_top_hits(self, label: str, hits: list, limit: int = 3) -> None:
        if not hits:
            print(f"[AdvancedRAG] {label}: no hits")
            return

        preview_count = min(limit, len(hits))
        print(f"[AdvancedRAG] {label}: showing top {preview_count}/{len(hits)}")
        for rank, item in enumerate(hits[:preview_count], start=1):
            if isinstance(item, RetrievalResult):
                idx = item.metadata.get("id", "n/a")
                score = item.score
                question = item.question
                answer = item.answer
                method = f" method={item.retrieval_method}"
            else:
                idx, score = item
                example = self._train_examples[idx]
                question = example.get("question", "")
                answer = example.get("answer", "")
                method = ""

            print(
                f"[AdvancedRAG]   {rank}. id={idx} score={score:.4f}{method} | "
                f"Q: {self._shorten_text(question, 100)} | "
                f"A: {self._shorten_text(answer, 90)}"
            )

    def _resolve_device(self) -> str:
        """Pick a safe runtime device for sentence-transformers models."""
        requested = os.getenv("ADVANCED_RAG_DEVICE", "cpu").strip().lower()

        if requested == "cpu":
            return "cpu"

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested
            print("[AdvancedRAG] ADVANCED_RAG_DEVICE requests CUDA but CUDA is unavailable, falling back to CPU")
            return "cpu"

        if requested != "auto":
            print(f"[AdvancedRAG] Unrecognized ADVANCED_RAG_DEVICE='{requested}', defaulting to CPU")
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major < 7:
                print(
                    "[AdvancedRAG] GPU compute capability "
                    f"{major}.{minor} is below recommended threshold for this PyTorch build; using CPU"
                )
                return "cpu"

            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as exc:
            print(f"[AdvancedRAG] CUDA probe failed ({exc}); using CPU")
            return "cpu"
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            print(f"[AdvancedRAG] Loading embedding model: {self.embedding_model_name} on {self._device}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self._device)
        return self._embedding_model
    
    def _get_cross_encoder(self) -> CrossEncoder:
        """Lazy load cross-encoder for reranking"""
        if self._cross_encoder is None:
            print(f"[AdvancedRAG] Loading cross-encoder: {self.cross_encoder_name} on {self._device}")
            self._cross_encoder = CrossEncoder(self.cross_encoder_name, device=self._device)
        return self._cross_encoder
    
    def _get_openai_client(self) -> OpenAI:
        """Lazy load OpenAI client for query rewriting"""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return self._openai_client
    
    def load_train_data(self) -> bool:
        """Load and index VQA-RAD train split"""
        json_path = os.path.join(self.data_dir, "vqa_rad.json")
        
        if not os.path.exists(json_path):
            print(f"[AdvancedRAG] Dataset not found at {json_path}. Run download_dataset.py first.")
            return False
        
        print(f"[AdvancedRAG] Loading dataset from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Filter train split
        train_data = [item for item in all_data if item.get('split') == 'train']
        
        print(f"[AdvancedRAG] Loaded {len(train_data)} training examples")
        
        # Store with computed metadata
        self._train_examples = []
        for idx, item in enumerate(train_data):
            # Infer modality and body_region from question if not present
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # Simple keyword extraction for metadata
            modality = self._infer_modality(question)
            body_region = self._infer_body_region(question)
            
            example = {
                'id': idx,
                'question': question,
                'answer': answer,
                'image_name': item.get('image_name', ''),
                'modality': modality,
                'body_region': body_region,
                'combined_text': f"{question} {answer}"
            }
            
            self._train_examples.append(example)
            
            # Build metadata indices
            if modality:
                self._modality_index[modality.lower()].append(idx)
            if body_region:
                self._body_region_index[body_region.lower()].append(idx)
        
        return True
    
    def _infer_modality(self, question: str) -> str:
        """Infer imaging modality from question text"""
        question_lower = question.lower()
        modalities = {
            'x-ray': 'x-ray', 'xray': 'x-ray', 'radiograph': 'x-ray',
            'ct': 'ct', 'cat scan': 'ct', 'computed tomography': 'ct',
            'mri': 'mri', 'magnetic resonance': 'mri',
            'ultrasound': 'ultrasound', 'us': 'ultrasound', 'sonogram': 'ultrasound',
            'fundus': 'fundus', 'fundoscopy': 'fundus',
            'dermoscopy': 'dermoscopy', 'skin': 'dermoscopy',
            'mammogram': 'mammogram', 'mammography': 'mammogram'
        }
        
        for keyword, modality in modalities.items():
            if keyword in question_lower:
                return modality
        return 'unknown'
    
    def _infer_body_region(self, question: str) -> str:
        """Infer body region from question text"""
        question_lower = question.lower()
        regions = {
            'chest': 'chest', 'thorax': 'chest', 'lung': 'chest', 'pulmonary': 'chest',
            'heart': 'chest', 'cardiac': 'chest',
            'head': 'head', 'brain': 'head', 'cranial': 'head', 'skull': 'head',
            'abdomen': 'abdomen', 'abdominal': 'abdomen', 'liver': 'abdomen',
            'kidney': 'abdomen', 'renal': 'abdomen', 'spleen': 'abdomen',
            'knee': 'knee', 'joint': 'knee', 'bone': 'knee',
            'eye': 'eye', 'ocular': 'eye', 'retina': 'eye', 'fundus': 'eye',
            'skin': 'skin', 'dermal': 'skin', 'lesion': 'skin'
        }
        
        for keyword, region in regions.items():
            if keyword in question_lower:
                return region
        return 'unknown'
    
    def build_index(self) -> bool:
        """Build dense embeddings and sparse BM25 index"""
        if not self._train_examples:
            if not self.load_train_data():
                return False
        
        print("[AdvancedRAG] Building indices...")
        
        # 1. Dense embeddings with FAISS
        print(f"[AdvancedRAG] Computing embeddings for {len(self._train_examples)} examples...")
        texts = [ex['combined_text'] for ex in self._train_examples]
        model = self._get_embedding_model()
        self._embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Get actual embedding dimension
        self._embedding_dim = self._embeddings.shape[1]
        
        # Build FAISS index for fast similarity search
        print(f"[AdvancedRAG] Building FAISS index (dim={self._embedding_dim})...")
        self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)  # Inner product (cosine similarity with normalized vectors)
        
        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(self._embeddings)
        self._faiss_index.add(self._embeddings)
        
        print(f"[AdvancedRAG] FAISS index built with {self._faiss_index.ntotal} vectors")
        
        # 2. Sparse BM25 index
        print("[AdvancedRAG] Building BM25 index...")
        self._tokenized_corpus = [
            self._tokenize(text) for text in texts
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        
        self._is_indexed = True
        print("[AdvancedRAG] Indexing complete!")
        return True
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Lowercase, remove punctuation, split
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return tokens
    
    def rewrite_query(
        self,
        question: str,
        modality: str = None,
        body_region: str = None
    ) -> str:
        """
        Query rewriting/expansion using LLM
        Expands medical abbreviations and adds synonyms
        """
        try:
            client = self._get_openai_client()
            
            context_parts = []
            if modality and modality != 'unknown':
                context_parts.append(f"Imaging modality: {modality}")
            if body_region and body_region != 'unknown':
                context_parts.append(f"Body region: {body_region}")
            
            context = "\n".join(context_parts) if context_parts else "No specific context provided."
            
            prompt = f"""Rewrite and expand the following medical question to improve retrieval effectiveness.
Add relevant medical synonyms, expand abbreviations, and include related terms.

Original question: {question}
{context}

Return ONLY the expanded query (1-2 sentences), no explanations."""
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.3
            )
            
            expanded = response.choices[0].message.content.strip()
            return expanded if expanded else question
            
        except Exception as e:
            print(f"[AdvancedRAG] Query rewriting failed: {e}")
            return question
    
    def retrieve(
        self,
        question: str,
        modality: str = None,
        body_region: str = None,
        use_query_rewrite: bool = True,
        use_reranking: bool = True,
        use_self_reflection: bool = True
    ) -> List[RetrievalResult]:
        """
        Full advanced RAG retrieval pipeline
        """
        if not self._is_indexed:
            if not self.build_index():
                return []

        print(
            "[AdvancedRAG] Retrieval pipeline started | "
            f"question={self._shorten_text(question, 140)} | "
            f"modality={modality or 'unknown'} | body_region={body_region or 'unknown'} | "
            f"rewrite={use_query_rewrite} rerank={use_reranking} reflection={use_self_reflection}"
        )
        
        # 1. Query Rewriting
        if use_query_rewrite:
            query = self.rewrite_query(question, modality, body_region)
            self._last_query_rewritten = query.strip() != question.strip()
            print(f"[AdvancedRAG] Original: {question}")
            print(f"[AdvancedRAG] Rewritten: {query}")
            if self._last_query_rewritten:
                print("[AdvancedRAG] Query rewrite: changed")
            else:
                print("[AdvancedRAG] Query rewrite: unchanged")
        else:
            query = question
            self._last_query_rewritten = False
            print("[AdvancedRAG] Query rewrite disabled; using original question")
        
        # 2. Metadata Filtering
        candidate_indices = self._apply_metadata_filter(modality, body_region)
        if candidate_indices is None:
            print("[AdvancedRAG] Metadata filter: none applied (full corpus search)")
        else:
            print(f"[AdvancedRAG] Metadata filter: {len(candidate_indices)} candidates retained")
        
        # 3. Hybrid Retrieval
        print(f"[AdvancedRAG] Dense retrieval: top_k={self.top_k_dense}")
        dense_results = self._dense_retrieval(query, candidate_indices)
        self._log_top_hits("Dense retrieval results", dense_results)

        print(f"[AdvancedRAG] Sparse retrieval: top_k={self.top_k_sparse}")
        sparse_results = self._sparse_retrieval(query, candidate_indices)
        self._log_top_hits("Sparse retrieval results", sparse_results)
        
        # Merge and deduplicate
        merged = self._merge_results(dense_results, sparse_results)
        print(f"[AdvancedRAG] Hybrid merge produced {len(merged)} unique candidates")
        self._log_top_hits("Merged candidates", merged)
        
        # 4. Reranking (if enabled)
        if use_reranking and len(merged) > self.top_k_final:
            print(
                f"[AdvancedRAG] Reranking enabled: reranking {len(merged)} candidates "
                f"down to top_k_final={self.top_k_final}"
            )
            reranked = self._rerank(query, merged)
        else:
            reranked = merged[:self.top_k_final]
            print(f"[AdvancedRAG] Reranking skipped; using first {len(reranked)} merged candidates")
        self._log_top_hits("Reranked candidates", reranked)
        
        # 5. Self-Reflection Validation (if enabled)
        if use_self_reflection:
            print("[AdvancedRAG] Self-reflection enabled: validating retrieved candidates")
            validated = self._self_reflect(question, reranked, modality, body_region)
            if validated is None:  # Reflection indicated poor quality
                print("[AdvancedRAG] Self-reflection flagged low quality, returning best available")
                validated = reranked[:self.top_k_final]
            else:
                print("[AdvancedRAG] Self-reflection accepted retrieved candidates")
        else:
            validated = reranked[:self.top_k_final]
            print("[AdvancedRAG] Self-reflection disabled")

        print(f"[AdvancedRAG] Retrieval complete: returning {len(validated)} final candidates")
        self._log_top_hits("Final retrieved cases", validated)
        
        return validated
    
    def _apply_metadata_filter(
        self,
        modality: str = None,
        body_region: str = None
    ) -> Optional[List[int]]:
        """Apply metadata pre-filtering to reduce search space"""
        if not modality and not body_region:
            print("[AdvancedRAG] Metadata filter inputs empty; skipping filter")
            return None  # No filtering
        
        candidates = None
        
        if modality and modality != 'unknown':
            modality_lower = modality.lower()
            # Try exact match first, then partial
            matching = self._modality_index.get(modality_lower, [])
            if not matching:
                # Partial match
                for mod_key, indices in self._modality_index.items():
                    if modality_lower in mod_key or mod_key in modality_lower:
                        matching = indices
                        break
            if matching:
                print(f"[AdvancedRAG] Metadata filter modality='{modality_lower}': {len(matching)} matches")
                candidates = set(matching)
            else:
                print(f"[AdvancedRAG] Metadata filter modality='{modality_lower}': no exact matches")
        
        if body_region and body_region != 'unknown':
            region_lower = body_region.lower()
            matching = self._body_region_index.get(region_lower, [])
            if not matching:
                for region_key, indices in self._body_region_index.items():
                    if region_lower in region_key or region_key in region_lower:
                        matching = indices
                        break
            if matching:
                print(f"[AdvancedRAG] Metadata filter body_region='{region_lower}': {len(matching)} matches")
                if candidates is not None:
                    candidates = candidates.intersection(set(matching))
                else:
                    candidates = set(matching)
            else:
                print(f"[AdvancedRAG] Metadata filter body_region='{region_lower}': no exact matches")
        
        if candidates:
            result = list(candidates)
            print(f"[AdvancedRAG] Metadata filter: {len(result)} candidates")
            return result

        if modality or body_region:
            print("[AdvancedRAG] Metadata filter found no intersection; falling back to full corpus")
        
        return None
    
    def _dense_retrieval(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Dense retrieval using FAISS for fast similarity search"""
        model = self._get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        if candidate_indices is not None:
            print(f"[AdvancedRAG] Dense retrieval using filtered subset of {len(candidate_indices)} candidates")
        else:
            print(f"[AdvancedRAG] Dense retrieval searching full FAISS index with {self._faiss_index.ntotal} vectors")
        
        if candidate_indices is not None:
            # For filtered subsets, fall back to numpy search
            # (FAISS doesn't easily support arbitrary subset search without rebuilding index)
            candidate_embeddings = self._embeddings[candidate_indices]
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
            
            # Map back to original indices
            indexed_results = [
                (candidate_indices[i], float(score))
                for i, score in enumerate(similarities)
            ]
            indexed_results.sort(key=lambda x: x[1], reverse=True)
        else:
            # Use FAISS for full search - much faster
            scores, indices = self._faiss_index.search(query_embedding, self.top_k_dense)
            indexed_results = [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0])
            ]
        
        return indexed_results[:self.top_k_dense]
    
    def _sparse_retrieval(
        self,
        query: str,
        candidate_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Sparse retrieval using BM25"""
        tokenized_query = self._tokenize(query)

        if candidate_indices is not None:
            print(f"[AdvancedRAG] Sparse retrieval using filtered subset of {len(candidate_indices)} candidates")
        else:
            print(f"[AdvancedRAG] Sparse retrieval searching full BM25 corpus of {len(self._tokenized_corpus)} documents")
        
        if candidate_indices is not None:
            # Rebuild BM25 for filtered subset
            filtered_corpus = [self._tokenized_corpus[i] for i in candidate_indices]
            bm25_filtered = BM25Okapi(filtered_corpus)
            scores = bm25_filtered.get_scores(tokenized_query)
            
            indexed_results = [
                (candidate_indices[i], float(score))
                for i, score in enumerate(scores)
            ]
        else:
            scores = self._bm25.get_scores(tokenized_query)
            indexed_results = [
                (i, float(score)) for i, score in enumerate(scores)
            ]
        
        # Sort by score descending
        indexed_results.sort(key=lambda x: x[1], reverse=True)
        return indexed_results[:self.top_k_sparse]
    
    def _merge_results(
        self,
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]]
    ) -> List[RetrievalResult]:
        """Merge dense and sparse results using reciprocal rank fusion"""
        # Reciprocal rank fusion
        k = 60  # RRF constant
        fused_scores: Dict[int, float] = defaultdict(float)

        print(f"[AdvancedRAG] Merging results: dense={len(dense)} sparse={len(sparse)}")
        
        # Add dense scores
        for rank, (idx, _) in enumerate(dense):
            fused_scores[idx] += 1.0 / (k + rank + 1)
        
        # Add sparse scores
        for rank, (idx, _) in enumerate(sparse):
            fused_scores[idx] += 1.0 / (k + rank + 1)
        
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to RetrievalResult
        results = []
        for idx, score in sorted_results[:max(self.top_k_dense, self.top_k_sparse)]:
            example = self._train_examples[idx]
            results.append(RetrievalResult(
                question=example['question'],
                answer=example['answer'],
                score=score,
                metadata={
                    'modality': example['modality'],
                    'body_region': example['body_region'],
                    'id': example['id']
                },
                retrieval_method='hybrid_rrf'
            ))
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Cross-encoder reranking"""
        if len(candidates) <= self.top_k_final:
            print("[AdvancedRAG] Reranking skipped because candidate count is already within top_k_final")
            return candidates
        
        print(f"[AdvancedRAG] Reranking {len(candidates)} candidates...")
        
        model = self._get_cross_encoder()
        
        # Prepare pairs for cross-encoder
        pairs = [
            (query, f"{c.question} {c.answer}")
            for c in candidates
        ]
        
        # Get reranking scores
        scores = model.predict(pairs)
        
        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate.score = float(scores[i])
            candidate.retrieval_method = 'reranked'
        
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return reranked[:self.top_k_final]
    
    def _self_reflect(
        self,
        original_question: str,
        retrieved: List[RetrievalResult],
        modality: str = None,
        body_region: str = None
    ) -> Optional[List[RetrievalResult]]:
        """
        Self-reflection: validate retrieved context quality
        Returns None if reflection suggests poor quality
        """
        try:
            client = self._get_openai_client()
            print(f"[AdvancedRAG] Self-reflection evaluating {len(retrieved)} retrieved candidates")
            
            # Prepare context for reflection
            context = f"""Original question: {original_question}
Image context: Modality={modality or 'unknown'}, Region={body_region or 'unknown'}

Retrieved similar cases:
"""
            for i, r in enumerate(retrieved, 1):
                context += f"\n{i}. Q: {r.question}\n   A: {r.answer}"
            
            prompt = f"""You are validating retrieved context for a medical VQA system.

{context}

Evaluate the quality of these retrieved cases for answering the original question.
Consider:
1. Relevance: Do the retrieved cases relate to the same medical topic?
2. Modality match: Do they match the imaging modality (if specified)?
3. Consistency: Are the answers consistent with expected medical knowledge?

Respond with ONLY a JSON object:
{{
  "is_relevant": true/false,
  "confidence": "high/medium/low",
  "issues": ["list any issues or empty"],
  "recommendation": "use/augment/reject"
}}"""
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                is_relevant = result.get('is_relevant', True)
                recommendation = result.get('recommendation', 'use')
                confidence = result.get('confidence', 'unknown')
                issues = result.get('issues', [])
                
                print(
                    f"[AdvancedRAG] Self-reflection: relevant={is_relevant}, "
                    f"confidence={confidence}, recommendation={recommendation}"
                )
                if issues:
                    print(f"[AdvancedRAG] Self-reflection issues: {issues}")
                
                if not is_relevant or recommendation == 'reject':
                    return None
                
                # Could add feedback loop here for 'augment'
                return retrieved
            
            return retrieved
            
        except Exception as e:
            print(f"[AdvancedRAG] Self-reflection failed: {e}")
            return retrieved  # Fail open


# Global singleton instance
_rag_retriever: Optional[AdvancedRAGRetriever] = None


def get_rag_retriever() -> AdvancedRAGRetriever:
    """Get or create the global RAG retriever instance"""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = AdvancedRAGRetriever()
        _rag_retriever.build_index()
    return _rag_retriever


def retrieve_knowledge_advanced(
    question: str,
    modality: str = None,
    body_region: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for advanced RAG retrieval
    Returns formatted result for KnowledgeRetriever integration
    """
    retriever = get_rag_retriever()
    results = retriever.retrieve(
        question=question,
        modality=modality,
        body_region=body_region,
        **kwargs
    )
    
    return {
        "similar_cases": [
            {
                "question": r.question,
                "answer": r.answer,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results
        ],
        "count": len(results),
        "query_rewritten": retriever._last_query_rewritten
    }
