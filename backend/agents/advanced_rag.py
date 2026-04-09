"""
Advanced RAG Retriever for Medical VQA
Implements: hybrid search (dense + BM25), metadata filtering,
cross-encoder reranking, and confidence-based validation.
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

from logger import get_logger

_log = get_logger("agents.advanced_rag")


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
    - Hybrid retrieval (dense embeddings + BM25)
    - Metadata pre-filtering (modality, body_region, finding_category)
    - Mandatory cross-encoder reranking
    - Multi-signal confidence-based validation
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
        self._finding_index: Dict[str, List[int]] = defaultdict(list)
        self._device = self._resolve_device()
        _log.info("Using device: %s", self._device)

        self._is_indexed = False

    def _shorten_text(self, text: str, limit: int = 120) -> str:
        clean_text = " ".join(str(text or "").split())
        if len(clean_text) <= limit:
            return clean_text
        return clean_text[: max(0, limit - 1)].rstrip() + "…"

    def _log_top_hits(self, label: str, hits: list, limit: int = 3) -> None:
        if not hits:
            _log.debug("%s: no hits", label)
            return

        preview_count = min(limit, len(hits))
        _log.debug("%s: showing top %d/%d", label, preview_count, len(hits))
        for rank, item in enumerate(hits[:preview_count], start=1):
            if isinstance(item, RetrievalResult):
                idx = item.metadata.get("id", "n/a")
                score = item.score
                softmax_score = item.metadata.get("rerank_softmax")
                question = item.question
                answer = item.answer
                method = f" method={item.retrieval_method}"
                if softmax_score is not None:
                    method += f" softmax={softmax_score:.4f}"
            else:
                idx, score = item
                example = self._train_examples[idx]
                question = example.get("question", "")
                answer = example.get("answer", "")
                method = ""

            _log.debug(
                "  %d. id=%s score=%.4f%s | Q: %s | A: %s",
                rank,
                idx,
                score,
                method,
                self._shorten_text(question, 100),
                self._shorten_text(answer, 90),
            )

    def _resolve_device(self) -> str:
        """Pick a safe runtime device for sentence-transformers models."""
        requested = os.getenv("ADVANCED_RAG_DEVICE", "cpu").strip().lower()

        if requested == "cpu":
            return "cpu"

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested
            _log.warning("ADVANCED_RAG_DEVICE requests CUDA but CUDA is unavailable, falling back to CPU")
            return "cpu"

        if requested != "auto":
            _log.warning("Unrecognized ADVANCED_RAG_DEVICE='%s', defaulting to CPU", requested)
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major < 7:
                _log.warning(
                    "GPU compute capability %d.%d is below recommended threshold for this PyTorch build; using CPU",
                    major,
                    minor,
                )
                return "cpu"

            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as exc:
            _log.warning("CUDA probe failed (%s); using CPU", exc)
            return "cpu"
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            _log.info("Loading embedding model: %s on %s", self.embedding_model_name, self._device)
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=self._device)
        return self._embedding_model

    def _get_cross_encoder(self) -> CrossEncoder:
        """Lazy load cross-encoder for reranking"""
        if self._cross_encoder is None:
            _log.info("Loading cross-encoder: %s on %s", self.cross_encoder_name, self._device)
            self._cross_encoder = CrossEncoder(self.cross_encoder_name, device=self._device)
        return self._cross_encoder

    def load_train_data(self) -> bool:
        """Load and index VQA-RAD train split"""
        json_path = os.path.join(self.data_dir, "vqa_rad.json")

        if not os.path.exists(json_path):
            _log.error("Dataset not found at %s. Run download_dataset.py first.", json_path)
            return False

        _log.info("Loading dataset from %s", json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # Filter train split
        train_data = [item for item in all_data if item.get('split') == 'train']

        _log.info("Loaded %d training examples from dataset", len(train_data))
        
        # Store with computed metadata
        self._train_examples = []
        for idx, item in enumerate(train_data):
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            modality = self._infer_modality(question, answer)
            body_region = self._infer_body_region(question, answer)
            finding_category = self._infer_finding_category(question, answer)
            answer_style = self._infer_answer_style(answer)
            laterality = self._infer_laterality(question)

            dense_text, sparse_text = self._build_case_retrieval_texts(
                question=question,
                answer=answer,
                modality=modality,
                body_region=body_region,
                finding_category=finding_category,
                answer_style=answer_style,
                laterality=laterality,
            )
            
            example = {
                'id': idx,
                'question': question,
                'answer': answer,
                'image_name': item.get('image_name', ''),
                'modality': modality,
                'body_region': body_region,
                'finding_category': finding_category,
                'answer_style': answer_style,
                'laterality': laterality,
                'combined_text': sparse_text,
                'embedding_text': dense_text,
                'dense_text': dense_text,
                'sparse_text': sparse_text,
            }
            
            self._train_examples.append(example)
            
            # Build metadata indices
            if modality and modality != 'unknown':
                self._modality_index[modality.lower()].append(idx)
            if body_region and body_region != 'unknown':
                self._body_region_index[body_region.lower()].append(idx)
            if finding_category and finding_category != 'other':
                self._finding_index[finding_category.lower()].append(idx)
        
        return True

    def _build_case_retrieval_texts(
        self,
        question: str,
        answer: str,
        modality: str,
        body_region: str,
        finding_category: str,
        answer_style: str,
        laterality: str,
    ) -> Tuple[str, str]:
        """Build structured retrieval text for dense and sparse indexing.

        Dense text stays concise and question-centric with minimal metadata.
        Sparse text is more verbose and label-rich to help lexical matching.
        """
        dense_parts = [question]
        if modality and modality != 'unknown':
            dense_parts.append(modality)
        if body_region and body_region != 'unknown':
            dense_parts.append(body_region)
        if finding_category and finding_category != 'other':
            dense_parts.append(finding_category)

        sparse_parts = [
            f"Question: {question}",
            f"Answer: {answer}",
        ]
        if modality and modality != 'unknown':
            sparse_parts.append(f"Modality: {modality}")
        if body_region and body_region != 'unknown':
            sparse_parts.append(f"Body region: {body_region}")
        if finding_category and finding_category != 'other':
            sparse_parts.append(f"Finding category: {finding_category}")
        if laterality and laterality != 'unknown':
            sparse_parts.append(f"Laterality: {laterality}")
        if answer_style and answer_style != 'unknown':
            sparse_parts.append(f"Answer style: {answer_style}")

        return " ".join(part for part in dense_parts if part), "\n".join(part for part in sparse_parts if part)
    
    def _infer_modality(self, question: str, answer: str = '') -> str:
        """Infer imaging modality from question and answer text"""
        combined = f"{question} {answer}".lower()
        modalities = {
            'x-ray': 'x-ray', 'xray': 'x-ray', 'radiograph': 'x-ray', 'plain film': 'x-ray',
            'computed tomography': 'ct', 'cat scan': 'ct',
            'magnetic resonance': 'mri',
            'ultrasound': 'ultrasound', 'sonogram': 'ultrasound', 'sonography': 'ultrasound',
            'fundus': 'fundus', 'fundoscopy': 'fundus',
            'mammogram': 'mammogram', 'mammography': 'mammogram',
            'dermoscopy': 'dermoscopy',
            'fluoroscopy': 'fluoroscopy',
            'nuclear': 'nuclear', 'scintigraphy': 'nuclear', 'pet scan': 'nuclear',
            'angiography': 'angiography', 'angiogram': 'angiography',
        }
        # Multi-word before single-word to avoid partial false matches
        for keyword, modality in modalities.items():
            if keyword in combined:
                return modality
        # Single-char ambiguous tokens only after longer forms ruled out
        for token, modality in [('mri', 'mri'), ('ct', 'ct'), ('us', 'ultrasound')]:
            pattern = rf'\b{re.escape(token)}\b'
            if re.search(pattern, combined):
                return modality
        return 'unknown'

    def _infer_body_region(self, question: str, answer: str = '') -> str:
        """Infer body region from question and answer text"""
        combined = f"{question} {answer}".lower()
        regions = {
            'thorax': 'chest', 'pulmonary': 'chest', 'pleural': 'chest',
            'cardiac': 'chest', 'pericardial': 'chest', 'mediastin': 'chest',
            'cranial': 'head', 'cerebral': 'head', 'brainstem': 'head',
            'cerebell': 'head', 'ventricle': 'head', 'hippocampus': 'head',
            'abdominal': 'abdomen', 'hepatic': 'abdomen', 'splenic': 'abdomen',
            'renal': 'abdomen', 'bowel': 'abdomen', 'intestin': 'abdomen',
            'pancrea': 'abdomen', 'biliary': 'abdomen', 'gallbladder': 'abdomen',
            'bladder': 'pelvis', 'uterus': 'pelvis', 'ovarian': 'pelvis',
            'prostate': 'pelvis', 'pelvic': 'pelvis',
            'femur': 'musculoskeletal', 'tibia': 'musculoskeletal', 'fibula': 'musculoskeletal',
            'humerus': 'musculoskeletal', 'vertebra': 'musculoskeletal', 'spinal': 'musculoskeletal',
            'shoulder': 'musculoskeletal', 'elbow': 'musculoskeletal', 'wrist': 'musculoskeletal',
            'ankle': 'musculoskeletal', 'hip': 'musculoskeletal',
            'ocular': 'eye', 'retina': 'eye', 'optic': 'eye', 'cornea': 'eye',
            'dermal': 'skin', 'subcutaneous': 'skin',
            'thyroid': 'neck', 'cervical': 'neck', 'laryngeal': 'neck',
        }
        for keyword, region in regions.items():
            if keyword in combined:
                return region
        # Shorter/exact-match keywords checked with word boundaries
        for token, region in [
            ('chest', 'chest'), ('lung', 'chest'), ('heart', 'chest'),
            ('head', 'head'), ('brain', 'head'), ('skull', 'head'),
            ('abdomen', 'abdomen'), ('liver', 'abdomen'), ('kidney', 'abdomen'), ('spleen', 'abdomen'),
            ('knee', 'musculoskeletal'), ('bone', 'musculoskeletal'), ('joint', 'musculoskeletal'),
            ('eye', 'eye'), ('fundus', 'eye'),
            ('skin', 'skin'), ('lesion', 'skin'),
            ('neck', 'neck'), ('spine', 'musculoskeletal'),
        ]:
            if re.search(rf'\b{re.escape(token)}\b', combined):
                return region
        return 'unknown'

    def _infer_finding_category(self, question: str, answer: str) -> str:
        """Infer broad finding category from question and answer text"""
        combined = f"{question} {answer}".lower()
        categories = [
            ('normal',       ['normal', 'no abnormality', 'unremarkable', 'within normal', 'no finding']),
            ('fracture',     ['fracture', 'broken', 'break', 'cortical disruption']),
            ('hemorrhage',   ['hemorrhage', 'hemorrhagic', 'bleed', 'bleeding', 'hematoma', 'haemorrhage']),
            ('infarct',      ['infarct', 'infarction', 'ischemi', 'ischemia', 'stroke']),
            ('mass',         ['mass', 'tumor', 'tumour', 'neoplasm', 'carcinoma', 'malignant', 'cancer', 'nodule']),
            ('effusion',     ['effusion', 'pleural fluid', 'ascites', 'fluid collection']),
            ('edema',        ['edema', 'oedema', 'swelling', 'pulmonary edema']),
            ('pneumonia',    ['pneumonia', 'consolidation', 'opacity', 'infiltrate']),
            ('atelectasis',  ['atelectasis', 'collapse', 'collapsed']),
            ('pneumothorax', ['pneumothorax', 'air in pleural']),
            ('lesion',       ['lesion', 'abnormality', 'abnormal', 'defect', 'density', 'lucency']),
            ('device',       ['catheter', 'tube', 'pacemaker', 'stent', 'wire', 'implant', 'hardware', 'device']),
        ]
        for category, keywords in categories:
            for kw in keywords:
                if kw in combined:
                    return category
        return 'other'

    def _infer_answer_style(self, answer: str) -> str:
        """Classify answer as close_ended (yes/no, single value) or open_ended"""
        ans = answer.strip().lower()
        if ans in {'yes', 'no', 'y', 'n', 'true', 'false'}:
            return 'close_ended'
        if len(ans.split()) <= 3:
            return 'close_ended'
        return 'open_ended'

    def _infer_laterality(self, question: str) -> str:
        """Infer laterality from question text"""
        q = question.lower()
        if 'bilateral' in q or 'both sides' in q:
            return 'bilateral'
        has_left = bool(re.search(r'\bleft\b', q))
        has_right = bool(re.search(r'\bright\b', q))
        if has_left and has_right:
            return 'bilateral'
        if has_left:
            return 'left'
        if has_right:
            return 'right'
        return 'unknown'
    
    def build_index(self) -> bool:
        """Build dense embeddings and sparse BM25 index"""
        if not self._train_examples:
            if not self.load_train_data():
                return False

        _log.info("Building indices for %d examples...", len(self._train_examples))

        # 1. Dense embeddings with FAISS (question + metadata for richer semantic matching)
        _log.info("Computing dense embeddings (model=%s) on structured case text...", self.embedding_model_name)
        embedding_texts = [ex['embedding_text'] for ex in self._train_examples]
        model = self._get_embedding_model()
        self._embeddings = model.encode(embedding_texts, show_progress_bar=True, convert_to_numpy=True)

        # Get actual embedding dimension
        self._embedding_dim = self._embeddings.shape[1]

        # Build FAISS index for fast similarity search
        _log.info("Building FAISS index (dim=%d)...", self._embedding_dim)
        self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)  # Inner product (cosine similarity with normalized vectors)

        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(self._embeddings)
        self._faiss_index.add(self._embeddings)

        _log.info("FAISS index built with %d vectors", self._faiss_index.ntotal)

        # 2. Sparse BM25 index (uses structured question+answer text for keyword coverage)
        _log.info("Building BM25 sparse index on structured case text...")
        bm25_texts = [ex['combined_text'] for ex in self._train_examples]
        self._tokenized_corpus = [
            self._tokenize(text) for text in bm25_texts
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        self._is_indexed = True
        _log.info("Indexing complete — FAISS + BM25 ready")
        return True
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Lowercase, remove punctuation, split
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return tokens

    # Retrieval confidence thresholds
    # Lowered from original (0.55, 0.15) to allow more valid retrievals through
    _TOP1_SOFTMAX_THRESHOLD = 0.30  # Was 0.55 - cross-encoder softmax confidence
    _SOFTMAX_MARGIN_THRESHOLD = 0.08  # Was 0.15 - gap between top 1 and 2
    _DENSE_TOP1_THRESHOLD = 0.35  # Dense retrieval minimum score
    _AGREEMENT_TOPK = 5  # Check if top reranked candidate appears in both dense/sparse top-5

    def retrieve(
        self,
        question: str,
        modality: str = None,
        body_region: str = None,
        finding_category: str = None,
        notable_findings: list = None,
    ) -> List[RetrievalResult]:
        """
        Full advanced RAG retrieval pipeline.
        No LLM calls during retrieval — uses a multi-signal confidence gate instead of
        self-reflection, and skips query rewriting.

        Args:
            question: The query question
            modality: Imaging modality (e.g., 'x-ray', 'ct')
            body_region: Body region (e.g., 'chest', 'head')
            finding_category: Finding category to filter by
            notable_findings: List of findings from image interpretation (used to infer finding_category if not provided)
        """
        if not self._is_indexed:
            if not self.build_index():
                return []

        # Infer finding_category from notable_findings if not provided
        if finding_category is None and notable_findings:
            notable_findings_str = " ".join(notable_findings) if isinstance(notable_findings, list) else str(notable_findings)
            finding_category = self._infer_finding_category(question, notable_findings_str)
            _log.debug("Inferred finding_category='%s' from notable_findings", finding_category)

        _log.info(
            "Retrieval pipeline started | question=%s | modality=%s | body_region=%s | finding=%s | rerank=mandatory",
            self._shorten_text(question, 140),
            modality or "unknown",
            body_region or "unknown",
            finding_category or "unknown",
        )

        # 1. Use original question directly (no LLM rewrite)
        query = question
        _log.debug("Using original question as query (no LLM rewrite)")

        # 2. Metadata Filtering
        candidate_indices = self._apply_metadata_filter(modality, body_region, finding_category)
        if candidate_indices is None:
            _log.debug("Metadata filter: none applied (full corpus search)")
        else:
            _log.info("Metadata filter: %d candidates retained", len(candidate_indices))

        # 3. Hybrid Retrieval
        _log.info("Dense retrieval: top_k=%d", self.top_k_dense)
        dense_results = self._dense_retrieval(query, candidate_indices)
        self._log_top_hits("Dense retrieval results", dense_results)

        _log.info("Sparse (BM25) retrieval: top_k=%d", self.top_k_sparse)
        sparse_results = self._sparse_retrieval(query, candidate_indices)
        self._log_top_hits("Sparse retrieval results", sparse_results)

        # Merge and deduplicate
        merged = self._merge_results(dense_results, sparse_results)
        _log.info("Hybrid RRF merge produced %d unique candidates", len(merged))
        self._log_top_hits("Merged candidates", merged)

        # 4. Reranking (mandatory)
        _log.info(
            "Reranking %d candidates down to top_k_final=%d (cross-encoder=%s)",
            len(merged),
            self.top_k_final,
            self.cross_encoder_name,
        )
        reranked = self._rerank(query, merged)
        self._log_top_hits("Reranked candidates", reranked)

        # 5. Confidence-based validation (replaces LLM self-reflection)
        confidence = self._compute_retrieval_confidence(dense_results, sparse_results, reranked)
        validated = reranked if confidence["is_confident"] else []

        _log.info(
            "Retrieval confidence | level=%s weak_signals=%d top1_softmax=%.3f margin=%.3f dense_top1=%.3f agreement=%s",
            confidence["level"],
            confidence["weak_signals"],
            confidence["top1_softmax"],
            confidence["softmax_margin"],
            confidence["dense_top1"],
            confidence["top_candidate_in_dense_and_sparse_top5"],
        )

        if confidence["level"] == "medium":
            _log.warning("Retrieval confidence is medium; returning results")

        _log.info("Retrieval complete: returning %d final candidates", len(validated))
        self._log_top_hits("Final retrieved cases", validated)

        return validated
    
    def _apply_metadata_filter(
        self,
        modality: str = None,
        body_region: str = None,
        finding_category: str = None,
    ) -> Optional[List[int]]:
        """Apply metadata pre-filtering to reduce search space.

        Uses smart intersection: tries full filter (modality + body_region + finding),
        but gracefully degrades to partial filters if intersection is empty.
        Finding category is sparse, so it's applied last and skipped if it causes emptiness.
        """
        if not modality and not body_region and not finding_category:
            _log.debug("Metadata filter inputs empty; skipping filter")
            return None  # No filtering

        # Build individual filter sets
        modality_set = None
        body_region_set = None
        finding_set = None

        if modality and modality != 'unknown':
            modality_lower = modality.lower()
            matching = self._modality_index.get(modality_lower, [])
            if not matching:
                for mod_key, indices in self._modality_index.items():
                    if modality_lower in mod_key or mod_key in modality_lower:
                        matching = indices
                        break
            if matching:
                _log.debug("Metadata filter modality='%s': %d matches", modality_lower, len(matching))
                modality_set = set(matching)
            else:
                _log.debug("Metadata filter modality='%s': no exact matches", modality_lower)

        if body_region and body_region != 'unknown':
            region_lower = body_region.lower()
            matching = self._body_region_index.get(region_lower, [])
            if not matching:
                for region_key, indices in self._body_region_index.items():
                    if region_lower in region_key or region_key in region_lower:
                        matching = indices
                        break
            if matching:
                _log.debug("Metadata filter body_region='%s': %d matches", region_lower, len(matching))
                body_region_set = set(matching)
            else:
                _log.debug("Metadata filter body_region='%s': no exact matches", region_lower)

        if finding_category and finding_category not in ('unknown', 'other'):
            finding_lower = finding_category.lower()
            matching = self._finding_index.get(finding_lower, [])
            if not matching:
                for find_key, indices in self._finding_index.items():
                    if finding_lower in find_key or find_key in finding_lower:
                        matching = indices
                        break
            if matching:
                _log.debug("Metadata filter finding='%s': %d matches", finding_lower, len(matching))
                finding_set = set(matching)
            else:
                _log.debug("Metadata filter finding='%s': no exact matches", finding_lower)

        # Smart intersection: try full filter first, then degrade gracefully
        candidates = None
        filter_applied = []

        # Step 1: Apply modality filter (base)
        if modality_set is not None:
            candidates = modality_set
            filter_applied.append(f"modality({len(modality_set)})")

        # Step 2: Intersect with body_region if available
        if body_region_set is not None:
            if candidates is not None:
                intersected = candidates.intersection(body_region_set)
                if intersected:
                    candidates = intersected
                    filter_applied.append(f"body_region({len(body_region_set)})")
                else:
                    _log.warning("Modality+body_region intersection empty; using body_region only")
                    candidates = body_region_set
                    filter_applied = [f"body_region({len(body_region_set)})"]
            else:
                candidates = body_region_set
                filter_applied.append(f"body_region({len(body_region_set)})")

        # Step 3: Try to intersect with finding, but skip if it causes emptiness
        if finding_set is not None and candidates is not None:
            intersected = candidates.intersection(finding_set)
            if intersected:
                candidates = intersected
                filter_applied.append(f"finding({len(finding_set)})")
            else:
                _log.info("Skipping finding_category filter: would reduce candidates to zero")

        if candidates:
            result = list(candidates)
            _log.info("Metadata filter result: %d candidates (%s)", len(result), " + ".join(filter_applied))
            return result

        if modality or body_region:
            _log.warning("Metadata filter found no candidates; falling back to full corpus")

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
            _log.debug("Dense retrieval using filtered subset of %d candidates", len(candidate_indices))
        else:
            _log.debug("Dense retrieval searching full FAISS index with %d vectors", self._faiss_index.ntotal)
        
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
            _log.debug("Sparse (BM25) retrieval using filtered subset of %d candidates", len(candidate_indices))
        else:
            _log.debug("Sparse (BM25) retrieval searching full corpus of %d documents", len(self._tokenized_corpus))
        
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

        _log.debug("Merging results via RRF: dense=%d sparse=%d", len(dense), len(sparse))
        
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
        if not candidates:
            _log.debug("Reranking skipped: no candidates provided")
            return []

        _log.info("Cross-encoder reranking %d candidates...", len(candidates))
        
        model = self._get_cross_encoder()
        
        # Prepare pairs for cross-encoder
        pairs = [
            (query, f"{c.question} {c.answer}")
            for c in candidates
        ]
        
        # Get reranking scores
        scores = np.asarray(model.predict(pairs), dtype=np.float32)
        shifted_scores = scores - float(np.max(scores))
        exp_scores = np.exp(shifted_scores)
        score_sum = float(np.sum(exp_scores))
        if score_sum > 0.0:
            softmax_scores = exp_scores / score_sum
        else:
            softmax_scores = np.full_like(exp_scores, 1.0 / len(exp_scores))
        
        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate.score = float(scores[i])
            candidate.metadata["rerank_softmax"] = float(softmax_scores[i])
            candidate.retrieval_method = 'reranked'
        
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return reranked[:self.top_k_final]

    def _compute_retrieval_confidence(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        reranked: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """Compute a multi-signal retrieval confidence summary.

        Confidence signals:
        - softmax confidence of the top reranked candidate
        - margin between the top two reranked candidates
        - top dense score
        - agreement of the top reranked candidate with dense/sparse top-k
        """
        if not reranked:
            return {
                "level": "low",
                "is_confident": False,
                "weak_signals": 4,
                "top1_softmax": 0.0,
                "softmax_margin": 0.0,
                "dense_top1": 0.0,
                "top_candidate_in_dense_and_sparse_top5": False,
            }

        top1_softmax = float(reranked[0].metadata.get("rerank_softmax", 0.0) or 0.0)
        top2_softmax = float(reranked[1].metadata.get("rerank_softmax", 0.0) or 0.0) if len(reranked) > 1 else 0.0
        softmax_margin = top1_softmax - top2_softmax

        dense_top1 = float(dense_results[0][1]) if dense_results else 0.0
        top_candidate_id = reranked[0].metadata.get("id")
        dense_top_ids = {idx for idx, _ in dense_results[: self._AGREEMENT_TOPK]}
        sparse_top_ids = {idx for idx, _ in sparse_results[: self._AGREEMENT_TOPK]}
        agreement = top_candidate_id in dense_top_ids and top_candidate_id in sparse_top_ids

        weak_signals = 0
        if top1_softmax < self._TOP1_SOFTMAX_THRESHOLD:
            weak_signals += 1
        if softmax_margin < self._SOFTMAX_MARGIN_THRESHOLD:
            weak_signals += 1
        if dense_top1 < self._DENSE_TOP1_THRESHOLD:
            weak_signals += 1
        if not agreement:
            weak_signals += 1

        if weak_signals >= 2:
            level = "low"
            is_confident = False
        elif weak_signals == 1:
            level = "medium"
            is_confident = True
        else:
            level = "high"
            is_confident = True

        return {
            "level": level,
            "is_confident": is_confident,
            "weak_signals": weak_signals,
            "top1_softmax": top1_softmax,
            "softmax_margin": softmax_margin,
            "dense_top1": dense_top1,
            "top_candidate_in_dense_and_sparse_top5": agreement,
        }
    
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
    finding_category: str = None,
    notable_findings: list = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for advanced RAG retrieval.

    No LLM calls during retrieval — uses score thresholds instead of self-reflection,
    and skips query rewriting.
    """
    retriever = get_rag_retriever()
    results = retriever.retrieve(
        question=question,
        modality=modality,
        body_region=body_region,
        finding_category=finding_category,
        notable_findings=notable_findings,
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
        "count": len(results)
    }
