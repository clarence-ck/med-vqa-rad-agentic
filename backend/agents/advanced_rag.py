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
        embedding_model: str = "NeuML/pubmedbert-base-embeddings",
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_final: int = 3,
        data_dir: str = None,
        corpus_name: str = None,
    ):
        self.embedding_model_name = embedding_model
        self.cross_encoder_name = cross_encoder_model
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_final = top_k_final
        requested_corpus = (corpus_name or os.getenv("RAG_CORPUS_NAME", "slake")).strip().lower()
        if requested_corpus != "slake":
            _log.warning(
                "Requested corpus '%s' is not supported for the SLAKE RAG path; using 'slake' instead.",
                requested_corpus,
            )
        self.corpus_name = "slake"
        
        # Data directory
        if data_dir is None:
            self.data_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'slake', 'kb')
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
        self._embedding_dim: int = 768  # PubMedBERT-base-embeddings dimension
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        
        # Metadata indices for filtering
        self._modality_index: Dict[str, List[int]] = defaultdict(list)
        self._body_region_index: Dict[str, List[int]] = defaultdict(list)
        self._finding_index: Dict[str, List[int]] = defaultdict(list)
        self._concept_type_index: Dict[str, List[int]] = defaultdict(list)
        self._target_entity_index: Dict[str, List[int]] = defaultdict(list)
        self._question_family_index: Dict[str, List[int]] = defaultdict(list)
        self._device = self._resolve_device()
        _log.info("Using device: %s", self._device)

        self._is_indexed = False

    def _resolve_corpus_file(self) -> Optional[str]:
        """Resolve the corpus file for the configured RAG source."""
        candidates = ["slake_kb.jsonl", "slake_kb.json", "slake.json"]

        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                return path
        return None

    def _load_jsonl_records(self, jsonl_path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _shorten_text(self, text: str, limit: int = 120) -> str:
        clean_text = " ".join(str(text or "").split())
        if len(clean_text) <= limit:
            return clean_text
        return clean_text[: max(0, limit - 1)].rstrip() + "…"

    def _normalize_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _build_query_profile(
        self,
        question: str,
        question_type: str = "",
        question_family: str = "",
        target_entity: str = "",
        body_region: str = "",
        modality: str = "",
        visual_cues: Optional[List[str]] = None,
        answer_style: str = "",
    ) -> str:
        parts = [question]
        if question_type:
            parts.append(f"question_type={question_type}")
        if question_family:
            parts.append(f"question_family={question_family}")
        if target_entity and target_entity != "unknown":
            parts.append(f"target_entity={target_entity}")
        if body_region and body_region != "unknown":
            parts.append(f"body_region={body_region}")
        if modality and modality != "unknown":
            parts.append(f"modality={modality}")
        if answer_style:
            parts.append(f"answer_style={answer_style}")
        if visual_cues:
            parts.append("visual_cues=" + ", ".join(visual_cues[:6]))
        return " | ".join(part for part in parts if part)

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
                concept = item.metadata.get("concept", "")
                concept_type = item.metadata.get("concept_type", "")
                method = f" method={item.retrieval_method}"
                if softmax_score is not None:
                    method += f" softmax={softmax_score:.4f}"
            else:
                idx, score = item
                example = self._train_examples[idx]
                question = example.get("question", "")
                answer = example.get("answer", "")
                concept = example.get("concept", "")
                concept_type = example.get("concept_type", "")
                method = ""

            _log.debug(
                "  %d. id=%s score=%.4f%s | concept_type=%s concept=%s | Q: %s | A: %s",
                rank,
                idx,
                score,
                method,
                concept_type,
                self._shorten_text(concept, 50),
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
        """Load and index the configured train corpus."""
        json_path = self._resolve_corpus_file()

        if not json_path:
            _log.error(
                "Dataset not found in %s for corpus=%s. Run the appropriate downloader/converter first.",
                self.data_dir,
                self.corpus_name,
            )
            return False

        _log.info("Loading dataset from %s", json_path)

        if json_path.endswith('.jsonl'):
            all_data = self._load_jsonl_records(json_path)
        else:
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
            text_chunk = item.get('text_chunk', '')
            
            modality = item.get('modality') or self._infer_modality(question, answer)
            body_region = item.get('body_region') or self._infer_body_region(question, answer)
            finding_category = item.get('finding_category') or self._infer_finding_category(question, answer)
            answer_style = item.get('answer_style') or self._infer_answer_style(answer)
            laterality = item.get('laterality') or self._infer_laterality(question)
            concept_type = item.get('concept_type', 'case')
            concept = item.get('concept', '')
            target_entity = item.get('target_entity', '')
            question_family = item.get('question_family', '')
            support_count = int(item.get('support_count', 1) or 1)

            dense_text, sparse_text = self._build_case_retrieval_texts(
                question=question,
                answer=answer,
                modality=modality,
                body_region=body_region,
                finding_category=finding_category,
                answer_style=answer_style,
                laterality=laterality,
                text_chunk=text_chunk,
                concept_type=concept_type,
                concept=concept,
                target_entity=target_entity,
                question_family=question_family,
            )
            
            example = {
                'id': idx,
                'source': item.get('source', self.corpus_name),
                'question': question,
                'answer': answer,
                'image_name': item.get('image_name', ''),
                'modality': modality,
                'body_region': body_region,
                'finding_category': finding_category,
                'answer_style': answer_style,
                'laterality': laterality,
                'concept_type': concept_type,
                'concept': concept,
                'target_entity': target_entity,
                'question_family': question_family,
                'visual_cues': item.get('visual_cues', []),
                'support_count': support_count,
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
            if concept_type and concept_type != 'case':
                self._concept_type_index[concept_type.lower()].append(idx)
            if target_entity and target_entity != 'unknown':
                self._target_entity_index[self._normalize_key(target_entity)].append(idx)
            if question_family and question_family != 'other':
                self._question_family_index[question_family.lower()].append(idx)
        
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
        text_chunk: str = "",
        concept_type: str = "",
        concept: str = "",
        target_entity: str = "",
        question_family: str = "",
    ) -> Tuple[str, str]:
        """Build structured retrieval text for dense and sparse indexing.

        Dense text is concept-driven and compact, prioritizing concept metadata over raw question text.
        Sparse text is more verbose and label-rich to help lexical matching.
        """
        dense_parts = [part for part in [
            text_chunk,
            f"Concept type: {concept_type}" if concept_type and concept_type != 'case' else "",
            f"Concept: {concept}" if concept and concept != 'unknown' else "",
            f"Target entity: {target_entity}" if target_entity and target_entity != 'unknown' else "",
            f"Question family: {question_family}" if question_family and question_family != 'other' else "",
            f"Modality: {modality}" if modality and modality != 'unknown' else "",
            f"Body region: {body_region}" if body_region and body_region != 'unknown' else "",
            f"Finding category: {finding_category}" if finding_category and finding_category != 'other' else "",
            f"Laterality: {laterality}" if laterality and laterality != 'unknown' else "",
            f"Answer style: {answer_style}" if answer_style and answer_style != 'unknown' else "",
        ] if part]

        sparse_parts = [
            f"Question: {question}",
            f"Answer: {answer}",
        ]
        if text_chunk:
            sparse_parts.append(f"Text chunk: {text_chunk}")
        if concept_type and concept_type != 'case':
            sparse_parts.append(f"Concept type: {concept_type}")
        if concept and concept != 'unknown':
            sparse_parts.append(f"Concept: {concept}")
        if target_entity and target_entity != 'unknown':
            sparse_parts.append(f"Target entity: {target_entity}")
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
        """Simple English-only tokenization for BM25"""
        cleaned = str(text or "").lower()
        return re.findall(r'\b[a-zA-Z0-9]+\b', cleaned)

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
        question_type: str = None,
        question_family: str = None,
        target_entity: str = None,
        answer_style: str = None,
        visual_cues: list = None,
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

        query_profile = self._build_query_profile(
            question=question,
            question_type=question_type or "",
            question_family=question_family or "",
            target_entity=target_entity or "",
            body_region=body_region or "",
            modality=modality or "",
            visual_cues=visual_cues or notable_findings,
            answer_style=answer_style or "",
        )

        _log.info(
            "Retrieval pipeline started | question=%s | modality=%s | body_region=%s | finding=%s | rerank=mandatory",
            self._shorten_text(question, 140),
            modality or "unknown",
            body_region or "unknown",
            finding_category or "unknown",
        )

        # 1. Use original question directly (no LLM rewrite)
        query = query_profile
        _log.debug("Using structured query profile (no LLM rewrite): %s", self._shorten_text(query_profile, 240))

        # 2. Metadata Filtering
        candidate_indices = self._apply_metadata_filter(
            modality=modality,
            body_region=body_region,
            finding_category=finding_category,
            concept_type=question_family or question_type,
            target_entity=target_entity,
            answer_style=answer_style,
        )
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
        concept_type: str = None,
        target_entity: str = None,
        answer_style: str = None,
    ) -> Optional[List[int]]:
        """Apply metadata pre-filtering to reduce search space.

        Uses smart intersection: tries full filter (modality + body_region + finding),
        but gracefully degrades to partial filters if intersection is empty.
        Finding category is sparse, so it's applied last and skipped if it causes emptiness.
        """
        if not modality and not body_region and not finding_category and not concept_type and not target_entity and not answer_style:
            _log.debug("Metadata filter inputs empty; skipping filter")
            return None  # No filtering

        filter_sets: List[Tuple[str, set, bool]] = []

        def _resolve_matches(index: Dict[str, List[int]], value: str) -> List[int]:
            if not value or value in ("unknown", "other"):
                return []
            normalized = value.lower()
            matching = index.get(normalized, [])
            if not matching:
                for key, indices in index.items():
                    if normalized in key or key in normalized:
                        matching = indices
                        break
            return list(matching)

        modality_matches = _resolve_matches(self._modality_index, modality or "")
        if modality_matches:
            _log.debug("Metadata filter modality='%s': %d matches", modality, len(modality_matches))
            filter_sets.append((f"modality({len(modality_matches)})", set(modality_matches), True))

        body_region_matches = _resolve_matches(self._body_region_index, body_region or "")
        if body_region_matches:
            _log.debug("Metadata filter body_region='%s': %d matches", body_region, len(body_region_matches))
            filter_sets.append((f"body_region({len(body_region_matches)})", set(body_region_matches), True))

        finding_matches = _resolve_matches(self._finding_index, finding_category or "")
        if finding_matches:
            _log.debug("Metadata filter finding='%s': %d matches", finding_category, len(finding_matches))
            filter_sets.append((f"finding({len(finding_matches)})", set(finding_matches), False))

        concept_matches = _resolve_matches(self._concept_type_index, concept_type or "")
        if concept_matches:
            _log.debug("Metadata filter concept_type='%s': %d matches", concept_type, len(concept_matches))
            filter_sets.append((f"concept_type({len(concept_matches)})", set(concept_matches), False))

        target_matches = _resolve_matches(self._target_entity_index, target_entity or "")
        if target_matches:
            _log.debug("Metadata filter target_entity='%s': %d matches", target_entity, len(target_matches))
            filter_sets.append((f"target_entity({len(target_matches)})", set(target_matches), False))

        style_matches = [idx for idx, ex in enumerate(self._train_examples) if (ex.get('answer_style', '') or '').lower() == (answer_style or '').lower()] if answer_style and answer_style != 'unknown' else []
        if style_matches:
            _log.debug("Metadata filter answer_style='%s': %d matches", answer_style, len(style_matches))
            filter_sets.append((f"answer_style({len(style_matches)})", set(style_matches), False))

        if not filter_sets:
            _log.debug("Metadata filter produced no specific filters; skipping filter")
            return None

        # Combine filters. Prefer hard image-grounded filters first; degrade if they collapse to zero.
        hard_sets = [s for _, s, is_hard in filter_sets if is_hard]
        soft_sets = [s for _, s, is_hard in filter_sets if not is_hard]
        filter_applied = [label for label, _, _ in filter_sets]

        candidates = None
        if hard_sets:
            candidates = set.intersection(*hard_sets) if len(hard_sets) > 1 else set(next(iter(hard_sets)))
        if candidates is None or not candidates:
            # Fall back to the largest soft filter to avoid empty retrieval.
            if soft_sets:
                soft_sets_sorted = sorted(soft_sets, key=len, reverse=True)
                candidates = set(soft_sets_sorted[0])
                for s in soft_sets_sorted[1:]:
                    intersected = candidates.intersection(s)
                    if intersected:
                        candidates = intersected
            elif hard_sets:
                candidates = set(next(iter(hard_sets)))

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
                    'source': example.get('source', self.corpus_name),
                    'split': example.get('split', 'train'),
                    'modality': example['modality'],
                    'body_region': example['body_region'],
                    'finding_category': example.get('finding_category', 'other'),
                    'answer_style': example.get('answer_style', 'unknown'),
                    'laterality': example.get('laterality', 'unknown'),
                    'concept_type': example.get('concept_type', 'case'),
                    'concept': example.get('concept', ''),
                    'target_entity': example.get('target_entity', ''),
                    'question_family': example.get('question_family', ''),
                    'visual_cues': example.get('visual_cues', []),
                    'support_count': example.get('support_count', 1),
                    'text_chunk': example.get('dense_text', ''),
                    'source_question': example.get('question', ''),
                    'source_answer': example.get('answer', ''),
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
            (
                query,
                " ".join(part for part in [
                    c.metadata.get("text_chunk", ""),
                    c.metadata.get("concept_type", ""),
                    c.metadata.get("concept", ""),
                    c.metadata.get("target_entity", ""),
                    c.metadata.get("question_family", ""),
                    c.metadata.get("modality", ""),
                    c.metadata.get("body_region", ""),
                    c.metadata.get("finding_category", ""),
                    c.question,
                    c.answer,
                ] if part),
            )
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
            support_count = float(candidate.metadata.get("support_count", 1) or 1)
            support_bonus = min(0.25, np.log1p(support_count) / 10.0)
            concept_bonus = 0.08 if candidate.metadata.get("concept_type", "case") != "case" else 0.0
            candidate.score = float(scores[i] + support_bonus + concept_bonus)
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
    question_type: str = None,
    question_family: str = None,
    target_entity: str = None,
    answer_style: str = None,
    visual_cues: list = None,
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
        question_type=question_type,
        question_family=question_family,
        target_entity=target_entity,
        answer_style=answer_style,
        visual_cues=visual_cues,
    )
    
    return {
        "similar_cases": [
            {
                "question": r.question,
                "answer": r.answer,
                "score": r.score,
                "metadata": r.metadata,
                "text_chunk": r.metadata.get("text_chunk", ""),
                "concept_type": r.metadata.get("concept_type", "case"),
                "concept": r.metadata.get("concept", ""),
                "target_entity": r.metadata.get("target_entity", ""),
                "question_family": r.metadata.get("question_family", ""),
                "answer_style": r.metadata.get("answer_style", "unknown"),
            }
            for r in results
        ],
        "count": len(results)
    }
