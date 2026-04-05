# MedVisionQuery — Agentic Multimodal Medical VQA System

MedVisionQuery is an **agentic multimodal system** for medical visual question answering (VQA) over radiology images. It decomposes the task into 6 specialized agent stages — image interpretation, question understanding, difficulty routing, knowledge support, reasoning, and verification — orchestrated by a **Haystack Agent** with LLM-driven tool selection. The knowledge stage now uses an **Advanced RAG pipeline** (query rewriting, hybrid retrieval, reranking, and self-reflection). Results are exposed through a demo UI with **Dashboard**, **Case Explorer**, and **Evaluation** views.

> ⚠️ **For educational and research purposes only. Not for clinical use.**

---

## Agentic System Design

| Property | Implementation |
|---|---|
| Problem decomposition | 6 distinct agent stages with explicit roles |
| Agent architecture | Named agents wrapped as Haystack Tools; the controller decides tool use over shared state |
| Multimodal grounding | The image is interpreted once, then downstream stages consume structured visual state; OpenAI-backed generators drive the agentic stages while Advanced RAG uses local retrieval models |
| Dynamic tool selection | `retrieve_knowledge` is optional and skipped when the Difficulty Router sets `needs_retrieval=false` |
| Control & failure handling | Verification, guardrails, disclaimers, review flags, failure categories, and automatic retry |
| Evaluation | Against two baselines with normalized answer-match accuracy, token F1, close/open-ended accuracy, and disagreement analysis |

---

## Problem Definition

**Task:** medical visual question answering with explanation

| Input | Type |
|---|---|
| Medical image | User-uploaded medical image (JPEG / PNG); evaluated on VQA-RAD radiology cases |
| Natural language question | Any question about the image |

**Output:** structured JSON with answer, explanation, evidence, verification status, confidence, failure signals, and per-stage agent trace.

The task is decomposed rather than single-prompted because the system must separately solve:

1. **Perception** — what is in the image?
2. **Question interpretation** — what is being asked?
3. **Medical context** — what background knowledge applies?
4. **Synthesis** — what is the best-grounded answer?
5. **Verification** — is the answer actually supported?

---

## Agent Architecture

### System Diagram

```mermaid
flowchart LR
    UI[MedVisionQuery React UI]
    DASH[Dashboard]
    EXP[Case Explorer]
    EVAL[Evaluation]

    UI --> DASH
    UI --> EXP
    UI --> EVAL

    DASH --> API[FastAPI Backend]
    EXP --> API
    EVAL --> API

    API --> ORCH[Haystack Agent Controller\nOpenAI chat model]

    ORCH -->|tool call| T1[Tool: interpret_image\nImageInterpreter]
    ORCH -->|tool call| T2[Tool: analyze_question\nQuestionAnalyzer]
    ORCH -->|tool call| T3[Tool: route_difficulty\nDifficultyRouter]
    ORCH -->|tool call| T4[Tool: retrieve_knowledge\nKnowledgeRetriever]
    ORCH -->|tool call| T5[Tool: reason_answer\nReasoningAgent]
    ORCH -->|tool call| T6[Tool: verify_answer\nVerificationAgent]

    T1 -->|auto-block| ERR[Error\nnon-radiology image]

    T1 -->|outputs_to_state| STATE[(Shared Agent State\n16 typed fields)]
    T2 -->|outputs_to_state| STATE
    T3 -->|outputs_to_state| STATE
    T4 -->|outputs_to_state| STATE
    T5 -->|outputs_to_state| STATE
    T6 -->|outputs_to_state| STATE

    STATE -->|inputs_from_state| T2
    STATE -->|inputs_from_state| T3
    STATE -->|inputs_from_state| T4
    STATE -->|inputs_from_state| T5
    STATE -->|inputs_from_state| T6

    T6 --> GUARD[Guardrail\nverification + failures]
    GUARD --> VER[Verified]
    GUARD --> PART[Partially Verified]
    GUARD --> LOW[Low Confidence]
    GUARD --> REV[Review]
    GUARD --> ERR

    API --> B1[Baseline 1\nSingle-shot Multimodal Prompt]
    API --> B2[Baseline 2\nHand-designed Workflow]

    T4 --> RAG[Advanced RAG Engine\nquery rewrite + FAISS/BM25 + rerank + reflection]
    RAG --> TRAIN[(VQA-RAD train split\nretrieval corpus)]

    EXP --> DATA[Dataset Analysis\n/api/dataset/sample/sample_id/analysis]
    DATA --> CACHE[Case Analysis Cache\n.case_explorer_cache/]
    CACHE --> EVAL

    T1 --> LLM[OpenAI API\nOPENAI_MODEL]
    T2 --> LLM
    T3 --> LLM
    T4 --> LLM
    T5 --> LLM
    ORCH --> LLM
    B1 --> LLM
```

![System Diagram](system_diagram.png)

### Agent Roles

| Component | File | Role | Model |
|---|---|---|---|
| `ImageInterpreter` | `image_interpreter.py` | Agent 1: Perception — multimodal image interpretation with auto-block for non-radiology images | OpenAI |
| `QuestionAnalyzer` | `question_analyzer.py` | Agent 2: Task understanding — question classification and intent analysis | OpenAI |
| `DifficultyRouter` | `difficulty_router.py` | Agent 3: Planning — sets routing metadata in shared state | OpenAI |
| `KnowledgeRetriever` | `knowledge_retriever.py` | Agent 4: Context — optional Advanced RAG retrieval over VQA-RAD train split + LLM synthesis | OpenAI + local retrieval stack |
| `ReasoningAgent` | `reasoning_agent.py` | Agent 5: Synthesis — generates answer and explanation from all prior context | OpenAI |
| `VerificationAgent` | `verification_agent.py` | Agent 6: Evaluator — grounded consistency / safety checks with 5 structured checks | OpenAI |
| Agent Controller | `orchestrator.py` | Haystack `Agent` — routes tool calls, manages shared state, guardrails, retry | OpenAI |
| Workflow Baseline | `workflow_baseline.py` | Deterministic answer derivation over image/question analysis outputs | Rules + OpenAI intermediates |
| Single-shot Baseline | `baseline.py` | Direct image+question prompt with no decomposition | OpenAI |

### Tool-calling and Shared State

The orchestrator uses Haystack's `Agent` + `Tool` architecture:

- **6 agents** wrapped as `Tool` with `inputs_from_state` / `outputs_to_state` bindings:
  1. `interpret_image` — ImageInterpreter (multimodal)
  2. `analyze_question` — QuestionAnalyzer
  3. `route_difficulty` — DifficultyRouter (sets routing metadata in shared state)
  4. `retrieve_knowledge` — KnowledgeRetriever (conditional, based on `routing_decision.needs_retrieval`)
  5. `reason_answer` — ReasoningAgent
  6. `verify_answer` — VerificationAgent
- The controller LLM decides which tools to invoke and in what order based on prior outputs in state
- **Routing metadata**: DifficultyRouter stores `verification_policy` + `fallback_policy` in `routing_decision`; downstream payloads and UI surfaces read these fields from shared state
- **Auto-block**: ImageInterpreter validates radiology images; non-radiology images abort pipeline early → Error state
- 16 state fields: `image_bytes`, `mime_type`, `question`, `image_interpretation`, `question_analysis`, `routing_decision`, `knowledge`, `answer_data`, `verification_result`, 5 duration fields, `retry_context`
- Exit condition: `verify_answer` finalizes the session
- A fresh `Agent` is created per request to prevent shared mutable state under concurrency

### Dynamic Tool Selection

The controller follows `routing_decision.needs_retrieval` to decide whether to invoke `KnowledgeRetriever`:

| Question type | `retrieve_knowledge` called? |
|---|---|
| `modality_identification`, `anatomy_identification`, `presence_absence`, `location`, `counting`, `yes_no` | Usually skipped |
| `abnormality_description`, `explanation_request` | Usually retrieved |
| When `requires_medical_knowledge` is true | Usually retrieved |

### Knowledge Retrieval (Advanced RAG)

`KnowledgeRetriever` integrates `advanced_rag.py` to ground context in retrieved examples before synthesis.

1. Rewrite retrieval query from the original question (+ modality/body-region context)
2. Apply metadata pre-filtering (`modality`, `body_region`)
3. Run hybrid retrieval: dense vectors (FAISS) + sparse BM25
4. Fuse candidates via reciprocal rank fusion (RRF)
5. Rerank with cross-encoder and keep top-k similar cases
6. Run self-reflection on retrieval quality
7. Inject retrieved cases into the knowledge prompt

Retrieval corpus is built from the **train split** of `data/vqa-rad/vqa_rad.json`.

### Current runtime model configuration

The current runtime uses OpenAI for all LLM-driven stages; the Advanced RAG retrieval stack additionally uses local SentenceTransformers, CrossEncoder, FAISS, and BM25 models.

| Env vars | Used by |
|---|---|
| `OPENAI_API_KEY` | All agents, Haystack controller, `baseline.py` |
| `VISION_MODEL` | `ImageInterpreter`, `baseline.py` |
| `OPENAI_MODEL` | `QuestionAnalyzer`, `KnowledgeRetriever`, `ReasoningAgent`, `VerificationAgent`, Haystack controller |

Hugging Face is used to download the raw dataset via scripts. The retrieval stack itself is local, while the agentic generation steps use OpenAI.

### Other Technology Choices

- **Haystack `Agent` + `Tool`** — controller-managed tool calls with `inputs_from_state` / `outputs_to_state` bindings and per-tool timing
- **FastAPI** — async backend; blocking LLM calls run in `asyncio.to_thread()` to avoid event-loop blocking
- **React 19 + Vite** — frontend UI with Dashboard, Case Explorer, and Evaluation views
- **FAISS (`faiss-cpu`)** — dense vector index for fast semantic retrieval in Advanced RAG
- **SentenceTransformers + CrossEncoder** — dense embeddings and cross-encoder reranking
- **BM25 (`rank-bm25`)** — sparse lexical retrieval for hybrid search
- **Hugging Face `datasets`** — used only by `scripts/download_dataset.py` to materialize the local VQA-RAD data folder
- **Matplotlib** — `scripts/system_diagram.py` regenerates `system_diagram.png`

---

## Multimodal Grounding

- **Image** — converted to base64 PNG, passed as `image_url` content block to the vision model
- **Text** — user's natural language question

### How modalities interact across stages

| Stage | Image used? | Text used? |
|---|---|---|
| Image Interpretation | Base64 vision prompt | Extraction instructions |
| Question Understanding | No | Question + visual summary from stage 1 |
| Knowledge Support | No | `body_region` + `modality` + `visual_cues` from stages 1–2 |
| Reasoning & Answer | No | All prior structured outputs |
| Verification | No | Answer + image analysis + question analysis |

The image is consumed only once (stage 1); all downstream stages operate on the **structured visual summary** extracted from it.

---

## Agent Outputs (JSON Contracts)

### ImageInterpreter
```json
{
  "modality": "X-ray",
  "body_region": "chest",
  "visible_structures": ["lungs", "ribs", "heart"],
  "visual_summary": "...",
  "image_quality": "good | fair | poor",
  "notable_findings": ["..."]
}
```

### QuestionAnalyzer
```json
{
  "question_type": "anatomy_identification | modality_identification | presence_absence | location | counting | abnormality_description | explanation_request | yes_no | other",
  "target_entity": "...",
  "reasoning_type": "visual_recognition | spatial_reasoning | medical_knowledge | comparison | descriptive",
  "expected_answer_format": "single_word | short_phrase | yes_no | sentence | paragraph",
  "requires_medical_knowledge": true,
  "question_clarity": "clear | ambiguous | unclear",
  "rephrased_question": "..."
}
```

### KnowledgeRetriever
```json
{
  "contextual_notes": "...",
  "key_facts": ["..."],
  "expected_normal_findings": "...",
  "common_pathologies": ["..."],
  "terminology": {"term": "definition"}
}
```

### ReasoningAgent
```json
{
  "answer": "...",
  "explanation": "...",
  "reasoning_trace": "...",
  "evidence_used": ["..."],
  "answer_confidence": "high | medium | low",
  "limitations": "..."
}
```

### VerificationAgent — 5 explicit checks
```json
{
  "verification_status": "verified | partially_verified | low_confidence | unsupported",
  "confidence_label": "High | Medium | Low",
  "checks": {
    "answer_matches_question_type": true,
    "explanation_grounded_in_evidence": true,
    "answer_strength_appropriate": true,
    "no_contradictions": true,
    "format_appropriate": true
  },
  "issues_found": ["..."],
  "fallback_recommendation": "none | rephrase_answer | add_disclaimer | flag_for_review",
  "verification_notes": "...",
  "suggested_disclaimer": "..."
}
```

---

## Control and Failure Handling

### Verification status levels
| Status | Meaning | Final State |
|---|---|---|
| `verified` | All 5 checks passed | **Verified** — answer shown directly |
| `partially_verified` | Most checks passed with minor uncertainty | **Partially Verified** — some uncertainty noted |
| `low_confidence` | Insufficient evidence for a confident answer | **Low Confidence** — guarded answer + disclaimer |
| `unsupported` | Answer is not grounded in visible evidence | **Low Confidence** — guarded answer + disclaimer |
| `rejected` (image abort) | Non-radiology image detected | **Error** — invalid image type |

### Guardrail logic
The Guardrail combines `verification_status` + `failure_categories` to produce 5 final answer states:

1. **Verified** (`verification_status == "verified"`) — answer shown directly
2. **Partially Verified** (`verification_status == "partially_verified"`) — answer with minor uncertainty
3. **Low Confidence** (`verification_status in {"low_confidence", "unsupported"}`) — guarded answer + disclaimer
4. **Review** (`should_flag_for_review == true` or high-severity failures) — review recommended
5. **Error** (ImageInterpreter auto-block) — non-radiology image rejected

- Disclaimer auto-inserted when verifier does not provide one
- `flag_for_review` triggered by high-severity failure categories

### Failure categories (automatically detected)
| Code | Severity |
|---|---|
| `knowledge_retrieval_failed` | medium |
| `reasoning_generation_unstable` | high |
| `unsupported_claim` | high |
| `low_confidence` | high |
| `partially_verified` | medium |
| `visual_contradiction` | high |
| `question_answer_mismatch` | medium |
| `parser_failure` | high |

### Analysis signals
- **Self-consistency signal** — ratio of passed verification checks (`high` / `medium` / `low`)
- **Cross-modal agreement** — derived from `explanation_grounded_in_evidence` and `no_contradictions`

### Automatic retry
- `run_pipeline()` evaluates the first result with `_should_retry_pipeline()`
- Retry triggers on: `low_confidence`, `unsupported`, `should_flag_for_review`, or any high-severity failure
- Second run receives `retry_context` in state (status, confidence, failure categories) so `ReasoningAgent` can self-correct
- Best result selected by `_payload_score()`: verification rank → confidence rank → severity count → review flag

---

## Evaluation Design

### Metrics computed per cached case
- **Normalized answer match** — evaluation method depends on ground truth answer style:
  - **Close-ended** (binary ground truth like yes/no): synonym matching only
  - **Open-ended** (non-binary ground truth): Token F1 ≥ 0.7 only
- **Yes/no normalization** — `yes/true/present/positive/abnormal/found/visible/seen` → `yes`; `no/false/absent/negative/normal/not found/not visible/not seen` → `no`
- **Token F1** — unigram overlap between predicted and ground-truth answer
- **Winner** — `baseline` / `workflow` / `agentic` / `tie: ...` / `tie_all_wrong`
- **Disagreement** — whether any two systems produced different answers

### Aggregate evaluation summary
- Overall normalized answer-match accuracy per system (baseline / workflow / agentic)
- Close-ended and open-ended accuracy per system
- Win counts, disagreement rate, and average duration (ms)
- Per-question-type accuracy breakdown
- Recent-case rows including verification status, self-consistency, and cross-modal agreement

### Cache and evaluation workflow
1. Selecting a sample in Case Explorer runs all 3 systems concurrently via `asyncio.gather` and writes results to `backend/.case_explorer_cache/sample_{id}_{question_hash}.json`
2. Subsequent opens serve from cache — use the **Refresh** button to force a re-run
3. The Evaluation tab reads all cache files and computes the aggregate summary live

### Systems compared
| System | File | Approach |
|---|---|---|
| Single-shot baseline | `baseline.py` | One multimodal prompt, no decomposition — answer + explanation + confidence |
| Workflow baseline | `workflow_baseline.py` | Deterministic answer derivation over image/question analysis outputs |
| Agentic pipeline | `orchestrator.py` | 6-tool Haystack Agent with LLM controller, verification, guardrails, retry |

---

## API Reference

All endpoints served at `http://localhost:8000`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check, lists all endpoints |
| `POST` | `/api/analyze` | Run full agentic pipeline |
| `POST` | `/api/analyze/baseline` | Run single-shot baseline |
| `POST` | `/api/analyze/workflow` | Run workflow baseline |
| `GET` | `/api/dataset/samples?limit=N` | List **test-split only** dataset samples (balanced across question-type categories) |
| `GET` | `/api/dataset/sample/{id}` | Get sample with base64-encoded image |
| `GET` | `/api/dataset/sample/{id}/analysis?refresh=false` | Run all 3 systems on a sample, cache result |
| `GET` | `/api/dataset/evaluation` | Aggregate evaluation metrics from all cached analyses |

`POST /api/analyze*` endpoints accept `multipart/form-data` with fields `image` (file) and `question` (string).

---

## UI

### Dashboard
Purpose: **ad-hoc analysis of any uploaded image**.

- Upload a medical image and type a question
- All 3 systems run in parallel; results appear as each completes
- Toggle between Agentic Pipeline / Workflow Baseline / Single-shot Baseline views
- Components: `CaseHeader`, `CaseInput`, `ImageViewer`, `ComparisonToggle`, `AnswerCard`, `ExplanationCard`, `AgentTrace`, `RoutingCard`, `VerificationPanel`, `FailureModeDrawer`, `ConfidenceBadge`

### Case Explorer
Purpose: **systematic browsing and analysis of the VQA-RAD dataset**.

- Browse a curated set of **test-split only** samples (default 30)
- Sample selector is grouped by question-type categories with deterministic ordering
- Select a sample to load its image, question, and ground-truth answer
- Run full 3-system analysis with cache awareness (Refresh to force re-run)
- Inspect ground truth, all 3 answers, agent trace, verification, and failure modes side-by-side

### Evaluation
Purpose: **aggregate performance comparison across all cached analyses**.

- Reads all cached Case Explorer results automatically
- Displays normalized answer-match accuracy, close/open-ended accuracy, win counts, disagreement rate, and average duration per system
- Shows a per-question-type breakdown and recent evaluated-case table with verification signals
- Hit **Refresh** to re-aggregate after running new cases in Case Explorer

---

## Quick Start

### Prerequisites

- Python 3.12
- Node.js 18+
- OpenAI API key with access to the selected `OPENAI_MODEL`

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/Medical-VQA-Explanation-Agent.git
cd Medical-VQA-Explanation-Agent
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-...
VISION_MODEL=gpt-5.4-mini-2026-03-17
OPENAI_MODEL=gpt-5.4-mini-2026-03-17
HF_TOKEN=hf_...

# Frontend origin allowed by CORS
BACKEND_CORS_ORIGINS=http://localhost:5173
```

Optional: create `frontend/.env` if the frontend should point to a non-default backend URL.
```env
VITE_API_BASE=http://localhost:8000
```

Note: The current runtime path requires `OPENAI_API_KEY`, `OPENAI_MODEL`, and `BACKEND_CORS_ORIGINS`. `HF_TOKEN` is optional but recommended to avoid Hugging Face Hub unauthenticated-rate-limit warnings and speed up model downloads.

### 2. Install Python dependencies

**Option A — conda (recommended)**
```bash
conda env create -f environment.yml
conda activate medvisionquery
```

This path installs native ML dependencies (`numpy`, `faiss-cpu`, `pytorch`) from conda first, then installs Python-level packages with pip. This is the recommended setup to avoid OpenMP runtime conflicts on Windows.

**Option B — pip (fallback, more fragile for native deps on Windows)**
```bash
pip install -r backend/requirements.txt
```

If you use the pip fallback, install native packages from conda first in the same environment:
```bash
conda install -c conda-forge numpy faiss-cpu pytorch
pip install -r backend/requirements.txt
```

### 3. Download VQA-RAD dataset

```bash
python scripts/download_dataset.py
```

Downloads `flaviagiammarino/vqa-rad` from Hugging Face, saves images to `data/vqa-rad/images/`, and writes metadata to `data/vqa-rad/vqa_rad.json`. Takes ~1–2 minutes.

### 4. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### 6. Open the app

Navigate to `http://localhost:5173`.

---

## Project Structure

```text
Medical-VQA-Explanation-Agent/
├── .env.example                     # env template (OpenAI credentials and CORS)
├── environment.yml                  # conda environment (Python 3.12 + pip dependencies)
├── backend/
│   ├── main.py                      # FastAPI app, CORS middleware, router registration
│   ├── config.py                    # loads .env: OPENAI_*, legacy TEXT_MODEL*, DATA_DIR, CORS
│   ├── requirements.txt             # pip dependencies
│   ├── agents/
│   │   ├── orchestrator.py          # Haystack Agent controller, tool routing, retry logic, guardrails
│   │   ├── image_interpreter.py     # Agent 1: multimodal image interpretation (OpenAI)
│   │   ├── question_analyzer.py     # Agent 2: question classification and intent analysis (OpenAI)
│   │   ├── advanced_rag.py          # Advanced RAG engine: query rewrite, FAISS+BM25 retrieval, reranking, reflection
│   │   ├── knowledge_retriever.py   # Agent 4: optional medical context retrieval with Advanced RAG grounding
│   │   ├── reasoning_agent.py       # Agent 5: answer synthesis with retry context support (OpenAI)
│   │   ├── verification_agent.py    # Agent 6: structured verification and fallback recommendations (OpenAI)
│   │   ├── baseline.py              # Single-shot multimodal baseline (OpenAI)
│   │   └── workflow_baseline.py     # Deterministic answer derivation over image/question analysis outputs
│   ├── routes/
│   │   ├── analyze.py               # POST /api/analyze, /baseline, /workflow
│   │   └── dataset.py               # GET /api/dataset/* — samples, analysis, evaluation, cache
│   └── utils/
│       ├── data_loader.py           # VQA-RAD JSON loader, sample bucketing, question classification
│       └── json_utils.py            # strip_json_fences() — removes markdown code fences from LLM output
├── frontend/
│   ├── package.json                 # React 19 + Vite app
│   └── src/
│       ├── App.jsx                  # root component, tab routing (Dashboard / Case Explorer / Evaluation)
│       ├── config.js                # API base URL (`VITE_API_BASE` override)
│       ├── index.css                # custom CSS (no framework)
│       ├── components/
│       │   ├── AgentTrace.jsx       # stage timeline, status badges, expandable outputs
│       │   ├── AnswerCard.jsx       # final answer, guarded answer, confidence, disclaimer
│       │   ├── CaseExplorer.jsx     # dataset browser and cached 3-system case comparison
│       │   ├── CaseHeader.jsx       # case metadata summary bar
│       │   ├── CaseInput.jsx        # image upload + question input (Dashboard)
│       │   ├── ComparisonToggle.jsx # switch between agentic / workflow / baseline result views
│       │   ├── ConfidenceBadge.jsx  # High / Medium / Low confidence badge
│       │   ├── EvaluationView.jsx   # aggregate metrics, close/open-ended accuracy, recent cases
│       │   ├── ExplanationCard.jsx  # explanation text, evidence used, reasoning trace
│       │   ├── FailureModeDrawer.jsx# collapsible failure category list and failure hints
│       │   ├── ImageViewer.jsx      # image display with zoom and predicted tags
│       │   └── VerificationPanel.jsx# verification checks, notes, fallback, review flag
│       └── hooks/
│           ├── useAnalysis.js       # analyze / analyzeBaseline / analyzeWorkflow API calls
│           ├── useDataset.js        # sample listing + fetchSampleAnalysis
│           └── useEvaluation.js     # evaluation summary fetch
├── scripts/
│   ├── download_dataset.py          # downloads VQA-RAD from Hugging Face, writes JSON + images
│   ├── system_diagram.py            # regenerates the architecture figure
├── system_diagram.png               # rendered system diagram asset
└── data/
    └── vqa-rad/                     # gitignored — created by download_dataset.py
        ├── vqa_rad.json             # flattened QA manifest with split labels and image filenames
        └── images/                  # PNG images named {split}_{index:04d}.png
```

---

## Dataset

**VQA-RAD** — a publicly available radiology VQA benchmark from [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) on Hugging Face.

| Field | Value |
|---|---|
| QA pairs in local manifest | 2,244 total |
| Split breakdown | 1,793 train / 451 test |
| Generated image files | 2,244 PNGs under `data/vqa-rad/images/` |
| Image types | Radiology images |
| Question styles | Yes/no (close-ended) and open-ended |

The current download script flattens the Hugging Face dataset into one local PNG per QA item and writes a flat `vqa_rad.json` with split labels. Local image filenames are saved as `{split}_{index:04d}.png`, so the number of generated image files matches the number of manifest rows.

---

## Reflection

### When agentic decomposition helps

- The task requires both low-level perception and high-level reasoning that are better separated
- Failures can be localized to a specific stage (image extraction vs. question understanding vs. verification)
- Uncertainty needs to be handled explicitly — the medical domain requires disclaimers and review flags
- Intermediate outputs have independent value (visual summary, question classification, knowledge notes)

### Limitations

- ~8–10 serial LLM calls per pass (one controller dispatch + one tool call per stage) — up to ~20 with automatic retry; significantly slower and more expensive than single-shot inference
- Retrieval is grounded to VQA-RAD train examples, but coverage is still limited to dataset distribution (not a comprehensive medical knowledge base)
- Token F1 thresholding is still brittle for open-ended answers; near-correct responses can fall below the cutoff
- Self-consistency signal is lightweight (single verifier pass) rather than multi-sample voting
- Research and demonstration use only — not suitable for clinical deployment

---

## License

MIT
