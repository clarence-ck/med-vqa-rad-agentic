# ClariScan — Agentic Multimodal Medical VQA System

ClariScan is an **agentic multimodal system** for medical visual question answering (VQA) over radiology images. It decomposes the task into 5 specialized agent stages — image interpretation, question understanding, knowledge support, reasoning, and verification — orchestrated by a **Haystack Agent** with LLM-driven tool selection. Results are exposed through a demo UI with **Dashboard**, **Case Explorer**, and **Evaluation** views.

> ⚠️ **For educational and research purposes only. Not for clinical use.**

---

## Agentic System Design

This repository implements an **agentic multimodal system**:

| Property | Implementation |
|---|---|
| Problem decomposition | 5 distinct agent stages with explicit roles |
| Agent architecture | Named agents wrapped as Haystack Tools, LLM controller decides execution |
| Multimodal grounding | Image (base64 vision) + text at every stage |
| Dynamic tool selection | `retrieve_knowledge` skipped for simple questions by the controller |
| Control & failure handling | Verification, guardrails, disclaimers, review flags, automatic retry |
| Evaluation | Against two baselines with token F1, exact match, and disagreement analysis |

---

## Problem Definition

**Task:** medical visual question answering with explanation

| Input | Type |
|---|---|
| Medical image | Radiology / pathology image (JPEG / PNG) |
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
    UI[ClariScan React UI]
    DASH[Dashboard]
    EXP[Case Explorer]
    EVAL[Evaluation]

    UI --> DASH
    UI --> EXP
    UI --> EVAL

    DASH --> API[FastAPI Backend]
    EXP --> API
    EVAL --> API

    API --> ORCH[Haystack Agent Controller\nopenai chat generator]

    ORCH -->|tool call| T1[Tool: interpret_image\nImageInterpreter]
    ORCH -->|tool call| T2[Tool: analyze_question\nQuestionAnalyzer]
    ORCH -->|tool call| T3[Tool: retrieve_knowledge\nKnowledgeRetriever]
    ORCH -->|tool call| T4[Tool: reason_answer\nReasoningAgent]
    ORCH -->|tool call| T5[Tool: verify_answer\nVerificationAgent]

    T1 -->|outputs_to_state| STATE[(Shared Agent State\n14 typed fields)]
    T2 -->|outputs_to_state| STATE
    T3 -->|outputs_to_state| STATE
    T4 -->|outputs_to_state| STATE
    T5 -->|outputs_to_state| STATE

    STATE -->|inputs_from_state| T2
    STATE -->|inputs_from_state| T3
    STATE -->|inputs_from_state| T4
    STATE -->|inputs_from_state| T5

    API --> B1[Baseline 1\nSingle-shot Multimodal Prompt]
    API --> B2[Baseline 2\nHand-designed Workflow]

    T5 --> CACHE[Case Analysis Cache\n.case_explorer_cache/]
    B1 --> CACHE
    B2 --> CACHE
    CACHE --> EVAL

    ORCH --> OPENAI[OpenAI Chat Model]
    T1 --> OPENAI
    T2 --> OPENAI
    T3 --> OPENAI
    T4 --> OPENAI
    T5 --> OPENAI
    B1 --> OPENAI
```

### Agent Roles

| Component | File | Role | LLM calls |
|---|---|---|---|
| `ImageInterpreter` | `image_interpreter.py` | Perception — multimodal, extracts visual findings | 1 (vision) |
| `QuestionAnalyzer` | `question_analyzer.py` | Task understanding — classifies question type and intent | 1 (text) |
| `KnowledgeRetriever` | `knowledge_retriever.py` | Context — retrieves medical background; **dynamically skipped** for simple questions | 0–1 (text) |
| `ReasoningAgent` | `reasoning_agent.py` | Synthesis — generates answer and explanation from all prior context | 1 (text) |
| `VerificationAgent` | `verification_agent.py` | Evaluator — runs 5 structured checks on the answer | 1 (text) |
| Agent Controller | `orchestrator.py` | Haystack `Agent` — routes tool calls via LLM, manages shared state | 1+ (text) |
| Workflow Baseline | `workflow_baseline.py` | Deterministic rule-based answer logic using image + question analysis only | 0 (rules only) |
| Single-shot Baseline | `baseline.py` | Direct image+question prompt with no decomposition | 1 (vision) |

### Tool-calling and Shared State

The orchestrator uses Haystack's `Agent` + `Tool` architecture:

- Each agent is wrapped as a `Tool` with `inputs_from_state` / `outputs_to_state` bindings
- The controller LLM decides which tools to invoke based on prior outputs in state
- 14 state fields: `image_bytes`, `mime_type`, `question`, `image_interpretation`, `question_analysis`, `knowledge`, `answer_data`, `verification_result`, 5 duration fields, `retry_context`
- Exit condition: `verify_answer` finalizes the session
- A fresh `Agent` is created per request to prevent shared mutable state under concurrency

### Dynamic Tool Selection

The controller uses the `question_type` from `QuestionAnalyzer` to decide whether to call `KnowledgeRetriever`:

| Question type | `retrieve_knowledge` called? |
|---|---|
| `modality_identification`, `anatomy_identification`, `presence_absence`, `location`, `counting`, `yes_no` | No — skipped |
| `abnormality_description`, `explanation_request` | Yes |
| When `requires_medical_knowledge` is true | Yes |

### Model and Tool Choices

- **OpenAI chat model** (`gpt-5.4-mini-2026-03-17` by default)
  - Used for all 5 agent tools, the agent controller, and Baseline 1
  - All agents use `OpenAIChatGenerator` from Haystack via `Secret.from_token()`
  - Baseline 1 uses the `openai` Python client directly
  - Supports structured JSON-only prompting via `Return ONLY the JSON object` constraints
  - `max_completion_tokens=3000` for all LLM calls

- **Haystack `Agent` + `Tool`**
  - Controller LLM decides which tools to call based on question type
  - `inputs_from_state` / `outputs_to_state` replaces explicit data-wiring in a fixed pipeline
  - Per-tool timing tracked via `time.time()` within each tool function

- **FastAPI** — lightweight async backend, per-request agent instantiation

- **React 19 + Vite 8** — frontend UI, no CSS framework (custom CSS), no external UI library

---

## Multimodal Grounding

### Modalities

- **Image** — medical radiology / pathology image, converted to base64 PNG, passed as `image_url` content block
- **Text** — the user's natural language question

### How modalities interact across stages

| Stage | Image used? | Text used? |
|---|---|---|
| Image Interpretation | ✅ Base64 vision prompt | ✅ Extraction instructions |
| Question Understanding | ❌ | ✅ Question + visual summary from stage 1 |
| Knowledge Support | ❌ | ✅ body_region + modality + visual_cues from stages 1–2 |
| Reasoning & Answer | ❌ | ✅ All prior structured outputs |
| Verification | ❌ | ✅ Answer + image analysis + question analysis |

The image is only directly consumed in stage 1; all downstream stages consume the **structured visual summary** extracted from it. This is the core multimodal grounding pattern.

---

## Agent Outputs (JSON Contracts)

### ImageInterpreter output
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

### QuestionAnalyzer output
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

### KnowledgeRetriever output
```json
{
  "contextual_notes": "...",
  "key_facts": ["..."],
  "expected_normal_findings": "...",
  "common_pathologies": ["..."],
  "terminology": {"term": "definition"}
}
```

### ReasoningAgent output
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

### VerificationAgent output — 5 explicit checks
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
| Status | Meaning |
|---|---|
| `verified` | All 5 checks passed |
| `partially_verified` | Most checks passed with minor uncertainty |
| `low_confidence` | Insufficient evidence for a confident answer |
| `unsupported` | Answer is not grounded in visible evidence |

### Guardrail logic (`orchestrator.py`)
- `low_confidence` or `unsupported` → answer replaced with guarded answer + disclaimer
- `flag_for_review` or severe failure categories → `should_flag_for_review = True`
- Disclaimer auto-inserted when none is provided by the verifier

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
- **Cross-modal agreement** — derived from `explanation_grounded_in_evidence` and `no_contradictions` checks

### Automatic retry
- `run_pipeline()` checks the first result with `_should_retry_pipeline()`
- Retry triggers on: `low_confidence`, `unsupported`, `should_flag_for_review`, or any high-severity failure category
- Second agent run receives `retry_context` in state (verification status, confidence, failure categories)
- `ReasoningAgent` uses `retry_context` to self-correct with extra caution
- Best result selected by `_payload_score()` scoring: verification rank → confidence rank → severity count → review flag

---

## Evaluation Design

### Metrics computed per case
- **Exact match** — normalized answer string comparison
- **Yes/no normalization** — `yes/true/present/positive` → `yes`, `no/false/absent/negative/normal` → `no`
- **Token F1** — unigram overlap between predicted and ground-truth answer
- **Winner** — `baseline` / `workflow` / `agentic` / `tie_multi_correct` / `tie_all_wrong`
- **Disagreement** — whether any two systems produced different answers

### Aggregate evaluation summary
- Overall accuracy per system (baseline / workflow / agentic)
- Average token F1 per system
- Win counts and disagreement rate
- Average duration (ms)
- Split counts (`train` / `test`)
- Per-question-type accuracy breakdown

### Cache and evaluation workflow
1. Opening a sample in Case Explorer runs all 3 systems concurrently (`asyncio.gather`) and caches results to `backend/.case_explorer_cache/sample_{id}_{question_hash}.json`
2. Subsequent opens use the cache (use `?refresh=true` to force re-run)
3. The Evaluation view reads all cache files and computes the aggregate summary live

### Systems compared
| System | File | Approach |
|---|---|---|
| Single-shot baseline | `baseline.py` | One multimodal prompt, OpenAI client directly, returns answer + explanation + confidence |
| Workflow baseline | `workflow_baseline.py` | Rule-based: deterministic answer logic per question type using LLM image + question outputs; no synthesis LLM |
| Agentic pipeline | `orchestrator.py` | 5-tool Haystack Agent with LLM controller, verification, guardrails, retry |

### Train / test split
Split is stored in `vqa_rad.json` (written by the download script) and falls back to filename prefix inference:
- `train_XXXX.png` → `train`
- `test_XXXX.png` → `test`

#### Intended usage
| Split | Purpose |
|---|---|
| `train` | Prompt iteration, debugging, Case Explorer investigation |
| `test` | Final benchmark reporting, trustworthy evaluation |

---

## API Reference

All endpoints served at `http://localhost:8000`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check, lists all endpoints |
| `POST` | `/api/analyze` | Run full agentic pipeline |
| `POST` | `/api/analyze/baseline` | Run single-shot baseline |
| `POST` | `/api/analyze/workflow` | Run workflow baseline |
| `GET` | `/api/dataset/samples?limit=N&split=train\|test` | List dataset samples |
| `GET` | `/api/dataset/sample/{id}` | Get sample with base64 image |
| `GET` | `/api/dataset/sample/{id}/analysis?refresh=false` | Run all 3 systems, cache result |
| `GET` | `/api/dataset/evaluation?split=train\|test` | Aggregate evaluation from cache |

`POST /api/analyze` accepts `multipart/form-data` with fields `image` (file) and `question` (string).

---

## UI Demo

### Dashboard
- Upload an image or select a dataset sample
- Submit a question
- All 3 systems run in parallel (separate loading states per system)
- Toggle between Agentic / Workflow / Baseline views via `ComparisonToggle`
- Components: `AnswerCard`, `ExplanationCard`, `AgentTrace`, `VerificationPanel`, `FailureModeDrawer`, `ConfidenceBadge`, `ImageViewer`

### Case Explorer
- Browse all dataset samples, filter by `train` / `test` split
- Select a sample to load it (fetches image from backend, pre-fills question)
- Run full 3-system analysis with cache awareness (`refresh` toggle)
- Inspect ground truth, all answers, agent trace, verification, failure modes side-by-side

### Evaluation
- Reads aggregate metrics from all cached analyses
- Displays accuracy, token F1, win counts, disagreement rate
- Question-type breakdown table
- Split-aware filtering (`train` / `test`)

---

## Quick Start

### Prerequisites

- Python 3.12
- Node.js 18+
- OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/Medical-VQA-Explanation-Agent.git
cd Medical-VQA-Explanation-Agent
cp .env.example .env
```

Edit `.env` and set:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini          # or any vision-capable OpenAI model
BACKEND_CORS_ORIGINS=http://localhost:5173
```

### 2. Install Python dependencies

**Option A — pip**
```bash
cd backend
pip install -r requirements.txt
```

**Option B — conda (recommended)**
```bash
conda env create -f environment.yml
conda activate clariscan
```

### 3. Download VQA-RAD dataset

```bash
python scripts/download_dataset.py
```

This downloads `flaviagiammarino/vqa-rad` from Hugging Face, saves images to `data/vqa-rad/images/`, and writes `data/vqa-rad/vqa_rad.json`. Takes ~1–2 minutes on a fast connection.

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
├── .env.example                     # environment variable template
├── environment.yml                  # conda environment (Python 3.12)
├── backend/
│   ├── main.py                      # FastAPI app, CORS, router registration
│   ├── config.py                    # loads .env (OPENAI_API_KEY, OPENAI_MODEL, DATA_DIR, CORS)
│   ├── requirements.txt             # pip dependencies
│   ├── agents/
│   │   ├── orchestrator.py          # Haystack Agent controller, 5 tools, retry, guardrails
│   │   ├── image_interpreter.py     # Agent 1: multimodal vision → structured findings
│   │   ├── question_analyzer.py     # Agent 2: question classification and intent
│   │   ├── knowledge_retriever.py   # Agent 3: medical background retrieval (conditional)
│   │   ├── reasoning_agent.py       # Agent 4: answer synthesis with retry context support
│   │   ├── verification_agent.py    # Agent 5: 5-check structured verification
│   │   ├── baseline.py              # Single-shot multimodal baseline
│   │   └── workflow_baseline.py     # Rule-based workflow baseline
│   ├── routes/
│   │   ├── analyze.py               # POST /api/analyze, /baseline, /workflow
│   │   └── dataset.py               # GET /api/dataset/* + evaluation + cache logic
│   └── utils/
│       ├── data_loader.py           # VQA-RAD loader, split inference, question classification
│       └── json_utils.py            # strip_json_fences() — removes markdown fences from LLM output
├── frontend/
│   ├── package.json                 # React 19, Vite 8
│   └── src/
│       ├── App.jsx                  # root component, tab routing (Dashboard / Explorer / Evaluation)
│       ├── config.js                # API_BASE URL
│       ├── index.css                # custom CSS (no UI framework)
│       ├── components/
│       │   ├── AgentTrace.jsx       # per-stage timing and output display
│       │   ├── AnswerCard.jsx       # final answer + guarded answer
│       │   ├── CaseExplorer.jsx     # dataset browser with analysis trigger
│       │   ├── CaseHeader.jsx       # case metadata display
│       │   ├── CaseInput.jsx        # image upload + sample select + question input
│       │   ├── ComparisonToggle.jsx # switch between agentic / workflow / baseline views
│       │   ├── ConfidenceBadge.jsx  # High/Medium/Low badge
│       │   ├── EvaluationView.jsx   # aggregate metrics + question-type table
│       │   ├── ExplanationCard.jsx  # answer explanation + evidence used
│       │   ├── FailureModeDrawer.jsx# failure category list with severity
│       │   ├── ImageViewer.jsx      # medical image display
│       │   └── VerificationPanel.jsx# checks, status, disclaimer, review flag
│       └── hooks/
│           ├── useAnalysis.js       # analyze / analyzeBaseline / analyzeWorkflow calls
│           ├── useDataset.js        # sample listing + fetchSampleAnalysis
│           └── useEvaluation.js     # evaluation summary fetch
├── scripts/
│   └── download_dataset.py          # downloads VQA-RAD from Hugging Face
└── data/
    └── vqa-rad/                     # gitignored — created by download script
        ├── vqa_rad.json
        └── images/

---

## Reflection

### When agentic behavior helps

- The task requires both low-level perception and high-level reasoning
- Failures need to be localized to a specific stage (image extraction vs. reasoning vs. verification)
- Uncertainty handling and output moderation matter (medical domain)
- Intermediate outputs have independent value (the visual summary, question classification, knowledge notes)

### Limitations

- 4–6 serial LLM calls per request — slower and more expensive than single-shot inference
- Knowledge support is LLM-synthesized, not retrieved from a real knowledge base
- Token F1 evaluation is brittle for open-ended answers; exact match misses near-correct responses
- Self-consistency signal is lightweight (single verifier pass) rather than multi-sample voting
- The system is research-oriented and not suitable for clinical deployment

---

## Dataset

**VQA-RAD** — a publicly available radiology VQA benchmark:

| Field | Value |
|---|---|
| Source | `flaviagiammarino/vqa-rad` (Hugging Face) |
| Images | 315 radiology images |
| QA pairs | 2,248 question-answer pairs |
| Splits | `train` and `test` |
| Image types | X-ray, CT, MRI |
| Question types | yes/no, open-ended |

Images are stored as PNGs named `{split}_{index:04d}.png` by the download script.

---

## License

MIT
