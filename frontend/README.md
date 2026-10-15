# MedVisionQuery frontend

React 19 + Vite UI for MedVisionQuery. It talks to the FastAPI backend at `VITE_API_BASE`
(default `http://localhost:8000`, see `src/config.js`). See the [root README](../README.md) for the system itself.

## Run

```bash
npm install
npm run dev      # http://localhost:5173 (start the backend first)
npm run lint
npm run build
```

## Views

- **Dashboard** — upload an image and ask a question; shows the answer, explanation, verification and the agent trace.
- **Case Explorer** — the evaluation questions (`experiments/eval_set.json`) with the single prompt, workflow and agentic loop side by side (`CaseExplorer.jsx`, `AgentTrace.jsx`,
  `VerificationPanel.jsx`, `FailureModeDrawer.jsx`).
- **Evaluation** (`EvaluationView.jsx`) — reads `/api/dataset/evaluation`: strict and LLM-judged accuracy, mean and
  median time per case, accuracy by question type and by controller path, disagreement, and a per-case table.
  Data hooks live in `src/hooks/`.

## Screenshots

`scripts/capture-screenshots.mjs` drives a local Chrome/Edge with `puppeteer-core` and saves the Evaluation view
and the Case Explorer traces of the current run's showcase cases (its `final_metrics.json`). The easiest way to run
it is `python scripts/run_all.py --stages screenshots` from the repo root, which starts and stops both servers.

```bash
node scripts/capture-screenshots.mjs                 # backend and npm run dev already running
# options: --cases 1900,2102 (cached cases only), --out <dir>, --url, --browser <path to chrome.exe>
```
