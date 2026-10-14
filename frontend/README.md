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
- **Case Explorer** — VQA-RAD test cases with all systems side by side (`CaseExplorer.jsx`, `AgentTrace.jsx`,
  `VerificationPanel.jsx`, `FailureModeDrawer.jsx`).
- **Evaluation** (`EvaluationView.jsx`) — reads `/api/dataset/evaluation`: strict and LLM-judged accuracy, mean and
  median time per case, accuracy by question type and by controller path, disagreement, and a per-case table.
  Data hooks live in `src/hooks/`.

## Screenshots

`scripts/capture-screenshots.mjs` drives a local Chrome/Edge with `puppeteer-core` and saves the Evaluation view
and selected Case Explorer traces. The backend and `npm run dev` must be running, and the cases must already be
cached by `scripts/run_case_explorer_cache.py`.

```bash
node scripts/capture-screenshots.mjs --out ../eval-dashboard-screenshots/2026-10 --cases 1900,2102,2009,1998
# options: --url (default http://localhost:5173), --browser <path to chrome.exe>
```
