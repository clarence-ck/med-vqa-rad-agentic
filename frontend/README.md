# MedVisionQuery frontend

This is the React 19 and Vite interface for MedVisionQuery. It connects to the FastAPI backend through
`VITE_API_BASE`. The default address is `http://localhost:8000`; see `src/config.js`. For information about the full
system, see the [root README](../README.md).

## Run

Run these commands from the `frontend/` folder:

```bash
npm install
npm run dev      # http://localhost:5173 (start the backend first)
npm run lint
npm run build
```

## Views

- **Dashboard** — upload an image and ask a question. Compare the agentic answer with the single-shot and workflow
  baselines. The agentic view also shows its explanation, verification result, and decision trace.
- **Case Explorer** — browse questions from `experiments/eval_set.json` and compare saved results from the single-shot
  baseline, workflow baseline, and agentic loop. The main components are `CaseExplorer.jsx`, `AgentTrace.jsx`,
  `VerificationPanel.jsx`, and `FailureModeDrawer.jsx`.
- **Evaluation** (`EvaluationView.jsx`) — reads `/api/dataset/evaluation` and shows strict and LLM-judged accuracy,
  mean and median response time, accuracy by question type and controller path, system disagreements, and a table of
  every case. Data-loading hooks are in `src/hooks/`.

## Screenshots

`scripts/capture-screenshots.mjs` controls a local Chrome or Edge browser with `puppeteer-core`. It saves the
Evaluation view and the Case Explorer traces for the example cases listed in the current run's `final_metrics.json`.
The simplest option is to run `python scripts/run_all.py --stages screenshots` from the repository root. That command
starts both servers, captures the images, and stops the servers.

```bash
node scripts/capture-screenshots.mjs                 # requires the backend and npm run dev to be running
# options: --cases 1900,2102 (cached cases only), --out <dir>, --url <address>, --browser <path>
```
