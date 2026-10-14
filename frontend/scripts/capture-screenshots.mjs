// Capture Evaluation and Case Explorer screenshots for the report.
//
// Prerequisites: backend on :8000 and frontend (npm run dev) on :5173, and an evaluation
// cache already filled by scripts/run_case_explorer_cache.py. scripts/run_all.py --stages screenshots
// starts both servers and runs this for you.
//
// Cases default to the showcase cases in experiments/final_metrics.json. Only pass --cases IDs that
// are cached: opening an uncached case in Case Explorer runs all systems live (and costs money).
//
// Usage (from frontend/):
//   node scripts/capture-screenshots.mjs                       # -> ../eval-dashboard-screenshots
//   node scripts/capture-screenshots.mjs --cases 1900,2102 --out ../some/dir
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index > -1 ? process.argv[index + 1] : fallback;
}

const BASE_URL = arg('url', 'http://localhost:5173');
const OUT_DIR = path.resolve(arg('out', '../eval-dashboard-screenshots'));

function showcaseCaseIds() {
  try {
    const metrics = JSON.parse(fs.readFileSync(path.resolve('../experiments/final_metrics.json'), 'utf-8'));
    return (metrics.showcase || []).map((item) => String(item.sample_id));
  } catch {
    return [];
  }
}

const CASE_IDS = arg('cases', '')
  ? arg('cases', '').split(',').map((id) => id.trim()).filter(Boolean)
  : showcaseCaseIds();
const BROWSER = arg('browser', [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].find((candidate) => fs.existsSync(candidate)));

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function shotRegion(page, file, firstSelector, lastSelector, padding = 16) {
  const first = await page.$(firstSelector);
  const last = await page.$(lastSelector);
  const a = await first.boundingBox();
  const b = await last.boundingBox();
  const clip = {
    x: Math.max(0, Math.min(a.x, b.x) - padding),
    y: Math.max(0, a.y - padding),
    width: Math.max(a.x + a.width, b.x + b.width) - Math.min(a.x, b.x) + padding * 2,
    height: b.y + b.height - a.y + padding * 2,
  };
  await page.screenshot({ path: path.join(OUT_DIR, file), clip, captureBeyondViewport: true });
  console.log('saved', file);
}

async function shotElement(page, file, element) {
  await element.screenshot({ path: path.join(OUT_DIR, file), captureBeyondViewport: true });
  console.log('saved', file);
}

async function main() {
  if (!BROWSER) throw new Error('No Chrome/Edge found; pass --browser <path>');
  fs.mkdirSync(OUT_DIR, { recursive: true });
  // Case screenshots from an earlier run may show other cases: remove them so only current ones remain.
  for (const file of fs.readdirSync(OUT_DIR)) {
    if (/^\d\d-case-\d+-.*\.png$/.test(file)) fs.unlinkSync(path.join(OUT_DIR, file));
  }
  const browser = await puppeteer.launch({ executablePath: BROWSER, headless: true, args: ['--hide-scrollbars'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1680, height: 1050, deviceScaleFactor: 1.5 });
  await page.goto(BASE_URL, { waitUntil: 'networkidle0' });

  // --- Evaluation tab -----------------------------------------------------------------
  await page.click('#nav-evaluation');
  await page.waitForSelector('.evaluation-system-grid', { timeout: 120000 });
  await sleep(1500);
  await shotRegion(page, '01-eval-kpi-cards.png', '.evaluation-system-grid', '.evaluation-metric-grid');

  const breakdownCard = async () => (await page.$$('.evaluation-two-col > .card'))[0];
  await shotElement(page, '02-accuracy-by-question-type.png', await breakdownCard());
  await page.select('select[aria-label="Select evaluation breakdown mode"]', 'controller_path');
  await sleep(800);
  await shotElement(page, '03-accuracy-by-controller-path.png', await breakdownCard());

  const casesCard = (await page.$$('.evaluation-two-col > .card'))[1];
  if (casesCard) await shotElement(page, '04-evaluated-cases-table.png', casesCard);

  // --- Case Explorer traces for selected (cached) cases ----------------------------------
  for (const [index, caseId] of CASE_IDS.entries()) {
    await page.click('#nav-explorer');
    await page.waitForSelector('.case-explorer-layout');
    // A case may still be selected from the previous iteration: clear it first.
    const [clear] = await page.$$('xpath/.//button[contains(., "Clear selection")]');
    if (clear) {
      await clear.click();
      await sleep(500);
    }
    const [openButton] = await page.$$('xpath/.//button[contains(., "Select a sample")]');
    await openButton.click();
    await page.type('input[placeholder="Search by ID or question..."]', caseId);
    await page.keyboard.press('Enter');
    await page.waitForSelector('.agent-trace', { timeout: 120000 });
    await sleep(1500);
    const prefix = `${String(5 + index).padStart(2, '0')}-case-${caseId}`;
    // Typing into the search box can scroll the page under the sticky header; reset before the overview.
    await page.evaluate(() => {
      document.activeElement?.blur();
      window.scrollTo(0, 0);
      document.querySelectorAll('*').forEach((el) => { if (el.scrollTop) el.scrollTop = 0; });
    });
    await sleep(400);
    await page.screenshot({ path: path.join(OUT_DIR, `${prefix}-overview.png`) });
    // Panels scroll internally; let them grow to full height so the whole agent trace is captured.
    const unclip = await page.addStyleTag({ content: `
      .case-explorer-layout, .case-explorer-layout .panel, .case-explorer-layout .panel-body {
        height: auto !important; max-height: none !important; overflow: visible !important;
      }
      .app-header { display: none !important; }` });
    await sleep(500);
    const panels = await page.$$('.case-explorer-layout > .panel');
    await shotElement(page, `${prefix}-analysis-panel.png`, panels[panels.length - 1]);
    await unclip.evaluate((node) => node.remove());
    console.log('saved case', caseId);
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
