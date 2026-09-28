// Vanilla UI for the Codebase RAG API. All dynamic text is inserted as text nodes, never parsed as HTML.
const $ = (sel) => document.querySelector(sel);
let selectedFiles = [];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Network error: the API is unreachable.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

function showMessage(node, text, isError = false) {
  node.textContent = text;
  node.classList.toggle("error", isError);
}

async function refreshStatus() {
  try {
    const health = await api("/health");
    $("#status").textContent = `${health.indexed_chunks} chunks indexed`;
  } catch {
    $("#status").textContent = "API unreachable";
  }
}

// ---------- Indexing ----------
function setFiles(files) {
  selectedFiles = [...files];
  const list = $("#file-list");
  list.replaceChildren(...selectedFiles.map((f) => el("li", {}, f.webkitRelativePath || f.name)));
  $("#index-btn").disabled = selectedFiles.length === 0;
  showMessage($("#index-result"), selectedFiles.length ? `${selectedFiles.length} file(s) selected` : "");
}

$("#file-input").addEventListener("change", (e) => setFiles(e.target.files));
$("#folder-input").addEventListener("change", (e) => setFiles(e.target.files));
const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  setFiles(e.dataTransfer.files);
});

$("#index-btn").addEventListener("click", async () => {
  const button = $("#index-btn");
  const form = new FormData();
  for (const f of selectedFiles) form.append("files", f, f.webkitRelativePath || f.name);
  button.disabled = true;
  showMessage($("#index-result"), "Indexing…");
  try {
    const r = await api("/index/files", { method: "POST", body: form });
    let text = `Indexed ${r.files_indexed} file(s) into ${r.chunks_indexed} chunks (${r.total_chunks} total).`;
    if (r.skipped.length) text += ` Skipped: ${r.skipped.map((s) => `${s.path} (${s.reason})`).join(", ")}`;
    showMessage($("#index-result"), text);
    refreshStatus();
  } catch (err) {
    showMessage($("#index-result"), err.message, true);
  } finally {
    button.disabled = selectedFiles.length === 0;
  }
});

// ---------- Querying ----------
function renderSources(sources) {
  const box = el("div", { class: "sources" });
  for (const s of sources) {
    const summary = el("summary", {}, `${s.id}  ·  lines ${s.start_line}-${s.end_line}  ·  score ${s.score.toFixed(3)}`);
    box.append(el("details", {}, summary, el("pre", {}, el("code", {}, s.content))));
  }
  return box;
}

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const chat = $("#chat");
  chat.querySelector(".hint")?.remove();
  chat.append(el("div", { class: "bubble user" }, question));
  const reply = el("div", { class: "bubble" }, "Thinking…");
  chat.append(reply);
  chat.scrollTop = chat.scrollHeight;
  $("#question").value = "";
  try {
    const r = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: Number($("#query-k").value) }),
    });
    reply.replaceChildren(r.answer);
    if (r.sources.length) reply.append(renderSources(r.sources));
  } catch (err) {
    reply.classList.add("error");
    reply.replaceChildren(err.message);
  }
  chat.scrollTop = chat.scrollHeight;
});

$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});

// ---------- Evaluation ----------
const pct = (x) => `${(x * 100).toFixed(1)}%`;

function metric(label, value) {
  return el("div", { class: "metric" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value));
}

function renderReport(report) {
  const k = report.k;
  const cards = [
    metric(`Precision@${k}`, pct(report.retrieval.precision_at_k)),
    metric(`Recall@${k}`, pct(report.retrieval.recall_at_k)),
    metric("MRR", report.retrieval.mrr.toFixed(3)),
  ];
  const g = report.generation;
  if (g) {
    cards.push(
      metric("Judge overall", pct(g.overall)),
      metric("Faithfulness", `${g.faithfulness.toFixed(2)} / 5`),
      metric("Relevance", `${g.relevance.toFixed(2)} / 5`),
      metric("Correctness", `${g.correctness.toFixed(2)} / 5`),
    );
  }
  $("#metrics").replaceChildren(...cards);

  const rows = report.results.map((row) => {
    const judge = row.judge ? pct(row.judge.overall) : row.error ? "error" : "—";
    const judgeCell = el("td", { class: "num" }, judge);
    if (row.judge) judgeCell.title = row.judge.reasoning;
    if (row.error) judgeCell.title = row.error;
    return el("tr", {},
      el("td", {}, row.question),
      el("td", { class: "num" }, pct(row.precision_at_k)),
      el("td", { class: "num" }, pct(row.recall_at_k)),
      el("td", { class: "num" }, row.reciprocal_rank.toFixed(2)),
      judgeCell);
  });
  $("#eval-table tbody").replaceChildren(...rows);
  $("#eval-table").hidden = false;
}

$("#eval-btn").addEventListener("click", async () => {
  const button = $("#eval-btn");
  const useJudge = $("#use-judge").checked;
  button.disabled = true;
  showMessage($("#eval-status"), useJudge ? "Running… the LLM judge can take a few minutes." : "Running…");
  try {
    const report = await api("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k: Number($("#eval-k").value), use_judge: useJudge }),
    });
    renderReport(report);
    const g = report.generation;
    const errors = report.results.filter((r) => r.error).length;
    let text = `Evaluated ${report.num_examples} questions.`;
    if (useJudge && !g) text += " LLM judge unavailable (see row tooltips); retrieval metrics only.";
    else if (errors) text += ` ${errors} judge error(s).`;
    showMessage($("#eval-status"), text);
  } catch (err) {
    showMessage($("#eval-status"), err.message, true);
  } finally {
    button.disabled = false;
  }
});

refreshStatus();
