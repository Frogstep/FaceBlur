// FaceBlur web app: upload a video, label its face cards, render, download.
// Plain ES module, no build step; talks to the same server's /api/v1.
// Every text the user sees goes through t()/tn() (i18n.js: English, Russian).

import { LANGUAGES, language, setLanguage, t, tn } from "./i18n.js";

const API = "/api/v1";
const app = document.getElementById("app");
let config = { max_upload: 8 * 2 ** 30, chunk_size: 16 * 2 ** 20 };
let view = null; // current view: { destroy() }
let signedIn = false;

// ------------------------------------------------------------------ helpers

class ApiError extends Error {
  constructor(status, message, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

// server messages are English; known ones are translated, the rest shown as they are
function detailText(d) {
  if (!d) return "";
  if (typeof d === "string") return t(d);
  if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  if (d.message) return t(d.message);
  return JSON.stringify(d);
}

async function api(path, { method = "GET", json } = {}) {
  const opt = { method, headers: { "X-FaceBlur": "1" }, credentials: "same-origin" };
  if (json !== undefined) {
    opt.body = JSON.stringify(json);
    opt.headers["Content-Type"] = "application/json";
  }
  let r;
  try {
    r = await fetch(API + path, opt);
  } catch (e) {
    throw new ApiError(0, t("The server is not reachable."));
  }
  if (r.status === 401 && !path.startsWith("/session")) {
    showSignIn();
    throw new ApiError(401, t("Please sign in."));
  }
  if (!r.ok) {
    let d;
    try { d = (await r.json()).detail; } catch { /* not JSON */ }
    throw new ApiError(r.status, detailText(d) || `${r.status} ${r.statusText}`, d);
  }
  if (r.status === 204) return null;
  return r.json();
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function toast(msg, kind = "") {
  const el = h("div", { class: `toast ${kind}` }, msg);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

function fail(e) {
  if (e && e.status === 401) return;
  console.error(e);
  toast(e && e.message ? e.message : String(e), "error");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fmtBytes(n) {
  if (n == null) return "";
  const u = language() === "ru" ? ["Б", "КБ", "МБ", "ГБ"] : ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toLocaleString(language(), { maximumFractionDigits: i && n < 10 ? 1 : 0 })} ${u[i]}`;
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function fmtDate(ts) {
  const d = new Date(ts * 1000);
  const time = d.toLocaleTimeString(language(), { hour: "2-digit", minute: "2-digit" });
  return new Date().toDateString() === d.toDateString() ? time
    : `${d.toLocaleDateString(language(), { month: "short", day: "numeric" })} ${time}`;
}

const STATUS = {
  uploading: "Uploading",
  queued_scan: "Waiting to detect faces",
  scanning: "Detecting faces",
  review: "Choose faces to blur",
  queued_render: "Waiting to render",
  rendering: "Rendering",
  checking: "Safety check",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};
const BUSY = new Set(["queued_scan", "scanning", "queued_render", "rendering", "checking"]);
const statusText = (s) => t(STATUS[s] || s);

function pill(status) {
  const cls = BUSY.has(status) || status === "uploading" ? "busy" : status === "review" ? "review"
    : status === "done" ? "done" : status === "failed" ? "failed" : "";
  return h("span", { class: `pill ${cls}` }, statusText(status));
}

function progressBar(done, total) {
  const bar = h("div", { class: "progress" + (total ? "" : " indeterminate") }, h("div"));
  if (total) bar.firstChild.style.width = `${Math.min(100, (100 * done) / total)}%`;
  return bar;
}

// ------------------------------------------------------------------ header

const langSelect = document.getElementById("lang");
const signOutBtn = document.getElementById("signout");

function drawHeader() {
  langSelect.replaceChildren(...Object.entries(LANGUAGES).map(([code, name]) => h("option", { value: code }, name)));
  langSelect.value = language();
  langSelect.setAttribute("aria-label", t("Language"));
  signOutBtn.textContent = t("Sign out");
  signOutBtn.hidden = !signedIn;
}

langSelect.addEventListener("change", () => {
  setLanguage(langSelect.value);
  drawHeader();
  if (signedIn) route(); else showSignIn();
});

signOutBtn.addEventListener("click", async () => {
  try { await api("/session", { method: "DELETE" }); } catch { /* ignore */ }
  showSignIn(t("Signed out."));
});

// ------------------------------------------------------------------ routing

function setView(v) {
  if (view && view.destroy) view.destroy();
  view = v;
}

function route() {
  const m = location.hash.match(/^#\/job\/([0-9a-f]+)/);
  if (m) setView(jobView(m[1]));
  else setView(jobsView());
}

window.addEventListener("hashchange", () => { if (signedIn) route(); });

// ------------------------------------------------------------------ sign in

let signInMessage = "";
function showSignIn(message) {
  if (message !== undefined) signInMessage = message;
  setView(null);
  signedIn = false;
  drawHeader();
  const input = h("input", { type: "password", placeholder: t("Access token"), autocomplete: "current-password", required: true });
  const err = h("p", { class: "error" }, signInMessage);
  const form = h("form", {
    onsubmit: async (e) => {
      e.preventDefault();
      try {
        await api("/session", { method: "POST", json: { token: input.value } });
        signInMessage = "";
        await start();
      } catch (ex) {
        err.textContent = ex.status === 401 ? t("That token is not right.") : ex.message;
      }
    },
  },
  h("p", { class: "muted" }, t("Paste the token that faceblur serve printed (it is also in server_token.txt), or open the sign-in link it printed.")),
  input, h("button", { class: "primary", type: "submit" }, t("Sign in")), err);
  app.replaceChildren(h("div", { class: "panel signin" }, h("h2", {}, t("Sign in to FaceBlur")), form));
  input.focus();
}

// ------------------------------------------------------------------ uploads
// Resumable: POST /uploads, then PATCH chunks with Upload-Offset. Uploads live
// outside the views, so navigating does not stop them.

const uploads = new Map(); // job id -> upload state
const uploadListeners = new Set();
let notifyPending = false;
function notifyUploads() {
  if (notifyPending) return;
  notifyPending = true;
  requestAnimationFrame(() => {
    notifyPending = false;
    uploadListeners.forEach((f) => f());
  });
}

window.addEventListener("beforeunload", (e) => {
  if ([...uploads.values()].some((u) => u.state === "uploading")) {
    e.preventDefault();
    e.returnValue = "";
  }
});

function sendChunk(up, offset, blob) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    up.xhr = xhr;
    xhr.open("PATCH", `${API}/jobs/${up.id}/upload`);
    xhr.setRequestHeader("Upload-Offset", String(offset));
    xhr.setRequestHeader("X-FaceBlur", "1");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { up.sent = offset + e.loaded; notifyUploads(); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* empty */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else {
        const d = body && body.detail;
        reject(new ApiError(xhr.status, detailText(d) || `${xhr.status}`, d));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network error"));
    xhr.onabort = () => reject(new ApiError(-1, "cancelled"));
    xhr.send(blob);
  });
}

async function startUpload(file) {
  if (file.size > config.max_upload) {
    toast(t("{name} is {size}; the server accepts up to {max}.",
      { name: file.name, size: fmtBytes(file.size), max: fmtBytes(config.max_upload) }), "error");
    return;
  }
  if ([...uploads.values()].some((u) => u.file.name === file.name && u.file.size === file.size && u.state === "uploading")) {
    toast(t("{name} is already uploading.", { name: file.name }));
    return;
  }
  let id, offset;
  try {
    // the same file picked again resumes an interrupted upload
    const jobs = await api("/jobs");
    const old = jobs.find((j) => j.status === "uploading" && j.name === file.name && j.upload && j.upload.size === file.size);
    if (old) {
      id = old.id;
      offset = (await api(`/jobs/${id}/upload`)).offset;
      toast(t("Resuming {name} at {size}.", { name: file.name, size: fmtBytes(offset) }));
    } else {
      const r = await api("/uploads", { method: "POST", json: { name: file.name, size: file.size } });
      id = r.id;
      offset = 0;
    }
  } catch (e) {
    fail(e);
    return;
  }
  const up = { id, file, sent: offset, state: "uploading", error: "", xhr: null };
  uploads.set(id, up);
  notifyUploads();
  let retries = 0;
  while (offset < file.size && up.state === "uploading") {
    const end = Math.min(offset + config.chunk_size, file.size);
    try {
      const r = await sendChunk(up, offset, file.slice(offset, end));
      offset = r.offset;
      up.sent = offset;
      retries = 0;
      up.error = "";
      if (r.status !== "uploading") break;
    } catch (e) {
      if (up.state !== "uploading") break; // cancelled
      if (e.status === 409 && e.detail && typeof e.detail.offset === "number") {
        offset = e.detail.offset;
        continue;
      }
      if (e.status >= 400 && e.status < 500) {
        up.state = "failed";
        up.error = e.message;
        break;
      }
      if (++retries > 8) {
        up.state = "failed";
        up.error = t("The connection keeps failing. Choose the same file again to resume.");
        break;
      }
      up.error = t("Connection problem, retrying ({k}/8)…", { k: retries });
      notifyUploads();
      await sleep(Math.min(30000, 1000 * 2 ** retries));
      try { offset = (await api(`/jobs/${id}/upload`)).offset; } catch { /* retry */ }
    }
  }
  if (up.state === "uploading") {
    up.state = "done";
    uploads.delete(id);
    toast(t("{name} uploaded. Detecting faces…", { name: file.name }));
    if (!location.hash.startsWith("#/job/")) location.hash = `#/job/${id}`;
  } else if (up.state === "failed") {
    toast(`${file.name}: ${up.error}`, "error");
  }
  notifyUploads();
}

async function cancelUpload(up) {
  up.state = "cancelled";
  if (up.xhr) up.xhr.abort();
  uploads.delete(up.id);
  notifyUploads();
  try { await api(`/jobs/${up.id}`, { method: "DELETE" }); } catch (e) { fail(e); }
}

function filePicker(label, primary = true) {
  const input = h("input", {
    type: "file", accept: "video/*,.mp4,.mov,.mkv,.avi,.m4v,.webm", multiple: true, class: "hidden",
    onchange: () => { [...input.files].forEach(startUpload); input.value = ""; },
  });
  const btn = h("button", { class: primary ? "primary" : "", type: "button", onclick: () => input.click() }, label);
  return h("span", {}, input, btn);
}

// ------------------------------------------------------------------ jobs list

function jobsView() {
  let timer = null;
  let alive = true;
  const uploadsBox = h("div");
  const listBox = h("div", {}, h("p", { class: "muted" }, t("Loading…")));

  const drop = h("div", { class: "drop" },
    h("strong", {}, t("Drop a video here")),
    h("span", { class: "muted small" }, t("or choose one (up to {max}). Large files upload in parts and can be resumed.",
      { max: fmtBytes(config.max_upload) })),
    filePicker(t("Choose video")));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    [...e.dataTransfer.files].forEach(startUpload);
  });

  app.replaceChildren(
    h("div", { class: "panel" }, drop, uploadsBox),
    h("div", { class: "panel" }, h("h2", {}, t("Videos")), listBox));

  function drawUploads() {
    const items = [...uploads.values()];
    uploadsBox.replaceChildren(...items.map((u) => h("div", { class: "row", style: { marginTop: "12px" } },
      h("strong", { style: { overflowWrap: "anywhere" } }, u.file.name),
      progressBar(u.sent, u.file.size),
      h("span", { class: "muted small" }, `${fmtBytes(u.sent)} / ${fmtBytes(u.file.size)}`),
      u.error ? h("span", { class: "small error" }, u.error) : null,
      h("button", { class: "danger", onclick: () => cancelUpload(u) }, t("Cancel")))));
  }
  uploadListeners.add(drawUploads);
  drawUploads();

  async function load() {
    try {
      const jobs = await api("/jobs");
      if (!alive) return;
      drawJobs(jobs);
      const busy = jobs.some((j) => BUSY.has(j.status));
      timer = setTimeout(load, busy ? 2000 : 8000);
    } catch (e) {
      if (!alive) return;
      fail(e);
      if (e.status !== 401) timer = setTimeout(load, 8000);
    }
  }

  function drawJobs(jobs) {
    if (!jobs.length) {
      listBox.replaceChildren(h("p", { class: "muted" }, t("No videos yet.")));
      return;
    }
    const rows = jobs.map((j) => {
      let progress = "";
      if (j.status === "uploading" && j.upload) {
        progress = uploads.has(j.id) ? ""
          : t("{done} of {size} — choose the file again to resume", { done: fmtBytes(j.upload.offset), size: fmtBytes(j.upload.size) });
      } else if (BUSY.has(j.status)) {
        progress = j.queue_position > 0 ? t("#{k} in queue", { k: j.queue_position })
          : j.progress.total ? `${t(j.progress.stage)} ${Math.round((100 * j.progress.done) / j.progress.total)}%` : t(j.progress.stage);
      } else if (j.error) progress = t(j.error);
      return h("tr", {},
        h("td", { class: "name" }, h("a", { href: `#/job/${j.id}` }, j.name)),
        h("td", {}, pill(j.status)),
        h("td", { class: "small muted" }, progress),
        h("td", { class: "num small muted hide-sm" }, j.video && j.video.duration ? fmtTime(j.video.duration) : ""),
        h("td", { class: "num small muted hide-sm" }, fmtDate(j.created)),
        h("td", { class: "num" }, h("button", {
          class: "link danger", disabled: BUSY.has(j.status) && !j.status.startsWith("queued"),
          onclick: async () => {
            if (!confirm(t("Delete {name} and its result?", { name: j.name }))) return;
            const up = uploads.get(j.id);
            if (up) { await cancelUpload(up); load(); return; }
            try { await api(`/jobs/${j.id}`, { method: "DELETE" }); load(); } catch (e) { fail(e); }
          },
        }, t("Delete"))));
    });
    listBox.replaceChildren(h("table", { class: "jobs" },
      h("thead", {}, h("tr", {}, [t("Video"), t("Status"), "", t("Length"), t("Added"), ""].map((x) => h("th", {}, x)))),
      h("tbody", {}, rows)));
  }

  load();
  return {
    destroy() {
      alive = false;
      clearTimeout(timer);
      uploadListeners.delete(drawUploads);
    },
  };
}

// ------------------------------------------------------------------ job page

function jobView(id) {
  let alive = true;
  let timer = null;
  let job = null;
  let cards = null; // from /cards
  let project = null; // { blur: [card numbers], settings }
  let cardsKey = ""; // the face scan the cards belong to
  let filter = "all";
  let modal = null;

  const head = h("div", { class: "jobhead" });
  const statusBox = h("div");
  const resultBox = h("div");
  const reviewBox = h("div");
  app.replaceChildren(head, statusBox, resultBox, reviewBox);

  // -------------------------------------------------------------- polling
  async function poll() {
    clearTimeout(timer);
    try {
      job = await api(`/jobs/${id}`);
      if (!alive) return;
      drawHead();
      drawStatus();
      // (re)load the cards after each finished face scan
      const key = job.cards ? String(job.scan_rev) : "";
      if (key && key !== cardsKey) {
        cardsKey = key;
        await loadReview();
      } else if (!key && cards !== null) {
        cards = null;
        project = null;
        cardsKey = "";
        reviewBox.replaceChildren();
      }
      drawResult();
      updateRenderButton();
      timer = setTimeout(poll, BUSY.has(job.status) || job.status === "uploading" ? 1500 : 10000);
    } catch (e) {
      if (!alive) return;
      if (e.status === 404) {
        app.replaceChildren(h("div", { class: "panel" }, h("p", {}, t("This video was deleted.")),
          h("a", { href: "#/" }, t("Back to the list"))));
        return;
      }
      fail(e);
      if (e.status !== 401) timer = setTimeout(poll, 5000);
    }
  }

  async function action(fn) {
    try { await fn(); } catch (e) { fail(e); }
    poll();
  }

  // -------------------------------------------------------------- header + status
  function drawHead() {
    const v = job.video || {};
    head.replaceChildren(
      h("a", { href: "#/", class: "button" }, t("← Videos")),
      h("div", { class: "title" },
        h("h1", {}, job.name),
        h("div", { class: "muted small" },
          [v.width && `${v.width}×${v.height}`, v.fps && `${v.fps} fps`, v.duration && fmtTime(v.duration),
            v.size && fmtBytes(v.size), v.has_audio === false && t("no audio")].filter(Boolean).join(" · "))),
      pill(job.status),
      h("button", {
        disabled: BUSY.has(job.status) || job.status === "uploading",
        title: t("Discard the detected faces and the choice of faces to blur, and scan the video again"),
        onclick: () => confirm(t("Detect faces again? The choice of faces to blur is cleared.")) &&
          action(() => api(`/jobs/${id}/scan`, { method: "POST" })),
      }, t("Detect again")),
      h("button", {
        class: "danger", disabled: ["scanning", "rendering", "checking"].includes(job.status),
        onclick: async () => {
          if (!confirm(t("Delete {name} and its result?", { name: job.name }))) return;
          try { await api(`/jobs/${id}`, { method: "DELETE" }); location.hash = "#/"; } catch (e) { fail(e); }
        },
      }, t("Delete")));
  }

  function drawStatus() {
    const parts = [];
    if (job.status === "uploading") {
      const up = uploads.get(id);
      if (up) {
        parts.push(h("div", { class: "row" }, h("strong", {}, t("Uploading")), progressBar(up.sent, up.file.size),
          h("span", { class: "small muted" }, `${fmtBytes(up.sent)} / ${fmtBytes(up.file.size)}`),
          up.error ? h("span", { class: "small error" }, up.error) : null));
      } else if (job.upload) {
        parts.push(h("div", { class: "alert warn" },
          h("p", {}, t("The upload stopped at {done} of {size}. Choose the same file to continue.",
            { done: fmtBytes(job.upload.offset), size: fmtBytes(job.upload.size) })),
          filePicker(t("Choose the file"))));
      }
    } else if (BUSY.has(job.status)) {
      const p = job.progress;
      const queued = job.queue_position > 0;
      const label = queued ? t("Waiting: #{k} in the queue", { k: job.queue_position }) : p.stage ? t(p.stage) : statusText(job.status);
      parts.push(h("div", { class: "row" },
        h("strong", {}, label),
        progressBar(p.done, queued ? 0 : p.total),
        p.total && !queued ? h("span", { class: "small muted" }, `${p.done} / ${p.total}`) : null,
        h("button", { onclick: () => action(() => api(`/jobs/${id}/cancel`, { method: "POST" })) }, t("Cancel"))));
    }
    if (job.error && !BUSY.has(job.status)) parts.push(h("div", { class: "alert" }, t(job.error)));
    statusBox.replaceChildren(...(parts.length ? [h("div", { class: "panel" }, ...parts)] : []));
  }

  // -------------------------------------------------------------- result
  let resultShown = "";
  function drawResult() {
    const res = job.result;
    const key = res ? `${res.size}:${job.status}:${res.flagged}` : "";
    if (key === resultShown) return;
    resultShown = key;
    if (!res) { resultBox.replaceChildren(); return; }
    const url = `${API}/jobs/${id}/result`;
    const note = h("p", { class: "small muted hidden" },
      t("This browser cannot play the video (it is H.265). Download it and play it in another app."));
    const video = h("video", { controls: true, preload: "metadata", playsinline: true, src: `${url}?download=false&s=${res.size}` });
    video.addEventListener("error", () => note.classList.remove("hidden"));
    let safety;
    if (res.safety === "off") safety = h("p", { class: "small muted" }, t("The safety check is off (turn it on in the settings below)."));
    else if (res.safety !== "done") safety = h("div", { class: "alert warn" }, t("The safety check was skipped."));
    else if (res.flagged) safety = h("div", { class: "alert warn" }, tn(res.flagged,
      "A masked face may still be recognizable on {n} frame. Check it below; fix hard frames in the desktop app if needed.",
      "A masked face may still be recognizable on {n} frames. Check them below; fix hard frames in the desktop app if needed."));
    else safety = h("div", { class: "alert ok" }, t("Safety check: no masked face is recognizable in the result."));
    const flaggedBox = h("div");
    resultBox.replaceChildren(h("div", { class: "panel" },
      h("div", { class: "row", style: { marginBottom: "10px" } },
        h("h2", { style: { margin: 0, flex: 1 } },
          BUSY.has(job.status) && job.status !== "checking" ? t("Previous result") : t("Result")),
        h("span", { class: "small muted" }, [res.frames && tn(res.frames, "{n} frame", "{n} frames"), fmtBytes(res.size)].filter(Boolean).join(" · ")),
        h("a", { class: "button primary", href: `${url}?download=true`, download: "" }, t("Download"))),
      video, note, safety, flaggedBox));
    if (res.safety === "done" && res.flagged) {
      api(`/jobs/${id}/safety`).then((rep) => {
        flaggedBox.replaceChildren(h("div", { class: "flagged" }, rep.flagged.slice(0, 60).map((f) =>
          h("figure", { onclick: () => openImage(f.image, t("Frame {f} ({time}), similarity {sim}", { f: f.frame, time: fmtTime(f.time), sim: f.sim.join(", ") })) },
            h("img", { src: f.image, loading: "lazy", alt: "" }),
            h("figcaption", {}, t("{time} · frame {f}", { time: fmtTime(f.time), f: f.frame }))))),
        rep.flagged.length > 60 ? h("p", { class: "small muted" }, t("…and {n} more.", { n: rep.flagged.length - 60 })) : null);
      }).catch(fail);
    }
  }

  // -------------------------------------------------------------- review
  // Every face starts as not blurred; a click on a card switches it.
  let cardEls = new Map(); // number -> element
  let blurSet = new Set(); // numbers of the cards to blur
  let counterEl, renderBtn, renderWhy, gridEl;

  async function loadReview() {
    try {
      [cards, project] = await Promise.all([api(`/jobs/${id}/cards`), api(`/jobs/${id}/project`)]);
    } catch (e) { fail(e); return; }
    if (!alive) return;
    blurSet = new Set(project.blur);
    drawReview();
  }

  function drawReview() {
    counterEl = h("strong", { class: "count" });
    gridEl = h("div", { class: "cards" });
    renderBtn = h("button", { class: "primary", onclick: () => action(() => api(`/jobs/${id}/render`, { method: "POST" })) }, t("Render"));
    renderWhy = h("span", { class: "small muted" });
    const filterSel = h("select", {
      "aria-label": t("Show"),
      onchange: () => { filter = filterSel.value; drawGrid(); },
    }, h("option", { value: "all" }, t("All faces")), h("option", { value: "blur" }, t("Blurred")),
    h("option", { value: "keep" }, t("Not blurred")));
    filterSel.value = filter;

    reviewBox.replaceChildren(
      h("div", { class: "panel" },
        h("h2", {}, t("Faces")),
        h("p", { class: "muted" }, t("Every face stays visible until you choose to blur it. Click a face to blur it; click it again to leave it visible.")),
        h("div", { class: "toolbar" },
          counterEl,
          h("span", { class: "spacer" }),
          h("div", { class: "group" }, filterSel,
            h("button", { onclick: () => setAll(true) }, t("Blur all")),
            h("button", { onclick: () => setAll(false) }, t("Blur none")))),
        gridEl),
      settingsPanel(),
      h("div", { class: "panel row" }, renderBtn, renderWhy));
    drawGrid();
    updateCounts();
  }

  function visibleCards() {
    if (filter === "blur") return cards.filter((c) => blurSet.has(c.number));
    if (filter === "keep") return cards.filter((c) => !blurSet.has(c.number));
    return cards;
  }

  function drawGrid() {
    cardEls = new Map();
    const list = visibleCards();
    if (!list.length) {
      gridEl.replaceChildren(h("p", { class: "muted" }, !cards.length ? t("No faces were found in this video.")
        : filter === "blur" ? t("No face is blurred yet.") : t("Every face is blurred.")));
      return;
    }
    gridEl.replaceChildren(...list.map((c) => {
      const state = h("span", { class: "state" });
      const el = h("div", { class: "card", tabindex: "0", role: "switch", "data-n": c.number, onclick: () => toggle(c.number) },
        h("div", { class: "head" },
          h("span", { class: "num" }, `#${c.number}`),
          state,
          h("span", { class: "meta" }, `${fmtTime(c.start)}–${fmtTime(c.end)} · ${tn(c.frames, "{n} frame", "{n} frames")}`),
          h("button", { class: "view", onclick: (e) => { e.stopPropagation(); openFrame(c); }, title: t("Show this face on the video") }, t("View"))),
        h("div", { class: "thumbs" }, c.thumbs.map((src) => h("img", { src, loading: "lazy", alt: "", width: "160", height: "160" }))));
      el._state = state;
      cardEls.set(c.number, el);
      paintCard(c.number);
      return el;
    }));
  }

  function paintCard(n) {
    const el = cardEls.get(n);
    if (!el) return;
    const on = blurSet.has(n);
    el.classList.toggle("blur", on);
    el.setAttribute("aria-checked", String(on));
    el.setAttribute("aria-label", t(on ? "Face #{n}: blurred" : "Face #{n}: not blurred", { n }));
    el._state.textContent = on ? t("Blurred") : t("Not blurred");
  }

  function updateCounts() {
    counterEl.textContent = tn(blurSet.size, "{n} of {total} faces blurred", "{n} of {total} faces blurred", { total: cards.length });
    counterEl.classList.toggle("on", blurSet.size > 0);
    updateRenderButton();
  }

  function updateRenderButton() {
    if (!renderBtn || !job || !project) return;
    let why = "";
    if (BUSY.has(job.status)) why = statusText(job.status) + "…";
    else if (!cards.length) why = t("No faces to blur.");
    else if (!blurSet.size) why = t("Click the faces you want to blur.");
    renderBtn.disabled = !!why;
    renderBtn.textContent = job.result ? t("Render again") : t("Render");
    renderWhy.textContent = why || (project.settings.safety_check
      ? tn(blurSet.size, "Blurs {n} face, then runs the safety check.", "Blurs {n} faces, then runs the safety check.")
      : tn(blurSet.size, "Blurs {n} face.", "Blurs {n} faces."));
  }

  async function setBlur(nums, on) {
    if (!nums.length) return;
    const before = new Set(blurSet);
    nums.forEach((n) => (on ? blurSet.add(n) : blurSet.delete(n)));
    nums.forEach(paintCard); // show the change at once; the server confirms it
    updateCounts();
    try {
      const r = await api(`/jobs/${id}/blur`, { method: "PATCH", json: { cards: Object.fromEntries(nums.map((n) => [String(n), on])) } });
      blurSet = new Set(r.blur);
    } catch (e) {
      blurSet = before;
      fail(e);
    }
    if (filter !== "all") drawGrid(); else for (const n of cardEls.keys()) paintCard(n);
    updateCounts();
  }

  function toggle(n) {
    return setBlur([n], !blurSet.has(n));
  }

  function setAll(on) {
    const nums = cards.map((c) => c.number).filter((n) => blurSet.has(n) !== on);
    if (!on && nums.length && !confirm(t("Remove the blur from all {n} faces?", { n: nums.length }))) return;
    return setBlur(nums, on);
  }

  // -------------------------------------------------------------- settings
  function settingsPanel() {
    const s = project.settings;
    let timer = null;
    let unsent = {}; // changes waiting to be sent (several may come within the delay)
    const save = (patch) => {
      Object.assign(s, patch);
      Object.assign(unsent, patch);
      updateRenderButton();
      clearTimeout(timer);
      timer = setTimeout(() => {
        const body = unsent;
        unsent = {};
        api(`/jobs/${id}/settings`, { method: "PATCH", json: body }).catch(fail);
      }, 250);
    };
    const range = (key, min, max, step, fmt) => {
      const out = h("span", { class: "muted" }, fmt(s[key]));
      const input = h("input", { type: "range", min, max, step, value: s[key],
        oninput: () => { out.textContent = fmt(+input.value); },
        onchange: () => save({ [key]: +input.value }) });
      return [input, out];
    };
    const style = h("select", { onchange: () => save({ style: style.value }) },
      h("option", { value: "pixelate" }, t("Pixelate")), h("option", { value: "blur" }, t("Blur")),
      h("option", { value: "fill" }, t("Solid fill")));
    style.value = s.style;
    const [strength, strengthOut] = range("strength", 0.5, 3, 0.1, (v) => `${v.toLocaleString(language(), { minimumFractionDigits: 1, maximumFractionDigits: 1 })}×`);
    const [pad, padOut] = range("pad", 0, 0.8, 0.05, (v) => `${Math.round(v * 100)}%`);
    const sam = h("input", { type: "checkbox", checked: s.use_sam, onchange: () => save({ use_sam: sam.checked }) });
    const check = h("input", { type: "checkbox", checked: s.safety_check,
      onchange: () => save({ safety_check: check.checked }) });
    return h("div", { class: "panel" },
      h("h2", {}, t("Settings")),
      h("div", { class: "settings" },
        h("label", {}, t("Style"), style),
        h("label", {}, h("span", {}, `${t("Strength")} `, strengthOut), strength),
        h("label", {}, h("span", {}, `${t("Margin around the face")} `, padOut), pad),
        h("label", { class: "inline" }, sam, h("span", {}, t("Exact face outline (SAM 2)"),
          h("br"), h("span", { class: "small muted" }, t("Slower. Off: ellipses, much faster")))),
        h("label", { class: "inline" }, check, h("span", {}, t("Run safety check after rendering"),
          h("br"), h("span", { class: "small muted" },
            t("Looks for blurred faces that are still recognizable in the result (slower)"))))));
  }

  // -------------------------------------------------------------- frame viewer
  function openFrame(card, sample = 0) {
    const fps = job.video.fps || 25;
    const total = job.video.frames || 1;
    let f = card.sample_frames[sample] ?? card.first_frame;
    const img = h("img", { class: "frame", alt: t("video frame") });
    const info = h("span", { class: "small muted" });
    const width = Math.min(1600, Math.round(Math.min(window.innerWidth, 1400) * (window.devicePixelRatio || 1)));
    let rev = 0; // reload the frame after a change (box colours)
    const show = () => {
      f = Math.max(0, Math.min(total - 1, Math.round(f)));
      img.src = `${API}/jobs/${id}/frames/${f}.jpg?w=${width}&v=${rev}`;
      info.textContent = t("Frame {f} · {time}", { f, time: fmtTime(f / fps) });
    };
    const step = (d) => () => { f += d; show(); };
    const sampleStep = (d) => () => {
      const fr = card.sample_frames;
      const i = d > 0 ? fr.findIndex((x) => x > f) : fr.length - 1 - [...fr].reverse().findIndex((x) => x < f);
      if (i >= 0 && i < fr.length && fr[i] !== undefined) { f = fr[i]; show(); }
    };
    const state = h("span", { class: "state" });
    const switchBtn = h("button", { class: "primary" });
    const paint = () => {
      const on = blurSet.has(card.number);
      state.textContent = on ? t("Blurred") : t("Not blurred");
      state.classList.toggle("on", on);
      switchBtn.textContent = on ? t("Don't blur #{n}", { n: card.number }) : t("Blur #{n}", { n: card.number });
    };
    const flip = async () => { await toggle(card.number); paint(); rev++; show(); };
    switchBtn.onclick = flip;
    paint();
    const box = h("div", { class: "box", role: "dialog", "aria-label": t("Face #{n}", { n: card.number }) },
      h("div", { class: "row" }, h("strong", {}, t("Face #{n}", { n: card.number })), state, info, h("span", { class: "spacer" }),
        h("button", { onclick: closeModal }, t("Close"))),
      img,
      h("div", { class: "row" },
        switchBtn,
        h("span", { class: "spacer" }),
        h("button", { onclick: sampleStep(-1), title: t("An earlier picture of this face") }, t("◀ Earlier")),
        h("button", { onclick: step(-fps), title: "Shift+←" }, "−1 s"),
        h("button", { onclick: step(-1), title: "←" }, "−1"),
        h("button", { onclick: step(1), title: "→" }, "+1"),
        h("button", { onclick: step(fps), title: "Shift+→" }, "+1 s"),
        h("button", { onclick: sampleStep(1), title: t("A later picture of this face") }, t("Later ▶"))),
      h("p", { class: "small muted", style: { margin: 0 } },
        t("Red boxes are blurred, green boxes stay visible. The numbers are the face numbers in the list.")));
    modal = h("div", { class: "modal", onclick: (e) => { if (e.target === modal) closeModal(); } }, box);
    modal._keys = (e) => {
      if (e.key === "Escape") closeModal();
      else if (e.key === "ArrowLeft") (e.shiftKey ? step(-fps) : step(-1))();
      else if (e.key === "ArrowRight") (e.shiftKey ? step(fps) : step(1))();
      else if (e.code === "KeyB") flip();
      else return;
      e.preventDefault();
    };
    document.body.append(modal);
    show();
  }

  function openImage(src, caption) {
    modal = h("div", { class: "modal", onclick: closeModal },
      h("div", { class: "box" }, h("img", { class: "frame", src, alt: caption }), h("span", { class: "small muted" }, caption)));
    modal._keys = (e) => { if (e.key === "Escape") closeModal(); };
    document.body.append(modal);
  }

  function closeModal() {
    if (modal) modal.remove();
    modal = null;
  }

  // -------------------------------------------------------------- keys
  // on a focused face card: Space/Enter switches it, V shows it on the video,
  // the arrow keys move between cards
  function onKey(e) {
    if (modal) { modal._keys(e); return; }
    if (!cards || e.ctrlKey || e.metaKey || e.altKey || e.target.closest("button, input, select, textarea")) return;
    const el = e.target.closest(".card");
    if (!el) return;
    const n = +el.dataset.n;
    if (e.key === " " || e.key === "Enter") toggle(n);
    else if (e.code === "KeyV") openFrame(cards[n - 1]);
    else if (["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) {
      const order = visibleCards().map((c) => c.number);
      const i = order.indexOf(n) + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1);
      const next = cardEls.get(order[Math.max(0, Math.min(order.length - 1, i))]);
      if (next) { next.focus(); next.scrollIntoView({ block: "nearest" }); }
    } else return;
    e.preventDefault();
  }
  document.addEventListener("keydown", onKey);

  const redraw = () => { if (job) drawStatus(); };
  uploadListeners.add(redraw);
  poll();

  return {
    destroy() {
      alive = false;
      clearTimeout(timer);
      closeModal();
      document.removeEventListener("keydown", onKey);
      uploadListeners.delete(redraw);
    },
  };
}

// ------------------------------------------------------------------ start

async function start() {
  try {
    config = await api("/config");
  } catch (e) {
    if (e.status !== 401) {
      app.replaceChildren(h("div", { class: "panel" }, h("p", { class: "error" }, e.message),
        h("button", { onclick: start }, t("Try again"))));
    }
    return;
  }
  signedIn = true;
  drawHeader();
  if (!location.hash.startsWith("#/")) history.replaceState(null, "", "#/");
  route();
}

async function boot() {
  drawHeader();
  // the sign-in link printed by `faceblur serve`: /#token=...
  const m = location.hash.match(/^#token=(.+)$/);
  if (m) {
    history.replaceState(null, "", location.pathname + "#/");
    try {
      await api("/session", { method: "POST", json: { token: decodeURIComponent(m[1]) } });
    } catch (e) {
      showSignIn(e.status === 401 ? t("The sign-in link is not valid (the token changed?).") : e.message);
      return;
    }
  }
  start();
}

boot();
