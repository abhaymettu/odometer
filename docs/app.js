const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("en-US");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- the wheels: an odometer turns a wheel only while the wheel to its right rolls from 9 to 0.
const DIGITS = 11;
const wheels = [];
function buildWheels() {
  const host = $("wheels");
  for (let i = 0; i < DIGITS; i++) {
    const place = DIGITS - 1 - i;
    const wheel = document.createElement("div");
    wheel.className = "wheel" + (place === 0 ? " last" : "");
    const strip = document.createElement("div");
    strip.className = "strip";
    strip.setAttribute("aria-hidden", "true");
    for (let d = 0; d <= 10; d++) {
      const s = document.createElement("span");
      s.textContent = d % 10;
      strip.append(s);
    }
    wheel.append(strip);
    host.append(wheel);
    wheels.push({ strip, wheel, place, last: -1, position: 0 });
    if (place % 3 === 0 && place !== 0) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = ",";
      sep.setAttribute("aria-hidden", "true");
      host.append(sep);
    }
  }
}
function setReading(value, spinning = 0) {
  $("wheels").setAttribute("aria-label", `Odometer reading ${fmt(value)}`);
  const v = Math.max(0, value);
  let carry = v - Math.floor(v); // the wheel on the right is this far through its turn
  for (let i = wheels.length - 1; i >= 0; i--) {
    const w = wheels[i];
    const unit = 10 ** w.place;
    const digit = Math.floor(v / unit) % 10;
    const position = digit + carry;
    carry = digit === 9 ? carry : 0; // only a wheel rolling from 9 to 0 carries the next one
    const blur = spinning > unit * 1.5;
    if (position !== w.last) {
      w.strip.style.transform = `translate3d(0, ${(-position * 100) / 11}%, 0)`;
      w.last = position;
    }
    if (blur !== w.blurred) {
      w.wheel.classList.toggle("spin", blur);
      w.blurred = blur;
    }
  }
}
function rollTo(target, ms = 1400) {
  if (reduced) return setReading(target);
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / ms);
    const eased = 1 - (1 - p) ** 4;
    const value = target * eased;
    const rate = (target * 4 * (1 - p) ** 3) / (ms / 16.7);
    setReading(p === 1 ? target : value, p === 1 ? 0 : rate);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---- the demo
const curve = { pts: [], max: 8 };
let shown = 0, target = 0, lastPost = 0, rate = 0, animating = false;
function drawCurve(budget) {
  if (curve.pts.length < 2) return;
  const d = curve.pts.map(([x, y], i) => `${i ? "L" : "M"}${(x / budget * 400).toFixed(1)} ${(4 + (1 - y / curve.max) * 102).toFixed(1)}`).join("");
  $("curve-path").setAttribute("d", d);
}
function tick() {
  if (!animating) return;
  const now = performance.now();
  // follow the last known reading at the measured rate, never past the budget
  const guess = Math.min(target + rate * (now - lastPost), BUDGET);
  const next = Math.max(shown, guess);
  const perFrame = next - shown;
  shown = next;
  setReading(shown, perFrame);
  requestAnimationFrame(tick);
}
let BUDGET = 2e9, reference = null;
function runDemo() {
  const run = $("run");
  run.disabled = true;
  $("run-label").textContent = "Running";
  $("live").hidden = false;
  $("result").hidden = true;
  $("sample").textContent = "";
  $("reading-label").textContent = "Ticks spent on this device";
  curve.pts = [];
  $("curve-path").removeAttribute("d");
  shown = 0; target = 0; rate = 0; lastPost = performance.now(); animating = false;
  setReading(0);
  const started = performance.now();
  const worker = new Worker("worker.js");
  worker.onmessage = ({ data: m }) => {
    if (m.type === "status") $("status").textContent = m.text;
    if (m.type === "step") {
      const now = performance.now();
      if (target > 0 && now > lastPost) rate = 0.7 * rate + 0.3 * (m.used - target) / (now - lastPost);
      target = m.used; lastPost = now;
      if (m.loss !== null) {
        curve.pts.push([m.used, m.loss]);
        if (curve.pts.length % 4 === 0) drawCurve(BUDGET);
        $("loss-now").textContent = `Training loss ${m.loss.toFixed(2)} bits per byte`;
      }
      if (!animating) { animating = true; requestAnimationFrame(tick); }
    }
    if (m.type === "token") {
      if (animating) { animating = false; settle(); $("status").textContent = "Sampling"; }
      $("sample").textContent = m.text;
    }
    if (m.type === "done") finish(m.report, performance.now() - started, worker);
    if (m.type === "error") {
      animating = false;
      $("status").textContent = "It stopped: " + m.text;
      $("run-label").textContent = "Try again";
      run.disabled = false;
      worker.terminate();
    }
  };
  worker.postMessage("go");
}
let finalReading = null;
function settle() {
  setReading(finalReading ?? target, 0);
}
function finish(r, totalMs, worker) {
  animating = false;
  finalReading = r.reading;
  setReading(r.reading, 0);
  drawCurve(BUDGET);
  $("status").textContent = "";
  $("reading-label").textContent = `Stopped at ${fmt(r.reading)} ticks of a ${fmt(r.budget)} budget`;
  $("sample").textContent = r.text;
  const same = reference && reference.reading === r.reading;
  const others = reference ? reference.rows.map((x) => `${x.machine} ${x.train_seconds.toFixed(1)} s`).join(", ") : "";
  const rows = [
    ["The reading", same ? "Matches the reference to the tick" : reference ? "Differs from the reference" : fmt(r.reading), reference ? `Reference ${fmt(reference.reading)}` : ""],
    ["Time on this device", `${r.train_seconds.toFixed(1)} s`, `${r.ns_per_tick.toFixed(1)} ns per tick. ${others}`],
    ["Held-out bits per byte", r.bpb.toFixed(4), reference ? `Reference ${reference.bpb.toFixed(4)}. 32 windows, so this is a demo and not a score.` : "32 windows, so this is a demo and not a score."],
  ];
  const dl = $("result");
  dl.replaceChildren(...rows.map(([k, v, s]) => {
    const div = document.createElement("div");
    div.innerHTML = `<dt></dt><dd><span></span><small></small></dd>`;
    div.querySelector("dt").textContent = k;
    div.querySelector("span").textContent = v;
    div.querySelector("small").textContent = s;
    return div;
  }));
  dl.hidden = false;
  $("run-label").textContent = "Run it again";
  $("run").disabled = false;
  finalReading = null;
  worker.terminate();
}

// ---- the record table
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function recordRow(r, i, max) {
  const d = el("details", "rec" + (r.current ? " current" : ""));
  const s = el("summary");
  const bar = el("div", "bar");
  const fill = el("i");
  fill.style.width = `${(r.test / max) * 100}%`;
  bar.append(fill);
  s.append(el("span", "n", String(i)), el("span", "name", r.title), el("span", "bpb", r.test.toFixed(4)), bar, el("span", "who", r.who));
  d.append(s);
  const body = el("div", "detail");
  d.append(body);
  d.addEventListener("toggle", async () => {
    if (!d.open || body.childElementCount) return;
    const seeds = el("div", "seeds");
    seeds.append(el("span", "", `Test, ${r.seeds.length} fresh seeds, SD ${r.sd.toFixed(4)}:`), ...r.seeds.map((x) => el("span", "", x.toFixed(4))));
    const links = el("div", "links");
    for (const [label, href] of r.links) {
      const a = el("a", "", label);
      a.href = href;
      links.append(a);
    }
    body.append(seeds, links);
    if (r.why) body.append(el("p", "review", r.why));
    const [code, review] = await Promise.all([fetch(r.code).then((x) => x.text()), fetch(r.review).then((x) => x.text())]);
    body.append(el("div", "review", review.trim()), Object.assign(el("pre"), { textContent: code }));
  });
  return d;
}

async function inlineFigures(root = document) {
  await Promise.all([...root.querySelectorAll("[data-svg]")].map(async (f) => {
    const r = await fetch(f.dataset.svg);
    if (r.ok) f.insertAdjacentHTML("afterbegin", await r.text());
    else f.remove();
  }));
}

async function main() {
  buildWheels();
  rollTo(4e10);
  $("run").addEventListener("click", runDemo);
  const [data, ref] = await Promise.all([fetch("data.json").then((r) => r.json()), fetch("demo/reference.json").then((r) => (r.ok ? r.json() : null)).catch(() => null)]);
  reference = ref;
  BUDGET = ref?.budget ?? 2e9;
  $("records-lead").textContent = data.lead;
  const max = Math.max(...data.records.map((r) => r.test));
  $("table").append(...data.records.map((r, i) => recordRow(r, data.records.length - i, max)));
  for (const [id, html] of Object.entries(data.html)) $(id).innerHTML = html;
  await inlineFigures();
}
main();
