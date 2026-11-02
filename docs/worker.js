// Runs the demo track off the main thread, so the wheels stay smooth while Python trains.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const FILES = ["harness.py", "train.py", "dev.bin", "train.bin.gz", "odometer/__init__.py", "odometer/_engine.py", "odometer/_data.py"];

self.onmessage = async () => {
  try {
    self.postMessage({ type: "status", text: "Loading Python and NumPy" });
    importScripts(PYODIDE + "pyodide.js");
    const py = await loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage("numpy", { messageCallback: () => {} });
    self.postMessage({ type: "status", text: "Fetching the record's code and the excerpt" });
    py.FS.mkdirTree("/app/odometer");
    for (const name of FILES) {
      const r = await fetch("demo/" + name);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      py.FS.writeFile("/app/" + name, new Uint8Array(await r.arrayBuffer()));
    }
    let steps = 0;
    py.globals.set("on_step", (used, loss) => {
      steps += 1;
      self.postMessage({ type: "step", used, loss: loss ?? null, steps });
    });
    py.globals.set("on_token", (text) => self.postMessage({ type: "token", text }));
    self.postMessage({ type: "status", text: "Training" });
    const out = py.runPython(`
import json, sys
sys.path.insert(0, "/app")
import harness
json.dumps(harness.run("/app/train.py", "/app/train.bin.gz", "/app/dev.bin", on_step, on_token))
`);
    self.postMessage({ type: "done", report: JSON.parse(out) });
  } catch (e) {
    self.postMessage({ type: "error", text: String(e.message || e).slice(0, 400) });
  }
};
