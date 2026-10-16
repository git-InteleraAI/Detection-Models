import { useEffect, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

async function api(url, options) {
  const fullUrl = url.startsWith("http") ? url : `${API_BASE}${url}`;
  const res = await fetch(fullUrl, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

const enc = encodeURIComponent;
export default function App() {
  const [tasks, setTasks] = useState([]);
  const [task, setTask] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(null);
  const [conf, setConf] = useState("");
  const [samples, setSamples] = useState([]);
  const [sample, setSample] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [connecting, setConnecting] = useState(true);

  const isChange = task === "change";
  const urls = !sample
    ? {}
    : isChange
    ? { before: `${API_BASE}/api/pair-samples/${dataset}/A/${enc(sample)}`, after: `${API_BASE}/api/pair-samples/${dataset}/B/${enc(sample)}` }
    : { image: `${API_BASE}/api/samples/${dataset}/${enc(sample)}` };

  useEffect(() => {
    setConnecting(true);
    api("/api/tasks")
      .then(async (tList) => {
        setTasks(tList);
        setConnecting(false);
        if (tList.length > 0) {
          await chooseTask(tList[0].id);
        }
      })
      .catch(() => {
        setConnecting(false);
        setError("Cannot reach the backend. If the free Render service was idle, it takes ~30-50s to wake up from sleep. Please click retry below.");
      });
  }, []);

  function clearImages() {
    setSample("");
    setResult(null);
  }

  async function chooseTask(id) {
    setTask(id);
    setDataset(null);
    setDatasets([]);
    setModels([]);
    setModel(null);
    setSamples([]);
    clearImages();
    setError("");
    try {
      const list = await api(`/api/datasets?task=${id}`);
      setDatasets(list);
      if (list.length > 0) await chooseDataset(list[0].id);
    } catch (e) {
      setError(e.message);
    }
  }

  async function chooseDataset(id) {
    setDataset(id);
    setModel(null);
    setModels([]);
    setSamples([]);
    setResult(null);
    setError("");
    setSample("");
    try {
      const [m, s] = await Promise.all([api(`/api/datasets/${id}/models`), api(`/api/datasets/${id}/samples`)]);
      setModels(m);
      setSamples(s);
      if (m.length > 0) {
        setModel(m[0].id);
        setConf(String(m[0].default_conf));
      }
      if (s.length > 0) {
        setSample(s[0]);
      }
    } catch (e) {
      setError(e.message);
    }
  }

  function chooseSample(name) {
    setSample(name);
    setResult(null);
    setError("");
  }

  async function runTest() {
    setLoading(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("dataset", dataset);
    form.append("model", model);
    if (conf !== "") form.append("conf", conf);
    form.append("sample", sample);
    try {
      setResult(await api(isChange ? "/api/change/predict" : "/api/predict", { method: "POST", body: form }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const canRun = dataset && model && sample && !loading;
  const current = models.find((m) => m.id === model);
  const baseName = (sample || "result").replace(/\.[^.]+$/, "");

  return (
    <main className="page">
      <header>
        <h1>Satellite model tester</h1>
        <p>Pick a task, a dataset and a model, choose your image(s), and see what the trained model finds.</p>
      </header>

      {connecting && (
        <div className="status-banner connecting">
          <span className="spinner"></span>
          <span>Connecting to backend server... (Render free tier wakes up in ~30–50s if idle)</span>
        </div>
      )}

      <section className="step">
        <h2><span className="num">1</span> Task</h2>
        <div className="choices">
          {tasks.map((t) => (
            <button key={t.id} className={"choice" + (task === t.id ? " on" : "")} onClick={() => chooseTask(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
      </section>

      <section className="step">
        <h2><span className="num">2</span> Dataset</h2>
        {!task && <p className="hint">Choose a task first.</p>}
        <div className="choices">
          {datasets.map((d) => (
            <button key={d.id} className={"choice" + (dataset === d.id ? " on" : "")} onClick={() => chooseDataset(d.id)}>
              {d.label}
            </button>
          ))}
        </div>
      </section>

      <section className="step">
        <h2><span className="num">3</span> Model</h2>
        {!dataset && <p className="hint">Choose a dataset to see its models.</p>}
        {model && (
          <label className="conf">
            Confidence threshold
            <input type="number" min="0.01" max="0.99" step="0.01" value={conf} onChange={(e) => setConf(e.target.value)} />
            <small className="hint">Lower it to see more, weaker detections.</small>
          </label>
        )}
        <div className="choices">
          {models.map((m) => (
            <button
              key={m.id}
              className={"choice" + (model === m.id ? " on" : "")}
              onClick={() => { setModel(m.id); setConf(String(m.default_conf)); setResult(null); setError(""); }}
            >
              {m.label}
              {!m.checkpoint_found && <small className="warn">checkpoint not found</small>}
              {m.checkpoint_found && !m.ready && <small className="warn">setup pending</small>}
            </button>
          ))}
        </div>
        {current?.note && <p className="hint note">{current.note}</p>}
      </section>

      <section className="step">
        <h2><span className="num">4</span> {isChange ? "Before and after images" : "Image"}</h2>
        <div className="image-source">
          <label>
            {isChange ? "Sample pair" : "Sample image"}
            <select value={sample} disabled={!dataset} onChange={(e) => chooseSample(e.target.value)}>
              <option value="">
                {!dataset
                  ? "Choose a dataset first"
                  : samples.length
                  ? "Select a sample…"
                  : isChange
                  ? `No pairs found: put the same file name in samples/${dataset}/A and /B`
                  : `No images in samples/${dataset}`}
              </option>
              {samples.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          </label>
        </div>
        <div className="slots">
          {isChange ? (
            <>
              {urls.before && <figure><img className="preview" src={urls.before} alt="Before" /><figcaption>Before</figcaption></figure>}
              {urls.after && <figure><img className="preview" src={urls.after} alt="After" /><figcaption>After</figcaption></figure>}
            </>
          ) : (
            urls.image && <figure><img className="preview" src={urls.image} alt="Selected sample" /></figure>
          )}
        </div>
      </section>

      <button className="run" disabled={!canRun} onClick={runTest}>
        {loading ? "Running model…" : isChange ? "Detect Changes" : "Test Image"}
      </button>
      {loading && (
        <p className="hint">
          {isChange
            ? "Each image is processed in tiles, so this can take a minute or more on CPU. The first run also loads the model."
            : "The first run of each model is slower because the model has to load."}
        </p>
      )}
      {error && (
        <div className="error" role="alert">
          <div>{error}</div>
          <button className="choice" style={{ marginTop: "10px", background: "#ffffff", padding: "6px 12px", fontSize: "0.9rem" }} onClick={() => window.location.reload()}>
            🔄 Retry Connection
          </button>
        </div>
      )}

      {result && (
        <section className="results">
          <h2>Results: {result.model} on {datasets.find((d) => d.id === result.dataset)?.label}</h2>
          <div className="pair">
            {result.task === "change" ? (
              <>
                <figure><img src={urls.before} alt="Before" /><figcaption>Before</figcaption></figure>
                <figure><img src={urls.after} alt="After" /><figcaption>After</figcaption></figure>
              </>
            ) : (
              <figure><img src={urls.image} alt="Original" /><figcaption>Original</figcaption></figure>
            )}
            <figure>
              <img src={result.image} alt="Prediction" />
              <figcaption>{result.task === "change" ? "Detected changes (drawn on the after image)" : "Prediction"}</figcaption>
            </figure>
          </div>
          <dl className="stats">
            <div><dt>{result.task === "change" ? "Changes found" : "Objects found"}</dt><dd>{result.total}</dd></div>
            <div><dt>Inference time</dt><dd>{result.inference_ms} ms ({result.device})</dd></div>
            <div><dt>Confidence threshold</dt><dd>{result.confidence_threshold}</dd></div>
            <div><dt>Image size</dt><dd>{result.image_size.join(" × ")}</dd></div>
          </dl>
          {result.notes?.map((n, i) => <p className="hint" key={i}>{n}</p>)}
          {result.total === 0 ? (
            <p className="hint">
              {result.task === "change"
                ? `No changes found at threshold ${result.confidence_threshold}. Try lowering it.`
                : `No objects above the confidence threshold (${result.confidence_threshold}).` +
                  (result.max_score != null
                    ? ` The highest-scoring candidate was ${result.max_score}. Set the threshold below that to see it.`
                    : " The model found no candidates at all, even at very low scores.")}
            </p>
          ) : (
            <>
              <p className="counts">{Object.entries(result.counts).map(([label, n]) => `${label}: ${n}`).join(", ")}</p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>#</th><th>Label</th><th>Confidence</th></tr></thead>
                  <tbody>
                    {result.detections.map((d, i) => (
                      <tr key={i}><td>{i + 1}</td><td>{d.label}</td><td>{(d.confidence * 100).toFixed(1)}%</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {result.extra_images?.length > 0 && (
            <details className="diag">
              <summary>Diagnostic views</summary>
              <div className="pair">
                {result.extra_images.map((x) => (
                  <figure key={x.title}><img src={x.image} alt={x.title} /><figcaption>{x.title}</figcaption></figure>
                ))}
              </div>
            </details>
          )}
          <a className="download" href={result.image} download={`${baseName}_${result.dataset}_${model}.png`}>
            Download output image
          </a>
        </section>
      )}
    </main>
  );
}
