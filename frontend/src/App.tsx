import { useMemo, useState } from "react";
import Dropzone from "./components/Dropzone";
import { runInference } from "./lib/api";

type ResultCardProps = {
  title: string;
  imageUrl?: string;
  alt: string;
};

function ResultCard({ title, imageUrl, alt }: ResultCardProps) {
  return (
    <div
      style={{
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "12px",
        background: "#111",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: "12px", textAlign: "center" }}>{title}</h3>
      {imageUrl ? (
        <img
          src={imageUrl}
          alt={alt}
          style={{
            width: "100%",
            borderRadius: "8px",
            display: "block",
            background: "#222",
          }}
        />
      ) : (
        <div style={{ color: "#bbb", textAlign: "center", padding: "48px 0" }}>{alt}</div>
      )}
    </div>
  );
}

export default function App() {
  const [hsiFile, setHsiFile] = useState<File | null>(null);
  const [wavelengthFile, setWavelengthFile] = useState<File | null>(null);
  const [sensor, setSensor] = useState("hyperion");
  const [device, setDevice] = useState("cuda");
  const [xmin, setXmin] = useState("563000");
  const [ymin, setYmin] = useState("1405000");
  const [xmax, setXmax] = useState("567000");
  const [ymax, setYmax] = useState("1409000");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const hsiPreviewName = useMemo(() => hsiFile?.name ?? "No file selected", [hsiFile]);
  const wavelengthPreviewName = useMemo(() => wavelengthFile?.name ?? "No file selected", [wavelengthFile]);

  const handleRun = async () => {
    if (!hsiFile) {
      alert("HSI file is required");
      return;
    }

    const formData = new FormData();
    formData.append("hsi_file", hsiFile);

    if (wavelengthFile) {
      formData.append("wavelength_file", wavelengthFile);
    }

    formData.append("sensor", sensor);
    formData.append("device", device);
    formData.append("xmin", xmin);
    formData.append("ymin", ymin);
    formData.append("xmax", xmax);
    formData.append("ymax", ymax);
    formData.append("patch_size", "64");
    formData.append("stride", "32");

    setLoading(true);
    setResult(null);

    try {
      const res = await runInference(formData);
      setResult(res);
    } catch (error) {
      console.error(error);
      alert("Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const urls = result?.urls ?? {};

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "24px", fontFamily: "sans-serif" }}>
      <h1 style={{ textAlign: "center" }}>HSI Water Detection UI</h1>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "24px" }}>
        <div>
          <Dropzone
            label="HSI file"
            onFileSelected={setHsiFile}
            selectedFile={hsiFile}
          />
          <div style={{ marginBottom: "12px", fontSize: "0.9rem", color: "#555" }}>
            {hsiPreviewName}
          </div>

          <Dropzone
            label="Wavelength sidecar (optional)"
            onFileSelected={setWavelengthFile}
            selectedFile={wavelengthFile}
          />
          <div style={{ marginBottom: "12px", fontSize: "0.9rem", color: "#555" }}>
            {wavelengthPreviewName}
          </div>

          <div style={{ marginBottom: "12px" }}>
            <label>Sensor: </label>
            <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
              <option value="hyperion">hyperion</option>
              <option value="hisui">hisui</option>
              <option value="generic">generic</option>
              <option value="auto">auto</option>
            </select>
          </div>

          <div style={{ marginBottom: "12px" }}>
            <label>Device: </label>
            <select value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "16px" }}>
            <input value={xmin} onChange={(e) => setXmin(e.target.value)} placeholder="xmin" />
            <input value={xmax} onChange={(e) => setXmax(e.target.value)} placeholder="xmax" />
            <input value={ymin} onChange={(e) => setYmin(e.target.value)} placeholder="ymin" />
            <input value={ymax} onChange={(e) => setYmax(e.target.value)} placeholder="ymax" />
          </div>

          <button onClick={handleRun} disabled={loading} style={{ padding: "10px 18px" }}>
            {loading ? "Running..." : "Run Inference"}
          </button>

          {result && (
            <div style={{ marginTop: "20px", fontSize: "0.92rem" }}>
              <div><strong>Success:</strong> {String(result.ok)}</div>
              <div><strong>Output dir:</strong> {result.output_dir}</div>
            </div>
          )}
        </div>

        <div>
          <h2>Results</h2>

          {!result && <div>No result yet.</div>}

          {result && (
            <>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "16px",
                  alignItems: "start",
                }}
              >
                <ResultCard
                  title="Pseudo Color"
                  imageUrl={urls.pseudocolor_png}
                  alt="Pseudo color preview"
                />

                <ResultCard
                  title="Water Detection"
                  imageUrl={urls.probability_map_png}
                  alt="Water detection probability map"
                />

                <ResultCard
                  title="Spatial Attention Overlay"
                  imageUrl={urls.spatial_attention_overlay_png}
                  alt="Spatial attention overlay"
                />

                <ResultCard
                  title="Spectral Attention"
                  imageUrl={urls.spectral_attention_png}
                  alt="Spectral attention chart"
                />
              </div>

              <details style={{ marginTop: "20px" }}>
                <summary>Additional files</summary>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    background: "#f5f5f5",
                    padding: "12px",
                    borderRadius: "8px",
                    maxHeight: "260px",
                    overflow: "auto",
                    fontSize: "0.82rem",
                  }}
                >
                  {JSON.stringify(result.urls, null, 2)}
                </pre>
              </details>

              <details style={{ marginTop: "20px" }}>
                <summary>Stdout / Stderr</summary>

                <h4>Stdout</h4>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    background: "#f5f5f5",
                    padding: "12px",
                    borderRadius: "8px",
                    maxHeight: "260px",
                    overflow: "auto",
                  }}
                >
                  {result.stdout}
                </pre>

                <h4>Stderr</h4>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    background: "#f5f5f5",
                    padding: "12px",
                    borderRadius: "8px",
                    maxHeight: "260px",
                    overflow: "auto",
                  }}
                >
                  {result.stderr}
                </pre>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
