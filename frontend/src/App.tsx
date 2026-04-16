import { useMemo, useState } from "react";
import Dropzone from "./components/Dropzone";
import RoiSelector from "./components/RoiSelector";
import { getGrayscalePreview, runInference } from "./lib/api";

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
      <h3 style={{ marginTop: 0, marginBottom: 12, textAlign: "center" }}>{title}</h3>
      {imageUrl ? (
        <img
          src={imageUrl}
          alt={alt}
          style={{ width: "100%", borderRadius: 8, display: "block", background: "#222" }}
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
  const [modelType, setModelType] = useState("ss");
  const [modelCheckpoint, setModelCheckpoint] = useState(
    "/home/zennakamura/MasterResearch/hsi_water_detection_app/experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt"
  );
  const [temperatureJson, setTemperatureJson] = useState(
    "/home/zennakamura/MasterResearch/hsi_water_detection_app/experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json"
  );
  const [decisionThreshold, setDecisionThreshold] = useState("0.327428693347738");
  const [manifestPath, setManifestPath] = useState(
    "/home/zennakamura/MasterResearch/hsi_water_detection_app/annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv"
  );
  const [patchDatasetPath, setPatchDatasetPath] = useState(
    "/home/zennakamura/MasterResearch/hsi_water_detection_app/datasets/processed/wetness_pretrain_v3"
  );
  const [splitPolicy, setSplitPolicy] = useState("spatial_block(block_size=224)");

  const [spatCheckpoint, setSpatCheckpoint] = useState(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth"
  );
  const [specCheckpoint, setSpecCheckpoint] = useState(
    "/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth"
  );

  const [previewBand, setPreviewBand] = useState("10");
  const [previewWavelength, setPreviewWavelength] = useState("");

  const [roi, setRoi] = useState<{
    row_start: number;
    row_stop: number;
    col_start: number;
    col_stop: number;
  } | null>(null);

  const [previewResult, setPreviewResult] = useState<any>(null);
  const [result, setResult] = useState<any>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loadingInference, setLoadingInference] = useState(false);

  const hsiPreviewName = useMemo(() => hsiFile?.name ?? "No file selected", [hsiFile]);
  const wavelengthPreviewName = useMemo(() => wavelengthFile?.name ?? "No file selected", [wavelengthFile]);

  const handlePreview = async () => {
    if (!hsiFile) {
      alert("HSI file is required");
      return;
    }

    const formData = new FormData();
    formData.append("hsi_file", hsiFile);
    if (wavelengthFile) formData.append("wavelength_file", wavelengthFile);

    formData.append("sensor", sensor);
    if (previewBand.trim() !== "") formData.append("preview_band", previewBand.trim());
    if (previewWavelength.trim() !== "") formData.append("preview_wavelength", previewWavelength.trim());

    setLoadingPreview(true);
    try {
      const res = await getGrayscalePreview(formData);
      setPreviewResult(res);
    } catch (err) {
      console.error(err);
      alert("Preview failed");
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleRun = async () => {
    if (!hsiFile) {
      alert("HSI file is required");
      return;
    }
    if (!roi) {
      alert("Select ROI first");
      return;
    }

    const formData = new FormData();
    formData.append("hsi_file", hsiFile);
    if (wavelengthFile) formData.append("wavelength_file", wavelengthFile);

    formData.append("sensor", sensor);
    formData.append("device", device);
    formData.append("model_type", modelType);
    if (modelCheckpoint.trim() !== "") formData.append("model_checkpoint", modelCheckpoint.trim());
    if (temperatureJson.trim() !== "") formData.append("temperature_json", temperatureJson.trim());
    if (decisionThreshold.trim() !== "") formData.append("decision_threshold", decisionThreshold.trim());
    if (manifestPath.trim() !== "") formData.append("manifest_path", manifestPath.trim());
    if (patchDatasetPath.trim() !== "") formData.append("patch_dataset_path", patchDatasetPath.trim());
    if (splitPolicy.trim() !== "") formData.append("split_policy", splitPolicy.trim());
    formData.append("spat_checkpoint", spatCheckpoint);
    formData.append("spec_checkpoint", specCheckpoint);

    formData.append("row_start", String(roi.row_start));
    formData.append("row_stop", String(roi.row_stop));
    formData.append("col_start", String(roi.col_start));
    formData.append("col_stop", String(roi.col_stop));
    formData.append("patch_size", "64");
    formData.append("stride", "32");

    setLoadingInference(true);
    setResult(null);
    try {
      const res = await runInference(formData);
      setResult(res);
    } catch (err) {
      console.error(err);
      alert("Inference failed");
    } finally {
      setLoadingInference(false);
    }
  };

  const urls = result?.urls ?? {};
  const previewUrls = previewResult?.urls ?? {};

  return (
    <div style={{ maxWidth: 1500, margin: "0 auto", padding: 24, fontFamily: "sans-serif" }}>
      <h1 style={{ textAlign: "center" }}>HSI Water Detection UI</h1>

      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24 }}>
        <div>
          <Dropzone label="HSI file" onFileSelected={setHsiFile} selectedFile={hsiFile} />
          <div style={{ marginBottom: 12, fontSize: "0.9rem", color: "#555" }}>{hsiPreviewName}</div>

          <Dropzone label="Wavelength sidecar (optional)" onFileSelected={setWavelengthFile} selectedFile={wavelengthFile} />
          <div style={{ marginBottom: 12, fontSize: "0.9rem", color: "#555" }}>{wavelengthPreviewName}</div>

          <div style={{ marginBottom: 12 }}>
            <label>Sensor: </label>
            <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
              <option value="hyperion">hyperion</option>
              <option value="hisui">hisui</option>
              <option value="generic">generic</option>
              <option value="auto">auto</option>
            </select>
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Device: </label>
            <select value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Model type: </label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="ss">ss</option>
              <option value="sa">sa</option>
            </select>
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Fine-tuned model checkpoint:</label>
            <input
              value={modelCheckpoint}
              onChange={(e) => setModelCheckpoint(e.target.value)}
              placeholder="Optional. Leave blank to use backbone-only inference."
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Calibration JSON:</label>
            <input
              value={temperatureJson}
              onChange={(e) => setTemperatureJson(e.target.value)}
              placeholder="Optional temperature scaling json"
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Decision threshold:</label>
            <input value={decisionThreshold} onChange={(e) => setDecisionThreshold(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Spatial checkpoint:</label>
            <input value={spatCheckpoint} onChange={(e) => setSpatCheckpoint(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Spectral checkpoint:</label>
            <input value={specCheckpoint} onChange={(e) => setSpecCheckpoint(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Manifest provenance:</label>
            <input value={manifestPath} onChange={(e) => setManifestPath(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Patch dataset provenance:</label>
            <input value={patchDatasetPath} onChange={(e) => setPatchDatasetPath(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Split policy provenance:</label>
            <input value={splitPolicy} onChange={(e) => setSplitPolicy(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Preview band index:</label>
            <input value={previewBand} onChange={(e) => setPreviewBand(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Preview wavelength (optional):</label>
            <input value={previewWavelength} onChange={(e) => setPreviewWavelength(e.target.value)} style={{ width: "100%" }} />
          </div>

          <button onClick={handlePreview} disabled={loadingPreview} style={{ padding: "10px 18px", marginRight: 8 }}>
            {loadingPreview ? "Loading Preview..." : "Load Grayscale Preview"}
          </button>

          <button onClick={handleRun} disabled={loadingInference} style={{ padding: "10px 18px" }}>
            {loadingInference ? "Running..." : "Run Inference with ROI"}
          </button>

          {previewResult && (
            <div style={{ marginTop: 16, fontSize: "0.92rem" }}>
              <div><strong>Preview band:</strong> {previewResult.preview_band_index}</div>
              <div><strong>Preview wavelength:</strong> {String(previewResult.preview_wavelength_nm ?? "(none)")}</div>
              <div><strong>Preview image:</strong> {previewResult.image_width} x {previewResult.image_height}</div>
            </div>
          )}

          {roi && (
            <div style={{ marginTop: 16, fontSize: "0.92rem" }}>
              <div><strong>ROI row:</strong> {roi.row_start} - {roi.row_stop}</div>
              <div><strong>ROI col:</strong> {roi.col_start} - {roi.col_stop}</div>
            </div>
          )}
        </div>

        <div>
          <RoiSelector
            imageUrl={previewUrls.grayscale_preview_png}
            imageWidth={previewResult?.image_width}
            imageHeight={previewResult?.image_height}
            onApply={setRoi}
          />

          <h2 style={{ marginTop: 24 }}>Results</h2>

          {!result && <div>No result yet.</div>}

          {result && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
                <ResultCard title="Pseudo Color" imageUrl={urls.pseudocolor_png} alt="Pseudo color image" />
                <ResultCard title="Water Detection Overlay" imageUrl={urls.probability_overlay_png} alt="Water detection over pseudo color" />
                <ResultCard title="Spatial Attention Overlay" imageUrl={urls.spatial_attention_overlay_png} alt="Spatial attention overlay" />
                <ResultCard title="Spectral Attention" imageUrl={urls.spectral_attention_png} alt="Spectral attention chart" />
              </div>

              <details style={{ marginTop: 20 }}>
                <summary>Additional files</summary>
                <pre style={{ whiteSpace: "pre-wrap", background: "#f5f5f5", padding: 12, borderRadius: 8, maxHeight: 260, overflow: "auto", fontSize: "0.82rem" }}>
                  {JSON.stringify(result.urls, null, 2)}
                </pre>
              </details>

              <details style={{ marginTop: 12 }}>
                <summary>Model provenance</summary>
                <pre style={{ whiteSpace: "pre-wrap", background: "#f5f5f5", padding: 12, borderRadius: 8, maxHeight: 260, overflow: "auto", fontSize: "0.82rem" }}>
                  {JSON.stringify(result.model_provenance, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
