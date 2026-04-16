import { useEffect, useMemo, useState } from "react";
import Dropzone from "./components/Dropzone";
import RoiSelector from "./components/RoiSelector";
import { getDeployConfig, getGrayscalePreview, runInference } from "./lib/api";

type ResultCardProps = {
  title: string;
  imageUrl?: string;
  alt: string;
};

type DeployConfigResponse = {
  deploy_config_path: string;
  deploy_config: Record<string, any>;
  resolved: Record<string, any>;
  reference_models?: Record<string, any>;
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

function InfoBlock({ title, value }: { title: string; value: string }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <strong>{title}:</strong> <span style={{ wordBreak: "break-all" }}>{value}</span>
    </div>
  );
}

function appendOverride(formData: FormData, name: string, value: string, defaultValue?: string | null) {
  const trimmed = value.trim();
  const defaultTrimmed = (defaultValue ?? "").trim();
  if (trimmed !== "" && trimmed !== defaultTrimmed) {
    formData.append(name, trimmed);
  }
}

export default function App() {
  const [deployInfo, setDeployInfo] = useState<DeployConfigResponse | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [loadingDeploy, setLoadingDeploy] = useState(true);

  const [hsiFile, setHsiFile] = useState<File | null>(null);
  const [wavelengthFile, setWavelengthFile] = useState<File | null>(null);
  const [sensor, setSensor] = useState("hyperion");
  const [device, setDevice] = useState("cuda");
  const [modelType, setModelType] = useState("ss");
  const [modelCheckpoint, setModelCheckpoint] = useState("");
  const [temperatureJson, setTemperatureJson] = useState("");
  const [decisionThreshold, setDecisionThreshold] = useState("");
  const [manifestPath, setManifestPath] = useState("");
  const [patchDatasetPath, setPatchDatasetPath] = useState("");
  const [splitPolicy, setSplitPolicy] = useState("");
  const [spatCheckpoint, setSpatCheckpoint] = useState("");
  const [specCheckpoint, setSpecCheckpoint] = useState("");
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

  useEffect(() => {
    const load = async () => {
      setLoadingDeploy(true);
      try {
        const deploy = (await getDeployConfig()) as DeployConfigResponse;
        setDeployInfo(deploy);
        setDeployError(null);
        setSensor(String(deploy.resolved.sensor ?? "hyperion"));
        setModelType(String(deploy.resolved.model_type ?? "ss"));
        setModelCheckpoint(String(deploy.resolved.model_checkpoint ?? ""));
        setTemperatureJson(String(deploy.resolved.temperature_json ?? ""));
        setDecisionThreshold(String(deploy.resolved.threshold ?? ""));
        setManifestPath(String(deploy.resolved.manifest_path ?? ""));
        setPatchDatasetPath(String(deploy.resolved.dataset_path ?? ""));
        setSplitPolicy(String(deploy.resolved.split_policy ?? ""));
        setSpatCheckpoint(String(deploy.resolved.spat_checkpoint ?? ""));
        setSpecCheckpoint(String(deploy.resolved.spec_checkpoint ?? ""));
      } catch (err) {
        console.error(err);
        setDeployError("Failed to load deploy config from backend.");
      } finally {
        setLoadingDeploy(false);
      }
    };
    void load();
  }, []);

  const deployDefaults = deployInfo?.resolved ?? {};
  const deployConfigPath = deployInfo?.deploy_config_path ?? "";

  const hsiPreviewName = useMemo(() => hsiFile?.name ?? "No file selected", [hsiFile]);
  const wavelengthPreviewName = useMemo(() => wavelengthFile?.name ?? "No file selected", [wavelengthFile]);

  const activeModelInfo = useMemo(
    () => ({
      runName: result?.resolved_run_name ?? deployDefaults.run_name ?? deployInfo?.deploy_config?.run_name ?? "",
      modelCheckpoint: modelCheckpoint.trim() || String(deployDefaults.model_checkpoint ?? ""),
      threshold: decisionThreshold.trim() || String(deployDefaults.threshold ?? ""),
      calibrationFile: temperatureJson.trim() || String(deployDefaults.temperature_json ?? ""),
      calibrationOn: Boolean((temperatureJson.trim() || String(deployDefaults.temperature_json ?? "")).trim()),
      manifestPath: manifestPath.trim() || String(deployDefaults.manifest_path ?? ""),
      datasetPath: patchDatasetPath.trim() || String(deployDefaults.dataset_path ?? ""),
      splitPolicy: splitPolicy.trim() || String(deployDefaults.split_policy ?? ""),
    }),
    [decisionThreshold, deployDefaults, deployInfo?.deploy_config?.run_name, manifestPath, modelCheckpoint, patchDatasetPath, result?.resolved_run_name, splitPolicy, temperatureJson]
  );

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
    if (deployConfigPath) formData.append("deploy_config_path", deployConfigPath);

    appendOverride(formData, "model_type", modelType, String(deployDefaults.model_type ?? ""));
    appendOverride(formData, "model_checkpoint", modelCheckpoint, String(deployDefaults.model_checkpoint ?? ""));
    appendOverride(formData, "temperature_json", temperatureJson, String(deployDefaults.temperature_json ?? ""));
    appendOverride(formData, "decision_threshold", decisionThreshold, String(deployDefaults.threshold ?? ""));
    appendOverride(formData, "manifest_path", manifestPath, String(deployDefaults.manifest_path ?? ""));
    appendOverride(formData, "patch_dataset_path", patchDatasetPath, String(deployDefaults.dataset_path ?? ""));
    appendOverride(formData, "split_policy", splitPolicy, String(deployDefaults.split_policy ?? ""));
    appendOverride(formData, "spat_checkpoint", spatCheckpoint, String(deployDefaults.spat_checkpoint ?? ""));
    appendOverride(formData, "spec_checkpoint", specCheckpoint, String(deployDefaults.spec_checkpoint ?? ""));

    formData.append("row_start", String(roi.row_start));
    formData.append("row_stop", String(roi.row_stop));
    formData.append("col_start", String(roi.col_start));
    formData.append("col_stop", String(roi.col_stop));

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

      <div style={{ marginBottom: 20, padding: 16, borderRadius: 12, background: "#f7f7f7", border: "1px solid #ddd" }}>
        <h2 style={{ marginTop: 0 }}>Deploy Default</h2>
        {loadingDeploy && <div>Loading deploy config...</div>}
        {deployError && <div style={{ color: "#b00020" }}>{deployError}</div>}
        {!loadingDeploy && !deployError && deployInfo && (
          <>
            <InfoBlock title="Deploy config" value={deployInfo.deploy_config_path} />
            <InfoBlock title="Run name" value={String(deployInfo.deploy_config.run_name ?? "")} />
            <InfoBlock title="Model checkpoint" value={String(deployDefaults.model_checkpoint ?? "")} />
            <InfoBlock title="Threshold" value={String(deployDefaults.threshold ?? "")} />
            <InfoBlock title="Calibration" value={deployDefaults.temperature_json ? `on (${deployDefaults.temperature_json})` : "off"} />
            <InfoBlock title="Manifest" value={String(deployDefaults.manifest_path ?? "")} />
            <InfoBlock title="Dataset" value={String(deployDefaults.dataset_path ?? "")} />
            <InfoBlock title="Split policy" value={String(deployDefaults.split_policy ?? "")} />
            {deployInfo.reference_models?.baseline_v3_block224 && (
              <details style={{ marginTop: 10 }}>
                <summary>Baseline reference path</summary>
                <pre style={{ whiteSpace: "pre-wrap", background: "#fff", padding: 12, borderRadius: 8, maxHeight: 220, overflow: "auto", fontSize: "0.82rem" }}>
                  {JSON.stringify(deployInfo.reference_models.baseline_v3_block224, null, 2)}
                </pre>
              </details>
            )}
          </>
        )}
      </div>

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
            <label>Model type:</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="ss">ss</option>
              <option value="sa">sa</option>
            </select>
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Deploy config path:</label>
            <input value={deployConfigPath} readOnly style={{ width: "100%", background: "#f5f5f5" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Fine-tuned model checkpoint override:</label>
            <input value={modelCheckpoint} onChange={(e) => setModelCheckpoint(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Calibration JSON override:</label>
            <input value={temperatureJson} onChange={(e) => setTemperatureJson(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Decision threshold override:</label>
            <input value={decisionThreshold} onChange={(e) => setDecisionThreshold(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Manifest provenance override:</label>
            <input value={manifestPath} onChange={(e) => setManifestPath(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Patch dataset provenance override:</label>
            <input value={patchDatasetPath} onChange={(e) => setPatchDatasetPath(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Split policy provenance override:</label>
            <input value={splitPolicy} onChange={(e) => setSplitPolicy(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Spatial checkpoint override:</label>
            <input value={spatCheckpoint} onChange={(e) => setSpatCheckpoint(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Spectral checkpoint override:</label>
            <input value={specCheckpoint} onChange={(e) => setSpecCheckpoint(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Preview band index:</label>
            <input value={previewBand} onChange={(e) => setPreviewBand(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label>Preview wavelength (optional):</label>
            <input value={previewWavelength} onChange={(e) => setPreviewWavelength(e.target.value)} style={{ width: "100%" }} />
          </div>

          <div style={{ marginTop: 16, padding: 12, borderRadius: 10, background: "#f8fbff", border: "1px solid #d7e7ff", fontSize: "0.92rem" }}>
            <div style={{ marginBottom: 6 }}><strong>Current model info</strong></div>
            <div><strong>Run name:</strong> {activeModelInfo.runName}</div>
            <div><strong>Checkpoint:</strong> {activeModelInfo.modelCheckpoint}</div>
            <div><strong>Threshold:</strong> {activeModelInfo.threshold}</div>
            <div><strong>Calibration:</strong> {activeModelInfo.calibrationOn ? "on" : "off"}</div>
            <div><strong>Manifest:</strong> {activeModelInfo.manifestPath}</div>
            <div><strong>Dataset:</strong> {activeModelInfo.datasetPath}</div>
            <div><strong>Split policy:</strong> {activeModelInfo.splitPolicy}</div>
          </div>

          <div style={{ marginTop: 16 }}>
            <button onClick={handlePreview} disabled={loadingPreview} style={{ padding: "10px 18px", marginRight: 8 }}>
              {loadingPreview ? "Loading Preview..." : "Load Grayscale Preview"}
            </button>

            <button onClick={handleRun} disabled={loadingInference || loadingDeploy} style={{ padding: "10px 18px" }}>
              {loadingInference ? "Running..." : "Run Inference with ROI"}
            </button>
          </div>

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

              <div style={{ marginTop: 20, padding: 14, borderRadius: 10, background: "#f8fbff", border: "1px solid #d7e7ff" }}>
                <div style={{ marginBottom: 6 }}><strong>Resolved inference settings</strong></div>
                <div><strong>Run name:</strong> {String(result.resolved_run_name ?? "")}</div>
                <div><strong>Checkpoint:</strong> {String(result.resolved_model_checkpoint ?? "")}</div>
                <div><strong>Threshold:</strong> {String(result.resolved_threshold ?? "")}</div>
                <div><strong>Calibration JSON:</strong> {String(result.resolved_temperature_json ?? "")}</div>
                <div><strong>Manifest:</strong> {String(result.resolved_manifest ?? "")}</div>
                <div><strong>Dataset:</strong> {String(result.resolved_dataset ?? "")}</div>
                <div><strong>Deploy config:</strong> {String(result.deploy_config_path ?? "")}</div>
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
