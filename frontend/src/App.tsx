import { useState } from "react";
import Dropzone from "./components/Dropzone";
import { runInference } from "./lib/api";

export default function App() {
  const [hsiFile, setHsiFile] = useState<File | null>(null);
  const [wavelengthFile, setWavelengthFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

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
    formData.append("sensor", "hyperion");
    formData.append("device", "cuda");
    formData.append("xmin", "563000");
    formData.append("ymin", "1405000");
    formData.append("xmax", "567000");
    formData.append("ymax", "1409000");

    setLoading(true);
    try {
      const res = await runInference(formData);
      setResult(res);
    } catch (e) {
      console.error(e);
      alert("Inference failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "24px", fontFamily: "sans-serif" }}>
      <h1>HSI Water Detection UI</h1>

      <Dropzone label="HSI file" onFileSelected={setHsiFile} />
      <Dropzone label="Wavelength sidecar (optional)" onFileSelected={setWavelengthFile} />

      <button onClick={handleRun} disabled={loading}>
        {loading ? "Running..." : "Run Inference"}
      </button>

      {result && (
        <pre style={{ marginTop: "24px", whiteSpace: "pre-wrap" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
