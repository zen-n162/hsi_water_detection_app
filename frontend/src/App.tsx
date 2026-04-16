import { useMemo, useState } from 'react';
import SidebarControls from './components/SidebarControls';
import RoiPage from './components/RoiPage';
import ResultsPage from './components/ResultsPage';
import ModelInfoPanel from './components/ModelInfoPanel';
import {
  loadPreview,
  runInference,
  type PreviewResponse,
  type InferenceResponse,
} from './lib/api';

type RoiRect = {
  rowStart: number;
  rowStop: number;
  colStart: number;
  colStop: number;
};

const DEFAULT_DEPLOY_CONFIG =
  '/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_calibrated.json';

export default function App() {
  const [page, setPage] = useState<'roi' | 'results'>('roi');

  const [sensor, setSensor] = useState('hyperion');
  const [device, setDevice] = useState('cuda');
  const [modelType, setModelType] = useState('ss');

  const [deployConfigPath, setDeployConfigPath] = useState(DEFAULT_DEPLOY_CONFIG);
  const [inputPath, setInputPath] = useState('');
  const [hsiFile, setHsiFile] = useState<File | null>(null);
  const [wavelengthFile, setWavelengthFile] = useState<File | null>(null);

  const [modelCheckpoint, setModelCheckpoint] = useState('');
  const [temperatureJson, setTemperatureJson] = useState('');
  const [threshold, setThreshold] = useState<string>('');

  const [previewBandIndex, setPreviewBandIndex] = useState('10');
  const [previewWavelength, setPreviewWavelength] = useState('');

  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [inference, setInference] = useState<InferenceResponse | null>(null);

  const [roi, setRoi] = useState<RoiRect | null>(null);

  const [busyPreview, setBusyPreview] = useState(false);
  const [busyInference, setBusyInference] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const canRunInference = useMemo(() => {
    return Boolean(preview?.grayscalePreviewUrl && roi);
  }, [preview, roi]);

  async function handleLoadPreview() {
    setErrorMessage('');
    setBusyPreview(true);
    try {
      const result = await loadPreview({
        sensor,
        device,
        model_type: modelType,
        deploy_config_path: deployConfigPath || undefined,
        input_path: inputPath || undefined,
        hsi_file: hsiFile,
        wavelength_file: wavelengthFile,
        preview_band_index: previewBandIndex ? Number(previewBandIndex) : undefined,
        preview_wavelength: previewWavelength ? Number(previewWavelength) : undefined,
      });

      setPreview(result);
      setInference(null);
      setPage('roi');

      const width = result.imageWidth || 0;
      const height = result.imageHeight || 0;

      if (width > 0 && height > 0) {
        const rowStart = Math.floor(height * 0.20);
        const rowStop = Math.min(height, rowStart + 64);
        const colStart = Math.floor(width * 0.20);
        const colStop = Math.min(width, colStart + 64);
        setRoi({ rowStart, rowStop, colStart, colStop });
      }
    } catch (e: any) {
      setErrorMessage(e?.message || 'Failed to load preview.');
    } finally {
      setBusyPreview(false);
    }
  }

async function handleRunInference() {
  if (!roi) return;
  setErrorMessage('');
  setBusyInference(true);

  try {
    const result = await runInference({
      sensor,
      device,
      model_type: modelType,
      deploy_config_path: deployConfigPath || undefined,
      input_path: inputPath || undefined,
      hsi_file: hsiFile,
      wavelength_file: wavelengthFile,
      model_checkpoint: modelCheckpoint || undefined,
      temperature_json: temperatureJson || undefined,
      threshold: threshold === '' ? undefined : Number(threshold),
      preview_band_index: previewBandIndex ? Number(previewBandIndex) : undefined,
      preview_wavelength: previewWavelength ? Number(previewWavelength) : undefined,
      row_start: roi.rowStart,
      row_stop: roi.rowStop,
      col_start: roi.colStart,
      col_stop: roi.colStop,
    });

    if (
      !result.pseudocolorUrl &&
      !result.probabilityOverlayUrl &&
      !result.spatialAttentionOverlayUrl &&
      !result.spectralAttentionUrl
    ) {
      console.warn('Inference succeeded but no renderable result URLs were returned.', result.raw);
    }

    setInference(result);
    setPage('results');
  } catch (e: any) {
    setErrorMessage(e?.message || 'Failed to run inference.');
  } finally {
    setBusyInference(false);
  }
}

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>HSI Water Detection UI</h1>
      </header>

      <div className="workspace-layout">
        <aside className="sidebar">
          <SidebarControls
            sensor={sensor}
            setSensor={setSensor}
            device={device}
            setDevice={setDevice}
            modelType={modelType}
            setModelType={setModelType}
            deployConfigPath={deployConfigPath}
            setDeployConfigPath={setDeployConfigPath}
            inputPath={inputPath}
            setInputPath={setInputPath}
            hsiFile={hsiFile}
            setHsiFile={setHsiFile}
            wavelengthFile={wavelengthFile}
            setWavelengthFile={setWavelengthFile}
            modelCheckpoint={modelCheckpoint}
            setModelCheckpoint={setModelCheckpoint}
            temperatureJson={temperatureJson}
            setTemperatureJson={setTemperatureJson}
            threshold={threshold}
            setThreshold={setThreshold}
            previewBandIndex={previewBandIndex}
            setPreviewBandIndex={setPreviewBandIndex}
            previewWavelength={previewWavelength}
            setPreviewWavelength={setPreviewWavelength}
            onLoadPreview={handleLoadPreview}
            onRunInference={handleRunInference}
            busyPreview={busyPreview}
            busyInference={busyInference}
            canRunInference={canRunInference}
            currentPage={page}
            onGoRoi={() => setPage('roi')}
            onGoResults={() => setPage('results')}
            hasResults={Boolean(inference)}
          />

          {errorMessage ? <div className="error-box">{errorMessage}</div> : null}

          <ModelInfoPanel preview={preview} inference={inference} />
        </aside>

        <main className="page-container">
          {page === 'roi' ? (
            <RoiPage
              preview={preview}
              roi={roi}
              setRoi={setRoi}
              onRunInference={handleRunInference}
              busyInference={busyInference}
              canRunInference={canRunInference}
            />
          ) : (
            <ResultsPage
              inference={inference}
              onBackToRoi={() => setPage('roi')}
            />
          )}
        </main>
      </div>
    </div>
  );
}
