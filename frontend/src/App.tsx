import { useEffect, useMemo, useState } from 'react';
import SidebarControls from './components/SidebarControls';
import RoiPage from './components/RoiPage';
import ResultsPage from './components/ResultsPage';
import ModelInfoPanel from './components/ModelInfoPanel';
import {
  ApiError,
  loadDeployConfig,
  loadPreview,
  runInference,
  type DeployConfigResponse,
  type PreviewResponse,
  type InferenceResponse,
} from './lib/api';
import { runtimeConfig } from './lib/runtime';

type RoiRect = {
  rowStart: number;
  rowStop: number;
  colStart: number;
  colStop: number;
};

type UiError = {
  title: string;
  detail: string;
  hint?: string;
  status?: number;
};

function withModeHint(hint?: string): string | undefined {
  if (runtimeConfig.appMode !== 'local_gpu_web') return hint;
  const tunnelHint =
    'The backend research PC or its secure tunnel may be offline. Confirm FastAPI, the CUDA runtime, and Cloudflare Tunnel or Tailscale Funnel are running.';
  if (!hint) return tunnelHint;
  if (hint.includes(tunnelHint)) return hint;
  return `${hint} ${tunnelHint}`;
}

function toUiError(error: unknown, fallbackTitle: string, fallbackDetail: string): UiError {
  if (error instanceof ApiError) {
    return {
      title: fallbackTitle,
      detail: error.message || fallbackDetail,
      hint: withModeHint(error.hint),
      status: error.status,
    };
  }

  if (error instanceof Error) {
    return {
      title: fallbackTitle,
      detail: error.message || fallbackDetail,
      hint: withModeHint(),
    };
  }

  return {
    title: fallbackTitle,
    detail: fallbackDetail,
    hint: withModeHint(),
  };
}

export default function App() {
  const [page, setPage] = useState<'roi' | 'results'>('roi');

  const [sensor, setSensor] = useState(runtimeConfig.defaultSensor);
  const [device, setDevice] = useState(runtimeConfig.defaultDevice);
  const [modelType, setModelType] = useState(runtimeConfig.defaultModelType);

  const [deployConfigPath, setDeployConfigPath] = useState('');
  const [inputPath, setInputPath] = useState('');
  const [hsiFile, setHsiFile] = useState<File | null>(null);
  const [wavelengthFile, setWavelengthFile] = useState<File | null>(null);

  const [modelCheckpoint, setModelCheckpoint] = useState('');
  const [temperatureJson, setTemperatureJson] = useState('');
  const [threshold, setThreshold] = useState('');

  const [previewBandIndex, setPreviewBandIndex] = useState('10');
  const [previewWavelength, setPreviewWavelength] = useState('');

  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [deployInfo, setDeployInfo] = useState<DeployConfigResponse | null>(null);

  const [roi, setRoi] = useState<RoiRect | null>(null);

  const [busyPreview, setBusyPreview] = useState(false);
  const [busyInference, setBusyInference] = useState(false);
  const [loadingDeployInfo, setLoadingDeployInfo] = useState(true);
  const [error, setError] = useState<UiError | null>(null);

  const canRunInference = useMemo(() => {
    return Boolean(preview?.grayscalePreviewUrl && roi);
  }, [preview, roi]);

  useEffect(() => {
    let cancelled = false;

    async function bootstrapDeployConfig() {
      setLoadingDeployInfo(true);
      try {
        const info = await loadDeployConfig();
        if (cancelled) return;

        setDeployInfo(info);
        setDeployConfigPath(info.deployConfigRelative || info.deployConfigPath || '');

        if (info.resolved?.sensor) {
          setSensor(String(info.resolved.sensor));
        }
        if (info.resolved?.model_type) {
          setModelType(String(info.resolved.model_type));
        }
        if (info.resolved?.threshold !== undefined && info.resolved?.threshold !== null) {
          setThreshold(String(info.resolved.threshold));
        }
        if (info.runtime?.default_device) {
          setDevice(String(info.runtime.default_device));
        }
      } catch (err) {
        if (cancelled) return;
        setError(
          toUiError(
            err,
            'Deployment Profile Error',
            'The backend deploy profile could not be loaded.'
          )
        );
      } finally {
        if (!cancelled) {
          setLoadingDeployInfo(false);
        }
      }
    }

    bootstrapDeployConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleLoadPreview() {
    if (!hsiFile && !inputPath) {
      setError({
        title: 'Input Required',
        detail: runtimeConfig.showServerPathInputs
          ? 'Choose an HSI upload or enter a server-side input path before loading preview.'
          : 'Upload an HSI file before loading preview.',
      });
      return;
    }

    setError(null);
    setBusyPreview(true);

    try {
      const result = await loadPreview({
        sensor,
        device,
        model_type: runtimeConfig.externalWebMode ? undefined : modelType,
        deploy_config_path: runtimeConfig.showDeployConfigInput ? (deployConfigPath || undefined) : undefined,
        input_path: runtimeConfig.showServerPathInputs ? (inputPath || undefined) : undefined,
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
        const rowStart = Math.floor(height * 0.2);
        const rowStop = Math.min(height, rowStart + 64);
        const colStart = Math.floor(width * 0.2);
        const colStop = Math.min(width, colStart + 64);
        setRoi({ rowStart, rowStop, colStart, colStop });
      }
    } catch (err) {
      setError(toUiError(err, 'Preview Error', 'Failed to load preview.'));
    } finally {
      setBusyPreview(false);
    }
  }

  async function handleRunInference() {
    if (!roi) return;

    setError(null);
    setBusyInference(true);

    try {
      const result = await runInference({
        sensor,
        device,
        model_type: runtimeConfig.externalWebMode ? undefined : modelType,
        deploy_config_path: runtimeConfig.showDeployConfigInput ? (deployConfigPath || undefined) : undefined,
        input_path: runtimeConfig.showServerPathInputs ? (inputPath || undefined) : undefined,
        hsi_file: hsiFile,
        wavelength_file: wavelengthFile,
        model_checkpoint: runtimeConfig.showAdvancedOverrides ? (modelCheckpoint || undefined) : undefined,
        temperature_json: runtimeConfig.showAdvancedOverrides ? (temperatureJson || undefined) : undefined,
        threshold: runtimeConfig.showAdvancedOverrides && threshold !== '' ? Number(threshold) : undefined,
        preview_band_index: previewBandIndex ? Number(previewBandIndex) : undefined,
        preview_wavelength: previewWavelength ? Number(previewWavelength) : undefined,
        row_start: roi.rowStart,
        row_stop: roi.rowStop,
        col_start: roi.colStart,
        col_stop: roi.colStop,
      });

      setInference(result);
      setPage('results');
    } catch (err) {
      setError(toUiError(err, 'Inference Error', 'Failed to run inference.'));
    } finally {
      setBusyInference(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>HSI Water Detection UI</h1>
      </header>

      {runtimeConfig.usesOwnerOperatedGpu ? (
        <div className="app-banner">
          GPU inference runs on the owner-operated research PC through a secure tunnel. If preview or inference
          fails, check that the backend PC is awake and the tunnel is online.
        </div>
      ) : null}

      <div className="workspace-layout">
        <aside className="sidebar">
          <SidebarControls
            appMode={runtimeConfig.appMode}
            deviceOptions={[...runtimeConfig.allowedDevices]}
            showModelTypeInput={!runtimeConfig.externalWebMode}
            showServerPathInputs={runtimeConfig.showServerPathInputs}
            showDeployConfigInput={runtimeConfig.showDeployConfigInput}
            showAdvancedOverrides={runtimeConfig.showAdvancedOverrides}
            loadingDeployConfig={loadingDeployInfo}
            deployConfigSummary={deployInfo?.deployConfigRelative || deployInfo?.deployConfigPath || 'backend default'}
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

          {error ? (
            <div className="error-box">
              <div className="error-title">
                {error.title}
                {error.status ? ` (${error.status})` : ''}
              </div>
              <div>{error.detail}</div>
              {error.hint ? <div className="error-hint">{error.hint}</div> : null}
            </div>
          ) : null}

          <ModelInfoPanel preview={preview} inference={inference} deployInfo={deployInfo} />
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
            <ResultsPage inference={inference} onBackToRoi={() => setPage('roi')} />
          )}
        </main>
      </div>
    </div>
  );
}
