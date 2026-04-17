import type { AppMode } from '../lib/runtime';

type Props = {
  appMode: AppMode;
  deviceOptions: string[];
  showModelTypeInput: boolean;
  showServerPathInputs: boolean;
  showDeployConfigInput: boolean;
  showAdvancedOverrides: boolean;
  loadingDeployConfig: boolean;
  deployConfigSummary: string;
  sensor: string;
  setSensor: (v: string) => void;
  device: string;
  setDevice: (v: string) => void;
  modelType: string;
  setModelType: (v: string) => void;
  deployConfigPath: string;
  setDeployConfigPath: (v: string) => void;
  inputPath: string;
  setInputPath: (v: string) => void;
  hsiFile: File | null;
  setHsiFile: (f: File | null) => void;
  wavelengthFile: File | null;
  setWavelengthFile: (f: File | null) => void;
  modelCheckpoint: string;
  setModelCheckpoint: (v: string) => void;
  temperatureJson: string;
  setTemperatureJson: (v: string) => void;
  threshold: string;
  setThreshold: (v: string) => void;
  previewBandIndex: string;
  setPreviewBandIndex: (v: string) => void;
  previewWavelength: string;
  setPreviewWavelength: (v: string) => void;
  onLoadPreview: () => void;
  onRunInference: () => void;
  busyPreview: boolean;
  busyInference: boolean;
  canRunInference: boolean;
  currentPage: 'roi' | 'results';
  onGoRoi: () => void;
  onGoResults: () => void;
  hasResults: boolean;
};

export default function SidebarControls(props: Props) {
  const runtimeNote =
    props.appMode === 'local_gpu_web'
      ? 'Netlify calls the owner-operated GPU backend through a secure tunnel. Local file-path inputs and deploy-profile overrides stay locked.'
      : props.appMode === 'public'
        ? 'Public mode is upload-first. Server-side file paths and deploy profile overrides are hidden by default.'
        : 'Local mode keeps server-side paths and deploy profile overrides available for development.';

  return (
    <div className="sidebar-controls">
      <div className="sidebar-nav">
        <button
          className={props.currentPage === 'roi' ? 'nav-btn active' : 'nav-btn'}
          onClick={props.onGoRoi}
          type="button"
        >
          ROI Selection
        </button>
        <button
          className={props.currentPage === 'results' ? 'nav-btn active' : 'nav-btn'}
          onClick={props.onGoResults}
          type="button"
          disabled={!props.hasResults}
        >
          Results
        </button>
      </div>

      <div className="runtime-note">{runtimeNote}</div>

      {props.showServerPathInputs ? (
        <label className="field">
          <span>HSI file path</span>
          <input
            value={props.inputPath}
            onChange={(e) => props.setInputPath(e.target.value)}
            placeholder="/path/to/input.tif"
          />
        </label>
      ) : null}

      <label className="field">
        <span>HSI file upload</span>
        <input
          type="file"
          accept=".tif,.tiff,.img"
          onChange={(e) => props.setHsiFile(e.target.files?.[0] || null)}
        />
      </label>

      <label className="field">
        <span>Wavelength sidecar</span>
        <input
          type="file"
          accept=".json,.txt,.csv"
          onChange={(e) => props.setWavelengthFile(e.target.files?.[0] || null)}
        />
      </label>

      <label className="field">
        <span>Sensor</span>
        <select value={props.sensor} onChange={(e) => props.setSensor(e.target.value)}>
          <option value="hyperion">hyperion</option>
        </select>
      </label>

      <label className="field">
        <span>Device</span>
        <select value={props.device} onChange={(e) => props.setDevice(e.target.value)}>
          {props.deviceOptions.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>

      {props.showModelTypeInput ? (
        <label className="field">
          <span>Model type</span>
          <select value={props.modelType} onChange={(e) => props.setModelType(e.target.value)}>
            <option value="ss">ss</option>
            <option value="sa">sa</option>
          </select>
        </label>
      ) : (
        <div className="field-note">
          <strong>Model type:</strong> {props.modelType}
        </div>
      )}

      {props.showDeployConfigInput ? (
        <label className="field">
          <span>Deploy config</span>
          <input value={props.deployConfigPath} onChange={(e) => props.setDeployConfigPath(e.target.value)} />
        </label>
      ) : (
        <div className="field-note">
          <strong>Deploy profile:</strong>{' '}
          {props.loadingDeployConfig ? 'loading...' : props.deployConfigSummary}
        </div>
      )}

      {props.showAdvancedOverrides ? (
        <details className="collapse">
          <summary>Advanced overrides</summary>

          <label className="field">
            <span>Fine-tuned model checkpoint</span>
            <input value={props.modelCheckpoint} onChange={(e) => props.setModelCheckpoint(e.target.value)} />
          </label>

          <label className="field">
            <span>Calibration JSON override</span>
            <input value={props.temperatureJson} onChange={(e) => props.setTemperatureJson(e.target.value)} />
          </label>

          <label className="field">
            <span>Decision threshold override</span>
            <input value={props.threshold} onChange={(e) => props.setThreshold(e.target.value)} />
          </label>

          <label className="field">
            <span>Preview band index</span>
            <input value={props.previewBandIndex} onChange={(e) => props.setPreviewBandIndex(e.target.value)} />
          </label>

          <label className="field">
            <span>Preview wavelength</span>
            <input value={props.previewWavelength} onChange={(e) => props.setPreviewWavelength(e.target.value)} />
          </label>
        </details>
      ) : null}

      <div className="action-stack">
        <button onClick={props.onLoadPreview} disabled={props.busyPreview} type="button">
          {props.busyPreview ? 'Loading Preview...' : 'Load Preview'}
        </button>
        <button onClick={props.onRunInference} disabled={!props.canRunInference || props.busyInference} type="button">
          {props.busyInference ? 'Running...' : 'Run Inference'}
        </button>
      </div>
    </div>
  );
}
