import type {
  DeployConfigResponse,
  PreviewResponse,
  InferenceResponse,
} from '../lib/api';

export default function ModelInfoPanel({
  preview,
  inference,
  deployInfo,
}: {
  preview: PreviewResponse | null;
  inference: InferenceResponse | null;
  deployInfo: DeployConfigResponse | null;
}) {
  const resolved = deployInfo?.resolved || {};

  return (
    <details className="collapse" open>
      <summary>Resolved inference settings</summary>
      <div className="model-info">
        <div><strong>Mode:</strong> {deployInfo?.runtime?.app_mode || '-'}</div>
        <div><strong>Deploy Config:</strong> {deployInfo?.deployConfigRelative || deployInfo?.deployConfigPath || '-'}</div>
        <div><strong>Run:</strong> {inference?.resolvedRunName || resolved.run_name || '-'}</div>
        <div><strong>Checkpoint:</strong> {inference?.resolvedModelCheckpoint || resolved.model_checkpoint || '-'}</div>
        <div><strong>Calibration:</strong> {inference?.resolvedTemperatureJson || resolved.temperature_json || '-'}</div>
        <div><strong>Threshold:</strong> {inference?.resolvedThreshold ?? resolved.threshold ?? '-'}</div>
        <div><strong>Manifest:</strong> {inference?.resolvedManifest || resolved.manifest_path || '-'}</div>
        <div><strong>Dataset:</strong> {inference?.resolvedDataset || resolved.dataset_path || '-'}</div>
        <div><strong>Output Root:</strong> {inference?.resolvedOutputRoot || deployInfo?.runtime?.output_root || '-'}</div>
        <div><strong>Device:</strong> {inference?.executedDevice || '-'}</div>
        <div><strong>Preview:</strong> {preview?.grayscalePreviewUrl ? 'loaded' : 'not loaded'}</div>
      </div>
    </details>
  );
}
