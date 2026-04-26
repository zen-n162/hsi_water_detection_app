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
  const publicProvenance =
    (inference?.modelProvenance as Record<string, unknown> | undefined) ||
    (resolved as Record<string, unknown>);
  const isExternalMode = deployInfo?.runtime?.provenance_visibility === 'public_safe';

  return (
    <details className="collapse" open>
      <summary>Resolved inference settings</summary>
      <div className="model-info">
        <div><strong>Mode:</strong> {deployInfo?.runtime?.app_mode || '-'}</div>
        <div><strong>Deploy Config:</strong> {deployInfo?.deployConfigRelative || deployInfo?.deployConfigPath || '-'}</div>
        <div><strong>Deploy name:</strong> {String(publicProvenance.deploy_name || resolved.deploy_name || '-')}</div>
        <div><strong>Run:</strong> {inference?.resolvedRunName || String(publicProvenance.run_name || resolved.run_name || '-')}</div>
        <div><strong>Sensor:</strong> {inference?.sensor || String(publicProvenance.sensor || resolved.sensor || '-')}</div>
        <div><strong>Model type:</strong> {String(publicProvenance.model_type || inference?.raw?.resolved_model_type || resolved.model_type || '-')}</div>
        <div><strong>Threshold:</strong> {inference?.resolvedThreshold ?? publicProvenance.threshold ?? resolved.threshold ?? '-'}</div>
        <div><strong>Patch / stride:</strong> {String(publicProvenance.patch_size || resolved.patch_size || '-')} / {String(publicProvenance.stride || resolved.stride || '-')}</div>
        <div><strong>Split policy:</strong> {String(publicProvenance.split_policy || resolved.split_policy || '-')}</div>
        <div><strong>Output Root:</strong> {inference?.resolvedOutputRoot || deployInfo?.runtime?.output_root || '-'}</div>
        <div><strong>Device:</strong> {inference?.executedDevice || '-'}</div>
        <div><strong>Provenance visibility:</strong> {deployInfo?.runtime?.provenance_visibility || inference?.provenanceVisibility || '-'}</div>
        <div><strong>Preview:</strong> {preview?.grayscalePreviewUrl ? 'loaded' : 'not loaded'}</div>
        {deployInfo?.runtime?.app_mode === 'local_gpu_web' ? (
          <div><strong>Runtime:</strong> owner-operated research PC GPU via tunnel</div>
        ) : null}
        {!isExternalMode ? (
          <>
            <div><strong>Checkpoint:</strong> {inference?.resolvedModelCheckpoint || resolved.model_checkpoint || '-'}</div>
            <div><strong>Calibration:</strong> {inference?.resolvedTemperatureJson || resolved.temperature_json || '-'}</div>
            <div><strong>Manifest:</strong> {inference?.resolvedManifest || resolved.manifest_path || '-'}</div>
            <div><strong>Dataset:</strong> {inference?.resolvedDataset || resolved.dataset_path || '-'}</div>
          </>
        ) : null}
      </div>
    </details>
  );
}
