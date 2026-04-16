const API_BASE = 'http://127.0.0.1:8000';

function withBase(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  if (url.startsWith('/')) return `${API_BASE}${url}`;
  return `${API_BASE}/${url}`;
}

function pickFirst<T>(...values: (T | undefined | null)[]): T | undefined {
  for (const v of values) {
    if (v !== undefined && v !== null) return v;
  }
  return undefined;
}

export type PreviewRequest = {
  sensor: string;
  device?: string;
  model_type?: string;
  deploy_config_path?: string;
  input_path?: string;
  hsi_file?: File | null;
  wavelength_file?: File | null;
  preview_band_index?: number;
  preview_wavelength?: number;
};

export type PreviewResponse = {
  ok: boolean;
  grayscalePreviewUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  previewBandIndex?: number;
  previewWavelength?: number;
  raw: any;
};

export type InferenceRequest = {
  sensor: string;
  device: string;
  model_type?: string;
  deploy_config_path?: string;
  input_path?: string;
  hsi_file?: File | null;
  wavelength_file?: File | null;
  model_checkpoint?: string;
  temperature_json?: string;
  threshold?: number;
  preview_band_index?: number;
  preview_wavelength?: number;
  row_start: number;
  row_stop: number;
  col_start: number;
  col_stop: number;
};

export type InferenceResponse = {
  ok: boolean;
  pseudocolorUrl?: string;
  probabilityOverlayUrl?: string;
  probabilityMapUrl?: string;
  spatialAttentionUrl?: string;
  spatialAttentionOverlayUrl?: string;
  spectralAttentionUrl?: string;
  metadataUrl?: string;

  resolvedRunName?: string;
  resolvedModelCheckpoint?: string;
  resolvedTemperatureJson?: string;
  resolvedThreshold?: number;
  resolvedManifest?: string;
  resolvedDataset?: string;
  resolvedSplitPolicy?: string;
  executedDevice?: string;
  requestedDevice?: string;

  stdout?: string;
  stderr?: string;
  raw: any;
};

function buildFormData(obj: Record<string, any>): FormData {
  const fd = new FormData();
  Object.entries(obj).forEach(([k, v]) => {
    if (v === undefined || v === null || v === '') return;
    if (v instanceof File) {
      fd.append(k, v);
    } else {
      fd.append(k, String(v));
    }
  });
  return fd;
}

export async function loadPreview(req: PreviewRequest): Promise<PreviewResponse> {
  const fd = buildFormData({
    sensor: req.sensor,
    device: req.device,
    model_type: req.model_type,
    deploy_config_path: req.deploy_config_path,
    input_path: req.input_path,
    hsi_file: req.hsi_file ?? undefined,
    wavelength_file: req.wavelength_file ?? undefined,
    preview_band_index: req.preview_band_index,
    preview_wavelength: req.preview_wavelength,
  });

  const res = await fetch(`${API_BASE}/preview/grayscale`, {
    method: 'POST',
    body: fd,
  });

  const json = await res.json();
  if (!res.ok || json?.ok === false) {
    throw new Error(json?.detail || json?.stderr || 'Failed to load preview.');
  }

  const grayscalePreviewUrl = withBase(
    pickFirst(
      json.grayscale_preview_url,
      json.preview_url,
      json.grayscale_url,
      json.urls?.grayscale_preview_png
    )
  );

  return {
    ok: Boolean(json.ok ?? true),
    grayscalePreviewUrl,
    imageWidth: pickFirst(json.image_width, json.width),
    imageHeight: pickFirst(json.image_height, json.height),
    previewBandIndex: pickFirst(json.preview_band_index, json.preview_band),
    previewWavelength: pickFirst(json.preview_wavelength_nm, json.preview_wavelength),
    raw: json,
  };
}

export async function runInference(req: InferenceRequest): Promise<InferenceResponse> {
  const fd = buildFormData({
    sensor: req.sensor,
    device: req.device,
    model_type: req.model_type,
    deploy_config_path: req.deploy_config_path,
    input_path: req.input_path,
    hsi_file: req.hsi_file ?? undefined,
    wavelength_file: req.wavelength_file ?? undefined,
    model_checkpoint: req.model_checkpoint,
    temperature_json: req.temperature_json,
    threshold: req.threshold,
    preview_band_index: req.preview_band_index,
    preview_wavelength: req.preview_wavelength,
    row_start: req.row_start,
    row_stop: req.row_stop,
    col_start: req.col_start,
    col_stop: req.col_stop,
  });

  const res = await fetch(`${API_BASE}/inference/run`, {
    method: 'POST',
    body: fd,
  });

  const json = await res.json();
  if (!res.ok || json?.ok === false) {
    throw new Error(json?.detail || json?.stderr || 'Failed to run inference.');
  }

  return {
    ok: Boolean(json.ok ?? true),

    pseudocolorUrl: withBase(
      pickFirst(json.pseudocolor_url, json.urls?.pseudocolor_png)
    ),
    probabilityOverlayUrl: withBase(
      pickFirst(json.probability_overlay_url, json.urls?.probability_overlay_png)
    ),
    probabilityMapUrl: withBase(
      pickFirst(json.probability_map_url, json.urls?.probability_map_png)
    ),
    spatialAttentionUrl: withBase(
      pickFirst(json.spatial_attention_url, json.urls?.spatial_attention_png)
    ),
    spatialAttentionOverlayUrl: withBase(
      pickFirst(
        json.spatial_attention_overlay_url,
        json.urls?.spatial_attention_overlay_png
      )
    ),
    spectralAttentionUrl: withBase(
      pickFirst(json.spectral_attention_url, json.urls?.spectral_attention_png)
    ),
    metadataUrl: withBase(
      pickFirst(json.metadata_url, json.urls?.metadata_json)
    ),

    resolvedRunName: json.resolved_run_name,
    resolvedModelCheckpoint: json.resolved_model_checkpoint,
    resolvedTemperatureJson: json.resolved_temperature_json,
    resolvedThreshold: json.resolved_threshold,
    resolvedManifest: json.resolved_manifest,
    resolvedDataset: json.resolved_dataset,
    resolvedSplitPolicy: json.resolved_split_policy,
    executedDevice: json.executed_device,
    requestedDevice: json.requested_device,

    stdout: json.stdout,
    stderr: json.stderr,
    raw: json,
  };
}
