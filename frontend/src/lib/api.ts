import { runtimeConfig } from './runtime';

const API_BASE = runtimeConfig.apiBaseUrl;

function withBase(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  if (!API_BASE) return url.startsWith('/') ? url : `/${url}`;
  if (url.startsWith('/')) return `${API_BASE}${url}`;
  return `${API_BASE}/${url}`;
}

function toApiUrl(path: string): string {
  return withBase(path) || path;
}

function pickFirst<T>(...values: (T | undefined | null)[]): T | undefined {
  for (const value of values) {
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function stringifyDetail(value: unknown, fallback: string): string {
  if (typeof value === 'string' && value.trim()) return value;
  if (value && typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

async function parseBody(response: Response): Promise<any> {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return await response.json();
    } catch {
      return undefined;
    }
  }

  const text = await response.text();
  return text ? { detail: text } : undefined;
}

export class ApiError extends Error {
  status?: number;
  hint?: string;
  code?: string;
  raw?: unknown;

  constructor(message: string, options?: { status?: number; hint?: string; code?: string; raw?: unknown }) {
    super(message);
    this.name = 'ApiError';
    this.status = options?.status;
    this.hint = options?.hint;
    this.code = options?.code;
    this.raw = options?.raw;
  }
}

async function requestJson(path: string, init: RequestInit, fallbackMessage: string): Promise<any> {
  const response = await fetch(toApiUrl(path), init);
  const body = await parseBody(response);

  if (!response.ok || body?.ok === false) {
    const message = stringifyDetail(
      pickFirst(body?.detail, body?.stderr, body?.message),
      fallbackMessage
    );
    throw new ApiError(message, {
      status: response.status,
      hint: typeof body?.hint === 'string' ? body.hint : undefined,
      code: typeof body?.code === 'string' ? body.code : undefined,
      raw: body,
    });
  }

  return body;
}

function buildFormData(obj: Record<string, any>): FormData {
  const fd = new FormData();
  Object.entries(obj).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    if (value instanceof File) {
      fd.append(key, value);
    } else {
      fd.append(key, String(value));
    }
  });
  return fd;
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
  label_file?: File | null;
  mask_file?: File | null;
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
  labelMapUrl?: string;
  labelVisualizationError?: string | null;
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
  resolvedOutputRoot?: string;
  executedDevice?: string;
  requestedDevice?: string;
  sensor?: string;
  provenanceVisibility?: string;
  modelProvenance?: Record<string, any>;
  inferenceMask?: Record<string, any> | null;
  inferenceMaskUrl?: string;
  inferenceMaskError?: string | null;
  evaluationMetrics?: Record<string, any> | null;
  evaluationMetricsUrl?: string;
  evaluationError?: string | null;

  stdout?: string;
  stderr?: string;
  raw: any;
};

export type DeployConfigResponse = {
  deployConfigPath?: string;
  deployConfigRelative?: string;
  deployConfig?: any;
  resolved: Record<string, any>;
  referenceModels?: Record<string, any>;
  runtime?: {
    app_mode?: string;
    allow_server_file_paths?: boolean;
    allow_deploy_config_override?: boolean;
    output_root?: string;
    output_url_prefix?: string;
    public_base_url?: string | null;
    default_device?: string;
    provenance_visibility?: string;
  };
};

export async function loadDeployConfig(deployConfigPath?: string): Promise<DeployConfigResponse> {
  const query = deployConfigPath
    ? `?deploy_config_path=${encodeURIComponent(deployConfigPath)}`
    : '';
  const json = await requestJson(
    `/inference/deploy-config${query}`,
    { method: 'GET' },
    'Failed to load deploy configuration.'
  );

  return {
    deployConfigPath: json.deploy_config_path,
    deployConfigRelative: json.deploy_config_relative,
    deployConfig: json.deploy_config,
    resolved: json.resolved || {},
    referenceModels: json.reference_models || {},
    runtime: json.runtime || {},
  };
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

  const json = await requestJson(
    '/preview/grayscale',
    {
      method: 'POST',
      body: fd,
    },
    'Failed to load preview.'
  );

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
    label_file: req.label_file ?? undefined,
    mask_file: req.mask_file ?? undefined,
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

  const json = await requestJson(
    '/inference/run',
    {
      method: 'POST',
      body: fd,
    },
    'Failed to run inference.'
  );

  return {
    ok: Boolean(json.ok ?? true),
    pseudocolorUrl: withBase(
      pickFirst(json.pseudocolor_url, json.urls?.pseudocolor_png)
    ),
    labelMapUrl: withBase(
      pickFirst(json.label_map_url, json.urls?.label_map_png)
    ),
    labelVisualizationError: json.label_visualization_error,
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
    resolvedOutputRoot: json.resolved_output_root,
    executedDevice: json.executed_device,
    requestedDevice: json.requested_device,
    sensor: json.sensor,
    provenanceVisibility: json.provenance_visibility,
    modelProvenance: json.model_provenance,
    inferenceMask: json.inference_mask,
    inferenceMaskUrl: withBase(
      pickFirst(json.inference_mask_url, json.urls?.inference_mask_json)
    ),
    inferenceMaskError: json.inference_mask_error,
    evaluationMetrics: json.evaluation_metrics,
    evaluationMetricsUrl: withBase(
      pickFirst(json.evaluation_metrics_url, json.urls?.evaluation_metrics_json)
    ),
    evaluationError: json.evaluation_error,
    stdout: json.stdout,
    stderr: json.stderr,
    raw: json,
  };
}
