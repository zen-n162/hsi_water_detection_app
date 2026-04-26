function parseBoolean(value: string | undefined, defaultValue: boolean): boolean {
  if (value === undefined) return defaultValue;
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

function parseCsv(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalizeApiBase(value: string | undefined): string {
  const trimmed = value?.trim() ?? '';
  return trimmed.replace(/\/+$/, '');
}

function normalizeAppMode(value: string | undefined): 'local' | 'public' | 'local_gpu_web' {
  const cleaned = value?.trim().toLowerCase();
  if (cleaned === 'public') return 'public';
  if (cleaned === 'local_gpu_web' || cleaned === 'local-gpu-web') return 'local_gpu_web';
  return 'local';
}

const appMode = normalizeAppMode(import.meta.env.VITE_APP_MODE);
const apiBaseFromEnv = normalizeApiBase(import.meta.env.VITE_API_BASE_URL);
const defaultApiBase = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';
const defaultDevice = (
  import.meta.env.VITE_DEFAULT_DEVICE ||
  (appMode === 'public' ? 'cpu' : 'cuda')
).trim();
const allowedDevices = parseCsv(import.meta.env.VITE_ALLOWED_DEVICES);
const defaultModelType = (import.meta.env.VITE_DEFAULT_MODEL_TYPE || 'ss').trim() || 'ss';
const defaultSensor = (import.meta.env.VITE_DEFAULT_SENSOR || 'hyperion').trim() || 'hyperion';
const externalWebMode = appMode === 'public' || appMode === 'local_gpu_web';

export const runtimeConfig = {
  appMode,
  apiBaseUrl: apiBaseFromEnv || defaultApiBase,
  defaultDevice,
  allowedDevices: allowedDevices.length > 0 ? allowedDevices : [defaultDevice],
  defaultModelType,
  defaultSensor,
  externalWebMode,
  usesOwnerOperatedGpu: appMode === 'local_gpu_web',
  showServerPathInputs: parseBoolean(import.meta.env.VITE_SHOW_SERVER_PATH_INPUTS, !externalWebMode),
  showDeployConfigInput: parseBoolean(import.meta.env.VITE_SHOW_DEPLOY_CONFIG_INPUT, !externalWebMode),
  showAdvancedOverrides: parseBoolean(import.meta.env.VITE_SHOW_ADVANCED_OVERRIDES, !externalWebMode),
} as const;

export type AppMode = (typeof runtimeConfig)['appMode'];
