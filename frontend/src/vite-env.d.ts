/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_APP_MODE?: 'local' | 'public' | 'local_gpu_web';
  readonly VITE_DEFAULT_DEVICE?: string;
  readonly VITE_ALLOWED_DEVICES?: string;
  readonly VITE_DEFAULT_MODEL_TYPE?: string;
  readonly VITE_DEFAULT_SENSOR?: string;
  readonly VITE_SHOW_SERVER_PATH_INPUTS?: string;
  readonly VITE_SHOW_DEPLOY_CONFIG_INPUT?: string;
  readonly VITE_SHOW_ADVANCED_OVERRIDES?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
