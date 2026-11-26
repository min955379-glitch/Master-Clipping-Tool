import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react';

type Provider = 'ollama' | 'gemini' | 'openai' | 'anthropic' | 'groq' | 'openrouter';
type ReframeMode = 'face_track' | 'podcast_split' | 'gaming' | 'fit_blur';
type CaptionStyle = 'Hormozi Bold' | 'Clean Minimal' | 'Gaming Neon' | 'Podcast Classic';
type ExportPreset = 'short_vertical' | 'square' | 'landscape';
type JobStatus = 'queued' | 'running' | 'completed' | 'partial' | 'failed';

type ProviderConfig = {
  model: string;
  base_url: string;
  configured: boolean;
  api_key_saved: boolean;
};

type ProviderSettings = {
  active_provider: Provider;
  providers: Record<Provider, ProviderConfig>;
};

type SettingsTestResult = {
  ok: boolean;
  provider: Provider;
  message: string;
  settings: ProviderSettings;
};

type Clip = {
  clip_id: string;
  start: number;
  end: number;
  virality_score: number;
  hook_title: string;
  reason: string;
  suggested_caption: string;
  encoder: string;
  file_name: string;
  download_url: string;
};

type Job = {
  job_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  source_label: string;
  clips: Clip[];
  warnings: string[];
  error: string | null;
};

type Runtime = {
  python?: string;
  ollama_url?: string;
  gemini_configured?: boolean;
  providers_configured?: Partial<Record<Provider, boolean>>;
  encoder?: { available: boolean; selected?: string; hardware_accelerated?: boolean; error?: string };
};

type IconName = 'film' | 'spark' | 'folder' | 'arrow' | 'upload' | 'download' | 'clock' | 'gear' | 'close' | 'eye' | 'paste' | 'trash';
type Feedback = { ok: boolean; message: string } | null;

const PROVIDERS: { value: Provider; label: string }[] = [
  { value: 'ollama', label: 'Local Ollama' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'groq', label: 'Groq' },
  { value: 'openrouter', label: 'OpenRouter' },
];

const DEFAULTS: Record<Provider, { model: string; base_url: string }> = {
  ollama: { model: 'qwen2.5:7b', base_url: 'http://localhost:11434' },
  gemini: { model: 'gemini-3.8-flash', base_url: 'https://generativelanguage.googleapis.com/v1beta' },
  openai: { model: 'gpt-4.1-mini', base_url: 'https://api.openai.com/v1' },
  anthropic: { model: 'claude-3-5-haiku-latest', base_url: 'https://api.anthropic.com' },
  groq: { model: 'llama-3.3-70b-versatile', base_url: 'https://api.groq.com/openai/v1' },
  openrouter: { model: 'openai/gpt-4.1-mini', base_url: 'https://openrouter.ai/api/v1' },
};

const EMPTY_SETTINGS: ProviderSettings = {
  active_provider: 'ollama',
  providers: Object.fromEntries(PROVIDERS.map(({ value }) => [value, {
    ...DEFAULTS[value], configured: value === 'ollama', api_key_saved: false,
  }])) as Record<Provider, ProviderConfig>,
};

const API_KEY_LABELS: Partial<Record<Provider, string>> = {
  gemini: 'Google AI API key',
  openai: 'OpenAI API key',
  anthropic: 'Anthropic API key',
  groq: 'Groq API key',
  openrouter: 'OpenRouter API key',
};

const VIDEO_ACCEPT = '.mp4,.mkv,.mov,.webm,.avi,.m4v,.mpeg,.mpg,.ts';

function Icon({ name }: { name: IconName }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  if (name === 'film') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v4m5-4v4m5-4v4M7 16v4m5-4v4m5-4v4M3 9h18m-18 6h18"/></svg>;
  if (name === 'folder') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5v-10Z"/><path d="M3.5 10h17"/></svg>;
  if (name === 'arrow') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M5 12h14m-6-6 6 6-6 6"/></svg>;
  if (name === 'upload') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M12 16V4m-5 5 5-5 5 5"/><path d="M5 14v5h14v-5"/></svg>;
  if (name === 'download') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M12 4v12m-5-5 5 5 5-5"/><path d="M5 19h14"/></svg>;
  if (name === 'clock') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>;
  if (name === 'gear') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z"/><path d="m19.4 13.5 1.1.8-1 1.8-1.3-.4a7.7 7.7 0 0 1-1.3 1.2l.1 1.4-2 .6-.7-1.2a7.6 7.6 0 0 1-1.7 0l-.7 1.2-2-.6.1-1.4a7.7 7.7 0 0 1-1.3-1.2l-1.3.4-1-1.8 1.1-.8a7.4 7.4 0 0 1-.2-1.5c0-.5.1-1 .2-1.5l-1.1-.8 1-1.8 1.3.4a7.7 7.7 0 0 1 1.3-1.2L10 5.7l2-.6.7 1.2a7.6 7.6 0 0 1 1.7 0l.7-1.2 2 .6-.1 1.4a7.7 7.7 0 0 1 1.3 1.2l1.3-.4 1 1.8-1.1.8a7.4 7.4 0 0 1 .2 1.5c0 .5-.1 1-.2 1.5Z"/></svg>;
  if (name === 'close') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="m6 6 12 12M18 6 6 18"/></svg>;
  if (name === 'eye') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M2.5 12s3.3-6 9.5-6 9.5 6 9.5 6-3.3 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></svg>;
  if (name === 'paste') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4.5h6v3H9zM9 12h6m-6 4h6"/></svg>;
  if (name === 'trash') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>;
  return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="m12 2 1.8 7.2L21 11l-7.2 1.8L12 20l-1.8-7.2L3 11l7.2-1.8L12 2Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/></svg>;
}

function formatTime(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remainder = whole % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}` : `${minutes}:${String(remainder).padStart(2, '0')}`;
}

function errorMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === 'string') return detail;
  }
  return fallback;
}

function App() {
  const fileInput = useRef<HTMLInputElement>(null);
  const settingsKeyInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState('');
  const [provider, setProvider] = useState<Provider>('ollama');
  const [model, setModel] = useState(DEFAULTS.ollama.model);
  const [baseUrl, setBaseUrl] = useState(DEFAULTS.ollama.base_url);
  const [providerSettings, setProviderSettings] = useState<ProviderSettings>(EMPTY_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsProvider, setSettingsProvider] = useState<Provider>('ollama');
  const [settingsModel, setSettingsModel] = useState(DEFAULTS.ollama.model);
  const [settingsBaseUrl, setSettingsBaseUrl] = useState(DEFAULTS.ollama.base_url);
  const [settingsKey, setSettingsKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [settingsFeedback, setSettingsFeedback] = useState<Feedback>(null);
  const [direction, setDirection] = useState('');
  const [layout, setLayout] = useState<ReframeMode>('face_track');
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>('Hormozi Bold');
  const [exportPreset, setExportPreset] = useState<ExportPreset>('short_vertical');
  const [clipCount, setClipCount] = useState(4);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [deletingClipId, setDeletingClipId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let alive = true;
    const checkHealth = async () => {
      try {
        const response = await fetch('/api/health', { cache: 'no-store' });
        if (!response.ok) throw new Error('API is not responding');
        if (alive) setApiOnline(true);
      } catch {
        if (alive) setApiOnline(false);
      }
    };
    const loadRuntime = async () => {
      try {
        const response = await fetch('/api/runtime', { cache: 'no-store' });
        if (response.ok && alive) setRuntime(await response.json());
      } catch {
        if (alive) setRuntime(null);
      }
    };
    const loadSettings = async () => {
      try {
        const response = await fetch('/api/settings', { cache: 'no-store' });
        if (!response.ok) return;
        const saved = (await response.json()) as ProviderSettings;
        if (!alive) return;
        setProviderSettings(saved);
        setProvider(saved.active_provider);
        setModel(saved.providers[saved.active_provider]?.model ?? DEFAULTS[saved.active_provider].model);
        setBaseUrl(saved.providers[saved.active_provider]?.base_url ?? DEFAULTS[saved.active_provider].base_url);
      } catch {
        // The studio remains usable with the built-in local defaults if settings cannot load.
      }
    };
    void checkHealth();
    void loadRuntime();
    void loadSettings();
    const healthTimer = window.setInterval(checkHealth, 15000);
    return () => {
      alive = false;
      window.clearInterval(healthTimer);
    };
  }, []);

  useEffect(() => {
    if (!settingsOpen) return;
    const config = providerSettings.providers[settingsProvider] ?? EMPTY_SETTINGS.providers[settingsProvider];
    setSettingsModel(config.model || DEFAULTS[settingsProvider].model);
    setSettingsBaseUrl(config.base_url || DEFAULTS[settingsProvider].base_url);
    setSettingsKey('');
    setShowKey(false);
  }, [settingsOpen, settingsProvider, providerSettings]);

  useEffect(() => {
    if (!settingsOpen) return undefined;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSettingsOpen(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [settingsOpen]);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Could not read job progress');
        const current = (await response.json()) as Job;
        if (!cancelled) setJob(current);
      } catch (problem) {
        if (!cancelled) setError(problem instanceof Error ? problem.message : 'Progress polling failed');
      }
    };
    const timer = window.setInterval(() => void poll(), 1200);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  function chooseProvider(next: Provider) {
    setProvider(next);
    const config = providerSettings.providers[next] ?? EMPTY_SETTINGS.providers[next];
    setModel(config.model || DEFAULTS[next].model);
    setBaseUrl(config.base_url || DEFAULTS[next].base_url);
  }

  function acceptFile(nextFile: File | null) {
    if (!nextFile) return;
    setFile(nextFile);
    setSource('');
    setError('');
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0] ?? null);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    if (!file && !source.trim()) {
      setError('Choose a local video or paste a YouTube/Twitch URL first.');
      return;
    }
    setBusy(true);
    setJob(null);
    try {
      let response: Response;
      if (file) {
        const form = new FormData();
        form.set('file', file);
        form.set('provider', provider);
        form.set('model', model.trim());
        form.set('base_url', baseUrl.trim());
        form.set('custom_direction', direction);
        form.set('layout', layout);
        form.set('caption_style', captionStyle);
        form.set('export_preset', exportPreset);
        form.set('clip_count', String(clipCount));
        form.set('min_clip_seconds', '15');
        form.set('max_clip_seconds', '60');
        form.set('transcribe_missing', 'true');
        response = await fetch('/api/jobs/upload', { method: 'POST', body: form });
      } else {
        response = await fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source: source.trim(), provider, model: model.trim() || null, base_url: baseUrl.trim(),
            custom_direction: direction, layout, caption_style: captionStyle,
            export_preset: exportPreset, clip_count: clipCount,
            min_clip_seconds: 15, max_clip_seconds: 60, transcribe_missing: true,
          }),
        });
      }
      const payload = await response.json();
      if (!response.ok) throw new Error(errorMessage(payload, 'Could not start the clipping job'));
      setJob(payload as Job);
      setApiOnline(true);
      setFile(null);
      if (fileInput.current) fileInput.current.value = '';
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Could not start the clipping job');
    } finally {
      setBusy(false);
    }
  }

  async function deleteClip(clip: Clip) {
    if (!window.confirm(`Delete “${clip.hook_title}” and its caption sidecar from this device?`)) return;
    setDeletingClipId(clip.clip_id);
    setError('');
    try {
      const response = await fetch(`/api/clips/${encodeURIComponent(clip.clip_id)}`, { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok) throw new Error(errorMessage(payload, 'Could not delete this clip'));
      setJob((current) => current ? { ...current, clips: current.clips.filter((item) => item.clip_id !== clip.clip_id) } : current);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Could not delete this clip');
    } finally {
      setDeletingClipId(null);
    }
  }

  async function pasteApiKey() {
    try {
      const value = await navigator.clipboard.readText();
      if (!value.trim()) throw new Error('Clipboard is empty');
      setSettingsKey(value.trim());
      settingsKeyInput.current?.focus();
      setSettingsFeedback({ ok: true, message: 'Key pasted locally. Save & Test to store it securely.' });
    } catch {
      setSettingsFeedback({ ok: false, message: 'Clipboard access is unavailable. Paste the key into the field.' });
    }
  }

  async function saveAndTestSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSettingsBusy(true);
    setSettingsFeedback(null);
    try {
      const response = await fetch('/api/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: settingsProvider,
          model: settingsModel.trim(),
          base_url: settingsBaseUrl.trim(),
          api_key: settingsKey || null,
        }),
      });
      const result = await response.json() as SettingsTestResult;
      if (result.settings) {
        setProviderSettings(result.settings);
        setProvider(settingsProvider);
        setModel(result.settings.providers[settingsProvider]?.model ?? settingsModel.trim());
        setBaseUrl(result.settings.providers[settingsProvider]?.base_url ?? settingsBaseUrl.trim());
      }
      if (!response.ok) throw new Error(errorMessage(result, 'Could not save provider settings'));
      setSettingsFeedback({ ok: result.ok, message: result.message });
      if (result.ok) setSettingsKey('');
    } catch (problem) {
      setSettingsFeedback({ ok: false, message: problem instanceof Error ? problem.message : 'Provider test failed' });
    } finally {
      setSettingsBusy(false);
    }
  }

  const completed = job && ['completed', 'partial', 'failed'].includes(job.status);
  const encoderLabel = runtime?.encoder?.selected ?? (runtime?.encoder?.available === false ? 'Unavailable' : 'Checking FFmpeg');
  const providerConfig = providerSettings.providers[provider];
  const providerNote = provider === 'ollama'
    ? `Local Ollama · ${baseUrl || runtime?.ollama_url || DEFAULTS.ollama.base_url}`
    : providerConfig?.configured
      ? `${PROVIDERS.find((item) => item.value === provider)?.label ?? provider} is configured in local settings.`
      : `Add a ${API_KEY_LABELS[provider] ?? 'provider API key'} in Settings to connect.`;
  const activeSettingsConfig = providerSettings.providers[settingsProvider] ?? EMPTY_SETTINGS.providers[settingsProvider];

  return (
    <div className="studio-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="FrameForge home">
          <span className="brand-mark"><Icon name="film" /></span>
          <span className="brand-copy"><strong>FRAME / FORGE</strong><small>LOCAL VIDEO STUDIO</small></span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <a className="nav-item nav-item-active" href="#studio"><span className="nav-icon"><Icon name="spark" /></span><span>Studio</span><span className="nav-indicator" /></a>
          <a className="nav-item nav-item-muted" href="#exports"><span className="nav-icon"><Icon name="folder" /></span><span>Clip library</span><span className="soon-label">LOCAL</span></a>
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-mark"><span className="privacy-dot" /> LOCAL-FIRST BY DESIGN</div>
          <p>Source media and exports stay on this machine. Cloud providers receive transcript context only.</p>
          <div className="sidebar-version">PHASE 03 · PIPELINE READY</div>
        </div>
      </aside>

      <main className="main-content" id="top">
        <header className="topbar">
          <div className="breadcrumb"><span>WORKSPACE</span><span className="crumb-divider">/</span><strong>STUDIO</strong></div>
          <div className="topbar-right">
            <span className={`runtime-pill ${apiOnline === false ? 'runtime-pill-offline' : ''}`}>
              <span className="runtime-dot" /> {apiOnline === null ? 'CONNECTING' : apiOnline ? 'LOCAL API READY' : 'API OFFLINE'}
            </span>
            <button className="icon-button settings-trigger" type="button" aria-label="Open provider settings" title="Provider settings" onClick={() => { setSettingsProvider(provider); setSettingsFeedback(null); setSettingsOpen(true); }}>
              <Icon name="gear" />
            </button>
            <span className="avatar">FF</span>
          </div>
        </header>

        <div className="content-wrap" id="studio">
          <section className="intro-section">
            <div className="intro-copy">
              <div className="eyebrow"><span className="eyebrow-line" /> THE CREATOR'S CUTTING ROOM</div>
              <h1>Find the moment<br />that <span>moves</span> the room.</h1>
              <p className="intro-description">Turn a long-form video into ranked, reframed short-form stories. Your source media and exports stay local; provider choice is yours.</p>
            </div>
            <div className="intro-stamp" aria-hidden="true"><div className="stamp-ring">CUT WITH INTENTION · CUT WITH INTENTION ·</div><div className="stamp-center"><Icon name="spark" /></div></div>
          </section>

          <form className="workspace-grid" onSubmit={submit}>
            <article className="panel source-panel">
              <div className="panel-heading"><div><div className="panel-kicker">01 / SOURCE</div><h2>Start with your footage</h2></div><span className="panel-number">01</span></div>
              <div
                className={`dropzone ${dragging ? 'dropzone-active' : ''} ${file ? 'dropzone-selected' : ''}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInput.current?.click()}
                aria-label="Choose or drop a video file"
              >
                <input ref={fileInput} className="visually-hidden" type="file" accept={VIDEO_ACCEPT} onChange={onFileChange} />
                <div className="dropzone-icon"><Icon name="upload" /></div>
                <div className="dropzone-title">{file ? file.name : 'Drop a video, or browse'}</div>
                <p>{file ? `${(file.size / (1024 ** 3)).toFixed(2)} GB · ready for local upload` : 'Choose an existing video file from this device.'}</p>
                <button className="button button-secondary" type="button" onClick={(event) => { event.stopPropagation(); fileInput.current?.click(); }}>
                  <Icon name="folder" /> {file ? 'Change video' : 'Choose a video'}
                </button>
                <div className="dropzone-footnote">MP4 · MKV · MOV · WEBM · YOUTUBE · TWITCH</div>
              </div>
              <div className="source-divider"><span>OR USE A VIDEO URL</span></div>
              <label className="field-label" htmlFor="source-url">YouTube or Twitch link</label>
              <div className="source-url-row">
                <span className="url-prefix">URL</span>
                <input id="source-url" value={source} onChange={(event) => { setSource(event.target.value); if (event.target.value) setFile(null); }} placeholder="https://www.youtube.com/watch?v=…" autoComplete="url" />
                <button className="icon-button" type="button" aria-label="Paste video URL" onClick={async () => { try { setSource(await navigator.clipboard.readText()); setFile(null); } catch { setError('Clipboard access is unavailable. Paste the link into the URL field.'); } }}><Icon name="arrow" /></button>
              </div>
              <p className="field-hint">Local files upload directly into this PC's job folder. Remote sources are fetched by yt-dlp.</p>
            </article>

            <article className="panel controls-panel">
              <div className="panel-heading"><div><div className="panel-kicker">02 / CREATIVE DIRECTION</div><h2>Shape the cut</h2></div><span className="panel-number">02</span></div>
              <div className="form-row form-row-two">
                <label className="field-group"><span className="field-label">AI provider</span><select value={provider} onChange={(event) => chooseProvider(event.target.value as Provider)}>{PROVIDERS.map((item) => <option value={item.value} key={item.value}>{item.label}{item.value === 'ollama' ? ' · local' : ''}</option>)}</select></label>
                <label className="field-group"><span className="field-label">Model</span><input value={model} onChange={(event) => setModel(event.target.value)} placeholder={DEFAULTS[provider].model} /></label>
              </div>
              <div className="provider-note"><span className="provider-note-dot" />{providerNote}<button type="button" className="text-button" onClick={() => { setSettingsProvider(provider); setSettingsFeedback(null); setSettingsOpen(true); }}>Configure</button></div>
              <label className="field-group direction-field"><span className="field-label">Custom AI direction <span className="optional">OPTIONAL</span></span><textarea value={direction} onChange={(event) => setDirection(event.target.value)} maxLength={2000} rows={3} placeholder="e.g. Prioritize the surprising payoff, keep clips self-contained, and avoid the intro." /></label>
              <div className="form-row form-row-three">
                <label className="field-group"><span className="field-label">Reframe layout</span><select value={layout} onChange={(event) => setLayout(event.target.value as ReframeMode)}><option value="face_track">Auto Face-Track · 9:16</option><option value="podcast_split">Podcast Split-Screen</option><option value="gaming">Gaming Webcam + Gameplay</option><option value="fit_blur">Fit-Blur</option></select></label>
                <label className="field-group"><span className="field-label">Captions</span><select value={captionStyle} onChange={(event) => setCaptionStyle(event.target.value as CaptionStyle)}><option>Hormozi Bold</option><option>Clean Minimal</option><option>Gaming Neon</option><option>Podcast Classic</option></select></label>
                <label className="field-group"><span className="field-label">Export format</span><select value={exportPreset} onChange={(event) => setExportPreset(event.target.value as ExportPreset)}><option value="short_vertical">Vertical · 9:16</option><option value="square">Square · 1:1</option><option value="landscape">Landscape · 16:9</option></select></label>
              </div>
              <div className="form-row form-row-bottom">
                <label className="field-group clip-count-field"><span className="field-label">Clips to find</span><select value={clipCount} onChange={(event) => setClipCount(Number(event.target.value))}><option value={2}>2 clips</option><option value={4}>4 clips</option><option value={6}>6 clips</option><option value={8}>8 clips</option><option value={12}>12 clips</option></select></label>
                <div className="duration-note"><Icon name="clock" /><span>15–60 seconds each</span><span className="duration-separator">·</span><span>CPU transcription</span></div>
              </div>
              {error && <div className="error-banner" role="alert">{error}</div>}
              <button className="button button-primary submit-button" type="submit" disabled={busy || apiOnline === false}>
                <Icon name="spark" /> {busy ? 'Starting job…' : job && !completed ? 'Start another job' : 'Find my best clips'} <Icon name="arrow" />
              </button>
              {apiOnline === false && <p className="field-hint api-hint">Start the local server with <code>run.bat</code> on Windows or <code>./run.sh</code> on Linux, then refresh.</p>}
            </article>
          </form>

          {job && <section className="job-panel panel" aria-live="polite">
            <div className="job-heading"><div><div className="panel-kicker">03 / JOB {job.job_id.slice(0, 8).toUpperCase()}</div><h2>{job.status === 'completed' ? 'Your clips are ready' : job.status === 'partial' ? 'Your job finished with notes' : job.status === 'failed' ? 'The job needs attention' : 'Finding the moments'}</h2></div><span className={`status-badge status-${job.status}`}><span className="status-dot" />{job.status.toUpperCase()}</span></div>
            <div className="progress-track" role="progressbar" aria-label="Job progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}><span style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
            <div className="progress-caption"><span>{job.message || job.stage}</span><strong>{Math.round(job.progress * 100)}%</strong></div>
            {job.error && <div className="error-banner" role="alert">{job.error}</div>}
            {job.warnings.length > 0 && <details className="warnings"><summary>{job.warnings.length} note{job.warnings.length === 1 ? '' : 's'} from this run</summary><ul>{job.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></details>}
            {job.clips.length > 0 && <div className="clip-grid" id="exports">{job.clips.map((clip) => <article className="clip-card" key={clip.clip_id}>
              <div className="clip-preview"><video src={clip.download_url} controls preload="metadata" playsInline aria-label={`Preview: ${clip.hook_title}`} /><span className="clip-score">{Math.round(clip.virality_score)}<small> SCORE</small></span></div>
              <div className="clip-copy"><div className="clip-meta"><span>{formatTime(clip.start)}–{formatTime(clip.end)}</span><span>{clip.encoder}</span></div><h3>{clip.hook_title}</h3><p>{clip.reason}</p><div className="clip-actions"><span className="caption-preview">“{clip.suggested_caption}”</span><div className="clip-links"><a className="download-link" href={clip.download_url} download={clip.file_name}><Icon name="download" /> Download</a><button className="delete-button" type="button" disabled={deletingClipId === clip.clip_id} onClick={() => void deleteClip(clip)}><Icon name="trash" />{deletingClipId === clip.clip_id ? 'Deleting…' : 'Delete'}</button></div></div></div>
            </article>)}</div>}
            {job.status !== 'running' && job.clips.length === 0 && job.status !== 'failed' && <p className="empty-clips">No clips remain in this job. Your other exports are unaffected.</p>}
          </section>}

          <section className="runtime-strip" aria-label="Local runtime">
            <div className="runtime-strip-heading"><span className="panel-kicker">LOCAL RUNTIME</span><span className="runtime-strip-state"><span className="status-dot" /> {apiOnline ? 'CONNECTED' : apiOnline === false ? 'OFFLINE' : 'CHECKING'}</span></div>
            <div className="runtime-grid">
              <div className="runtime-row"><span className="runtime-index">A</span><span className="runtime-label">Python</span><span className="runtime-value">{runtime?.python ?? '3.11.x pinned'}</span></div>
              <div className="runtime-row"><span className="runtime-index">B</span><span className="runtime-label">H.264 encoder</span><span className="runtime-value runtime-value-accent">{encoderLabel}</span></div>
              <div className="runtime-row"><span className="runtime-index">C</span><span className="runtime-label">Transcription</span><span className="runtime-value">faster-whisper · CPU/int8</span></div>
              <div className="runtime-row"><span className="runtime-index">D</span><span className="runtime-label">Storage</span><span className="runtime-value">Per-user local jobs</span></div>
            </div>
          </section>

          <section className="coming-strip" aria-label="Pipeline stages">
            <div className="coming-icon"><Icon name="spark" /></div>
            <div className="coming-copy"><span className="panel-kicker">ONE SHARED PIPELINE</span><strong>Ingest → AI selection → reframing → captions → render</strong></div>
            <span className="coming-phase">LOCAL-FIRST</span>
          </section>
          <footer className="page-footer"><span>FRAME / FORGE <span className="footer-separator">·</span> LOCAL VIDEO STUDIO</span><span>BUILT FOR THE MOMENTS THAT MATTER</span></footer>
        </div>
      </main>

      {settingsOpen && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !settingsBusy) setSettingsOpen(false); }}>
        <section className="settings-drawer" role="dialog" aria-modal="true" aria-labelledby="settings-title">
          <header className="settings-header"><div><span className="panel-kicker">LOCAL PROVIDER CONFIGURATION</span><h2 id="settings-title">Studio settings</h2></div><button className="icon-button close-settings" type="button" aria-label="Close settings" onClick={() => setSettingsOpen(false)}><Icon name="close" /></button></header>
          <p className="settings-intro">Provider keys are saved in this user's local FrameForge data folder and are never returned to the browser after saving.</p>
          <form className="settings-form" onSubmit={saveAndTestSettings}>
            <label className="field-group"><span className="field-label">AI provider</span><select value={settingsProvider} onChange={(event) => { setSettingsFeedback(null); setSettingsProvider(event.target.value as Provider); }}>{PROVIDERS.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label>
            <label className="field-group"><span className="field-label">Model name</span><input value={settingsModel} onChange={(event) => setSettingsModel(event.target.value)} required maxLength={200} /></label>
            <label className="field-group"><span className="field-label">{settingsProvider === 'ollama' ? 'Ollama base URL' : 'API base URL'}</span><input value={settingsBaseUrl} onChange={(event) => setSettingsBaseUrl(event.target.value)} placeholder={DEFAULTS[settingsProvider].base_url || 'https://provider.example/v1'} autoComplete="url" /></label>
            {API_KEY_LABELS[settingsProvider] && <div className="field-group key-field-group">
              <span className="field-label">{API_KEY_LABELS[settingsProvider]}</span>
              <div className="key-input-row">
                <input ref={settingsKeyInput} type={showKey ? 'text' : 'password'} value={settingsKey} onChange={(event) => setSettingsKey(event.target.value)} placeholder={activeSettingsConfig.api_key_saved ? 'Key saved · leave blank to keep it' : 'Paste API key'} autoComplete="new-password" spellCheck={false} />
                <button className="icon-button" type="button" aria-label={showKey ? 'Hide API key' : 'Show API key'} title={showKey ? 'Hide key' : 'Show key'} onClick={() => setShowKey((value) => !value)}><Icon name="eye" /></button>
                <button className="icon-button" type="button" aria-label="Paste API key" title="Paste key" onClick={() => void pasteApiKey()}><Icon name="paste" /></button>
              </div>
              <p className="field-hint">{activeSettingsConfig.configured ? 'A key is available locally. Enter a new one to replace it.' : 'Keys are stored on this device, not in job records or the browser.'}</p>
            </div>}
            {settingsFeedback && <div className={`settings-feedback ${settingsFeedback.ok ? 'feedback-ok' : 'feedback-error'}`} role="status" aria-live="polite">{settingsFeedback.message}</div>}
            <button className="button button-primary settings-submit" type="submit" disabled={settingsBusy}>
              <Icon name="spark" /> {settingsBusy ? 'Saving and testing…' : 'Save & Test'}
            </button>
            <p className="settings-footnote">Your settings are saved before the connection test, so a temporary provider outage will not discard the configuration.</p>
          </form>
        </section>
      </div>}
    </div>
  );
}

export default App;
