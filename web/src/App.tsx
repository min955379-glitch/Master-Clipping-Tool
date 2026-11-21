import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react';

type Provider = 'ollama' | 'gemini';
type ReframeMode = 'face_track' | 'podcast_split' | 'gaming' | 'fit_blur';
type CaptionStyle = 'Hormozi Bold' | 'Clean Minimal' | 'Gaming Neon' | 'Podcast Classic';
type ExportPreset = 'short_vertical' | 'square' | 'landscape';
type JobStatus = 'queued' | 'running' | 'completed' | 'partial' | 'failed';

type Clip = {
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
  encoder?: { available: boolean; selected?: string; hardware_accelerated?: boolean; error?: string };
};

const VIDEO_ACCEPT = '.mp4,.mkv,.mov,.webm,.avi,.m4v,.mpeg,.mpg,.ts';

function Icon({ name }: { name: 'film' | 'spark' | 'folder' | 'arrow' | 'upload' | 'download' | 'clock' }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  if (name === 'film') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v4m5-4v4m5-4v4M7 16v4m5-4v4m5-4v4M3 9h18m-18 6h18"/></svg>;
  if (name === 'folder') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5v-10Z"/><path d="M3.5 10h17"/></svg>;
  if (name === 'arrow') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M5 12h14m-6-6 6 6-6 6"/></svg>;
  if (name === 'upload') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M12 16V4m-5 5 5-5 5 5"/><path d="M5 14v5h14v-5"/></svg>;
  if (name === 'download') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><path d="M12 4v12m-5-5 5 5 5-5"/><path d="M5 19h14"/></svg>;
  if (name === 'clock') return <svg viewBox="0 0 24 24" aria-hidden="true" {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>;
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
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState('');
  const [provider, setProvider] = useState<Provider>('ollama');
  const [model, setModel] = useState('qwen2.5:7b');
  const [direction, setDirection] = useState('');
  const [layout, setLayout] = useState<ReframeMode>('face_track');
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>('Hormozi Bold');
  const [exportPreset, setExportPreset] = useState<ExportPreset>('short_vertical');
  const [clipCount, setClipCount] = useState(4);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
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
    void checkHealth();
    void loadRuntime();
    const healthTimer = window.setInterval(checkHealth, 15000);
    const runtimeTimer = window.setInterval(loadRuntime, 60000);
    return () => {
      alive = false;
      window.clearInterval(healthTimer);
      window.clearInterval(runtimeTimer);
    };
  }, []);

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
            source: source.trim(), provider, model: model.trim() || null,
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

  const completed = job && ['completed', 'partial', 'failed'].includes(job.status);
  const encoderLabel = runtime?.encoder?.selected ?? 'Checking FFmpeg';
  const providerNote = provider === 'gemini'
    ? runtime?.gemini_configured ? 'Gemini key found in the local environment.' : 'Set GEMINI_API_KEY on this PC to enable Gemini.'
    : `Local Ollama · ${runtime?.ollama_url ?? 'http://localhost:11434'}`;

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
          <p>Source media and exports stay on this machine. Gemini is used only when you opt in.</p>
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
            <span className="avatar">FF</span>
          </div>
        </header>

        <div className="content-wrap" id="studio">
          <section className="intro-section">
            <div className="intro-copy">
              <div className="eyebrow"><span className="eyebrow-line" /> THE CREATOR'S CUTTING ROOM</div>
              <h1>Find the moment<br />that <span>moves</span> the room.</h1>
              <p className="intro-description">Turn a long-form video into ranked, reframed short-form stories. Processing runs locally; Gemini is an explicit cloud option.</p>
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
                <label className="field-group"><span className="field-label">AI provider</span><select value={provider} onChange={(event) => { const next = event.target.value as Provider; setProvider(next); setModel(next === 'ollama' ? 'qwen2.5:7b' : 'gemini-2.5-flash'); }}><option value="ollama">Ollama · local</option><option value="gemini">Google Gemini · opt-in</option></select></label>
                <label className="field-group"><span className="field-label">Model (optional)</span><input value={model} onChange={(event) => setModel(event.target.value)} placeholder={provider === 'ollama' ? 'qwen2.5:7b' : 'gemini-2.5-flash'} /></label>
              </div>
              <div className={`provider-note ${provider === 'gemini' ? 'provider-note-gemini' : ''}`}><span className="provider-note-dot" />{providerNote}</div>
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
              {apiOnline === false && <p className="field-hint api-hint">Start the local API with <code>run.bat</code>, then refresh this page.</p>}
            </article>
          </form>

          {job && <section className="job-panel panel" aria-live="polite">
            <div className="job-heading"><div><div className="panel-kicker">03 / JOB {job.job_id.slice(0, 8).toUpperCase()}</div><h2>{job.status === 'completed' ? 'Your clips are ready' : job.status === 'partial' ? 'Your job finished with notes' : job.status === 'failed' ? 'The job needs attention' : 'Finding the moments'}</h2></div><span className={`status-badge status-${job.status}`}><span className="status-dot" />{job.status.toUpperCase()}</span></div>
            <div className="progress-track" role="progressbar" aria-label="Job progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}><span style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
            <div className="progress-caption"><span>{job.message || job.stage}</span><strong>{Math.round(job.progress * 100)}%</strong></div>
            {job.error && <div className="error-banner" role="alert">{job.error}</div>}
            {job.warnings.length > 0 && <details className="warnings"><summary>{job.warnings.length} note{job.warnings.length === 1 ? '' : 's'} from this run</summary><ul>{job.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></details>}
            {job.clips.length > 0 && <div className="clip-grid" id="exports">{job.clips.map((clip) => <article className="clip-card" key={clip.file_name}>
              <div className="clip-preview"><video src={clip.download_url} controls preload="metadata" playsInline aria-label={`Preview: ${clip.hook_title}`} /><span className="clip-score">{Math.round(clip.virality_score)}<small> SCORE</small></span></div>
              <div className="clip-copy"><div className="clip-meta"><span>{formatTime(clip.start)}–{formatTime(clip.end)}</span><span>{clip.encoder}</span></div><h3>{clip.hook_title}</h3><p>{clip.reason}</p><div className="clip-actions"><span className="caption-preview">“{clip.suggested_caption}”</span><a className="download-link" href={clip.download_url} download={clip.file_name}><Icon name="download" /> Download MP4</a></div></div>
            </article>)}</div>}
          </section>}

          <section className="runtime-strip" aria-label="Local runtime">
            <div className="runtime-strip-heading"><span className="panel-kicker">LOCAL RUNTIME</span><span className="runtime-strip-state"><span className="status-dot" /> {apiOnline ? 'CONNECTED' : apiOnline === false ? 'OFFLINE' : 'CHECKING'}</span></div>
            <div className="runtime-grid">
              <div className="runtime-row"><span className="runtime-index">A</span><span className="runtime-label">Python</span><span className="runtime-value">{runtime?.python ?? '3.11.x pinned'}</span></div>
              <div className="runtime-row"><span className="runtime-index">B</span><span className="runtime-label">H.264 encoder</span><span className="runtime-value runtime-value-gold">{encoderLabel}</span></div>
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
    </div>
  );
}

export default App;
