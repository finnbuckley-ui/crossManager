import SubtitlePreview from './SubtitlePreview'

function toTime(seconds) {
  const sec = Number(seconds || 0)
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

function toEta(seconds) {
  const sec = Math.max(0, Number(seconds || 0))
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}m ${String(s).padStart(2, '0')}s`
}

export default function ResultPanel({ result, statusData, onNewClip }) {
  if (!result) {
    return (
      <section className="panel">
        <h3 style={{ marginTop: 0 }}>Result</h3>
        <p className="meta">Generate a clip to see analysis, progress, and download options.</p>
      </section>
    )
  }

  const status = statusData?.status || result.status
  const words = statusData?.subtitle_words || []
  const videoId = statusData?.video_id || result.video_id
  const startTime = statusData?.start_time ?? result.start_time
  const elapsedSeconds = Number(statusData?.elapsed_seconds || 0)
  const fallbackTotal = Math.max(90, Math.round(75 + Number(result?.clip_duration || 55) * 2.2))
  const etaSeconds =
    typeof statusData?.eta_seconds === 'number'
      ? statusData.eta_seconds
      : typeof result?.eta_seconds === 'number'
        ? result.eta_seconds
        : Math.max(5, fallbackTotal - elapsedSeconds)

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <section className="panel">
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>{result.title}</h3>
          <span className="badge">Viral score: {result.viral_score}/10</span>
        </div>

        <p className="meta" style={{ marginTop: 8 }}>
          Start: {toTime(startTime)} | Duration: {result.clip_duration}s
        </p>
        <p className="meta">{result.reason}</p>

        {status === 'processing' && (
          <>
            <p className="meta">Processing video, smart crop, transcript, and subtitle burn-in...</p>
            {typeof etaSeconds === 'number' && <p className="meta">ETA: ~{toEta(etaSeconds)}</p>}
            <div className="progress">
              <div />
            </div>
          </>
        )}

        {status === 'error' && <p className="error">{statusData?.error || 'Processing failed.'}</p>}

        {status === 'complete' && (
          <div className="row">
            <a className="primary" style={{ display: 'inline-block', textDecoration: 'none' }} href={`/api/download/${result.job_id}`}>
              Download .mp4
            </a>
            <button className="mode-btn" type="button" onClick={onNewClip}>
              New clip / new source
            </button>
          </div>
        )}

        {status === 'error' && (
          <div className="row">
            <button className="mode-btn" type="button" onClick={onNewClip}>
              Try another source
            </button>
          </div>
        )}
      </section>

      <section className="panel">
        <h3 style={{ marginTop: 0 }}>Source Preview (9:16)</h3>
        <div className="video-frame">
          {videoId ? (
            <iframe
              title="youtube-preview"
              src={`https://www.youtube.com/embed/${videoId}?start=${startTime}&autoplay=0&controls=1&modestbranding=1`}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          ) : (
            <div className="meta" style={{ padding: 12 }}>No video source yet.</div>
          )}
        </div>
      </section>

      <SubtitlePreview words={words} />
    </div>
  )
}
