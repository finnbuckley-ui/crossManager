export default function ModeToggle({ mode, onChange }) {
  return (
    <div className="row" style={{ marginBottom: 14 }}>
      <button
        className={`mode-btn ${mode === 'tiktok' ? 'active' : ''}`}
        onClick={() => onChange('tiktok')}
      >
        TikTok page description
      </button>
      <button
        className={`mode-btn ${mode === 'youtube' ? 'active' : ''}`}
        onClick={() => onChange('youtube')}
      >
        YouTube video link
      </button>
    </div>
  )
}
