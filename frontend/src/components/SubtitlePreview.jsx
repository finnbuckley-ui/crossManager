import { useEffect, useMemo, useState } from 'react'

export default function SubtitlePreview({ words = [] }) {
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)

  const duration = useMemo(() => {
    if (!words.length) return 0
    return words[words.length - 1].end || 0
  }, [words])

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setCurrentTime((t) => {
        const next = t + 0.1
        if (duration && next >= duration) {
          setPlaying(false)
          return duration
        }
        return next
      })
    }, 100)
    return () => clearInterval(id)
  }, [playing, duration])

  useEffect(() => {
    setCurrentTime(0)
    setPlaying(false)
  }, [words])

  return (
    <section className="panel">
      <h3 style={{ marginTop: 0 }}>Subtitle Preview</h3>
      <div className="subtitle-box">
        {!words.length && <span className="meta">Transcript will appear after processing.</span>}
        {words.map((w, i) => {
          let cls = 'word'
          if (currentTime > w.end) cls += ' past'
          else if (currentTime >= w.start && currentTime <= w.end) cls += ' current'
          return (
            <span key={`${w.start}-${i}`} className={cls}>
              {w.word}
            </span>
          )
        })}
      </div>

      <div className="control-row">
        <button className="mode-btn" onClick={() => setPlaying((v) => !v)} disabled={!words.length}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={currentTime}
          onChange={(e) => setCurrentTime(Number(e.target.value))}
          disabled={!words.length}
        />
      </div>
    </section>
  )
}
