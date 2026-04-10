import { useEffect, useState } from 'react'
import InputPanel from './components/InputPanel'
import ModeToggle from './components/ModeToggle'
import ResultPanel from './components/ResultPanel'

export default function App() {
  const [mode, setMode] = useState('tiktok')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [statusData, setStatusData] = useState(null)

  function resetJob() {
    setInput('')
    setResult(null)
    setStatusData(null)
    setLoading(false)
  }

  useEffect(() => {
    if (!result?.job_id) return
    if (statusData?.status === 'complete' || statusData?.status === 'error') return

    const id = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${result.job_id}`)
        const data = await res.json()
        setStatusData(data)
      } catch {
        // Keep polling; transient network issues are expected.
      }
    }, 2000)

    return () => clearInterval(id)
  }, [result, statusData?.status])

  async function submit() {
    setLoading(true)
    setStatusData(null)
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, input }),
      })
      const raw = await res.text()
      let data
      try {
        data = raw ? JSON.parse(raw) : {}
      } catch {
        data = { detail: raw || 'Server returned a non-JSON response.' }
      }
      if (!res.ok) {
        throw new Error(data.detail || 'Generation failed')
      }
      setResult(data)
    } catch (err) {
      setResult({
        title: 'Failed to start job',
        viral_score: 0,
        clip_duration: 0,
        start_time: 0,
        reason: err.message,
        job_id: 'error',
        status: 'error',
      })
      setStatusData({ status: 'error', error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="header">
        <h1>TikTok Clip Factory</h1>
        <p>Turn a niche idea or a YouTube link into a ready-to-upload vertical clip with karaoke subtitles.</p>
      </header>

      <ModeToggle mode={mode} onChange={setMode} />

      <div className="grid">
        <InputPanel mode={mode} input={input} setInput={setInput} onSubmit={submit} loading={loading} />
        <ResultPanel result={result} statusData={statusData} onNewClip={resetJob} />
      </div>
    </main>
  )
}
