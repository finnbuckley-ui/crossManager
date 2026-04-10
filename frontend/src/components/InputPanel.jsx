const EXAMPLES = [
  'Fitness motivation',
  'True crime',
  'Finance tips',
  'Cooking & recipes',
  'Dark humour',
]

export default function InputPanel({ mode, input, setInput, onSubmit, loading }) {
  return (
    <section className="panel">
      <h3 style={{ marginTop: 0 }}>Input</h3>
      {mode === 'tiktok' ? (
        <>
          <textarea
            placeholder="Describe your TikTok niche..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <div className="row" style={{ marginTop: 10 }}>
            {EXAMPLES.map((item) => (
              <button key={item} className="chip" onClick={() => setInput(item)}>
                {item}
              </button>
            ))}
          </div>
        </>
      ) : (
        <input
          type="text"
          placeholder="https://www.youtube.com/watch?v=..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      )}

      <button className="primary" onClick={onSubmit} disabled={loading || !input.trim()}>
        {loading ? 'Finding viral clip...' : 'Find viral clip'}
      </button>
    </section>
  )
}
