import { useState } from 'react'
import Header from './components/Header'
import InputForm from './components/InputForm'
import ResultDashboard from './components/ResultDashboard'
import { AlertCircle } from 'lucide-react'

export default function App() {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(formData),
      })

      let data
      try {
        data = await res.json()
      } catch {
        throw new Error(
          res.status === 0 || !res.status
            ? 'Cannot reach Flask server — run: python app.py'
            : `Server error (HTTP ${res.status}) — check Flask console for traceback`
        )
      }

      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 text-white">
      {/* Ambient background blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-600/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10">
        <Header />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 pb-16">
          {error && (
            <div className="mb-6 flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300 animate-fade-in">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          <div className={`grid gap-8 ${result ? 'lg:grid-cols-2' : 'max-w-2xl mx-auto'}`}>
            <InputForm onPredict={handlePredict} loading={loading} />
            {result && <ResultDashboard result={result} />}
          </div>
        </main>
      </div>
    </div>
  )
}
