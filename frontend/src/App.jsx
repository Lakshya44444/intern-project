import { useState, useEffect } from 'react'
import { supabase } from './lib/supabaseClient'
import Header from './components/Header'
import InputForm from './components/InputForm'
import ResultDashboard from './components/ResultDashboard'
import AuthPage from './components/AuthPage'
import { AlertCircle, Loader2 } from 'lucide-react'

export default function App() {
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [user, setUser]           = useState(null)
  const [authReady, setAuthReady] = useState(false)   // true once session is known

  // Resolve session on mount, then listen for changes
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setAuthReady(true)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => subscription.unsubscribe()
  }, [])

  const handleLogout = async () => {
    await supabase.auth.signOut()
    setResult(null)
    setError(null)
  }

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    try {
      // Attach current JWT so the backend can verify the user
      const { data: { session } } = await supabase.auth.getSession()
      const headers = { 'Content-Type': 'application/json' }
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`
      }

      const res = await fetch('/api/predict', {
        method:  'POST',
        headers,
        body:    JSON.stringify(formData),
      })

      let data
      try {
        data = await res.json()
      } catch {
        throw new Error(
          res.status === 0 || !res.status
            ? 'Cannot reach API server — run: python app.py (or uvicorn app:app --port 5000)'
            : `Server error (HTTP ${res.status}) — check the API server console`
        )
      }

      if (res.status === 401) {
        // Token expired — force re-login
        await supabase.auth.signOut()
        throw new Error('Session expired. Please sign in again.')
      }

      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Splash while checking stored session
  if (!authReady) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950
                      flex items-center justify-center">
        <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
      </div>
    )
  }

  // Not logged in — show auth page
  if (!user) return <AuthPage />

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 text-white">
      {/* Ambient background blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-600/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10">
        <Header user={user} onLogout={handleLogout} />

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
