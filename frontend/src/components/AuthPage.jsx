import { useState } from 'react'
import { supabase } from '../lib/supabaseClient'
import { Truck, Zap, Mail, Lock, Eye, EyeOff, Loader2, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react'

export default function AuthPage() {
  const [tab, setTab]           = useState('signin')   // 'signin' | 'signup'
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw]     = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [success, setSuccess]   = useState(null)

  const reset = () => { setError(null); setSuccess(null) }

  const handleSignIn = async (e) => {
    e.preventDefault()
    setLoading(true); reset()
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) {
      setError(error.message)
      setPassword('')
    }
    setLoading(false)
  }

  const handleSignUp = async (e) => {
    e.preventDefault()
    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }
    setLoading(true); reset()
    const { error } = await supabase.auth.signUp({ email, password })
    if (error) {
      setError(error.message)
    } else {
      setSuccess('Account created! Check your email to confirm, then sign in.')
      setTab('signin')
      setPassword('')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 text-white
                    flex flex-col items-center justify-center px-4">

      {/* Ambient blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 w-full max-w-sm space-y-6">

        {/* Logo */}
        <div className="text-center space-y-3">
          <div className="flex justify-center">
            <div className="relative">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl
                              flex items-center justify-center shadow-xl shadow-blue-500/30">
                <Truck className="w-7 h-7 text-white" />
              </div>
              <div className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-gradient-to-br from-amber-400 to-orange-500
                              rounded-full flex items-center justify-center shadow-lg">
                <Zap className="w-3 h-3 text-white" />
              </div>
            </div>
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              <span className="text-gradient">FreightIQ</span>
              <span className="text-white/70 font-medium ml-1.5">Elite</span>
            </h1>
            <p className="text-sm text-white/40 mt-1">AI-Driven Logistics Pricing Engine</p>
          </div>
        </div>

        {/* Card */}
        <div className="glass p-6 space-y-5">

          {/* Tab switcher */}
          <div className="flex rounded-xl overflow-hidden border border-white/10 bg-white/[0.03]">
            {[
              { id: 'signin', label: 'Sign In' },
              { id: 'signup', label: 'Create Account' },
            ].map(t => (
              <button
                key={t.id}
                type="button"
                onClick={() => { setTab(t.id); reset() }}
                className={`flex-1 py-2.5 text-sm font-medium transition-all duration-200 ${
                  tab === t.id
                    ? 'bg-cyan-500/20 text-cyan-400 border-b-2 border-cyan-500'
                    : 'text-white/40 hover:text-white/60'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Success banner */}
          {success && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-sm">
              <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{success}</span>
            </div>
          )}

          {/* Error banner */}
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl bg-red-500/10 border border-red-500/25 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={tab === 'signin' ? handleSignIn : handleSignUp} className="space-y-4">

            {/* Email */}
            <div className="space-y-1.5">
              <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
                <Mail className="w-3.5 h-3.5 text-cyan-400" />
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="you@example.com"
                className="auth-input"
              />
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
                <Lock className="w-3.5 h-3.5 text-purple-400" />
                Password
                {tab === 'signup' && (
                  <span className="text-white/25 font-normal ml-1">(min 6 chars)</span>
                )}
              </label>
              <div className="relative">
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete={tab === 'signin' ? 'current-password' : 'new-password'}
                  placeholder="••••••••"
                  className="auth-input pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(p => !p)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/25 hover:text-white/50 transition-colors"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading || !email || !password}
              className="w-full py-3 rounded-xl font-semibold text-sm
                         bg-gradient-to-r from-cyan-500 to-blue-600
                         hover:from-cyan-400 hover:to-blue-500
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all duration-200 shadow-lg shadow-blue-500/25
                         flex items-center justify-center gap-2 mt-1"
            >
              {loading
                ? <><Loader2 className="w-4 h-4 animate-spin" /> {tab === 'signin' ? 'Signing in…' : 'Creating account…'}</>
                : <><Sparkles className="w-4 h-4" /> {tab === 'signin' ? 'Sign In' : 'Create Account'}</>
              }
            </button>
          </form>

          {/* Footer hint */}
          <p className="text-center text-xs text-white/25">
            {tab === 'signin'
              ? <>No account? <button onClick={() => { setTab('signup'); reset() }}
                  className="text-cyan-400/70 hover:text-cyan-400 underline underline-offset-2">
                  Create one
                </button></>
              : <>Already have one? <button onClick={() => { setTab('signin'); reset() }}
                  className="text-cyan-400/70 hover:text-cyan-400 underline underline-offset-2">
                  Sign in
                </button></>
            }
          </p>
        </div>

        <p className="text-center text-xs text-white/15">
          FreightIQ Elite v8.6 · Secure login via Supabase Auth
        </p>
      </div>

      <style>{`
        .auth-input {
          width: 100%;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 0.75rem;
          padding: 0.625rem 0.875rem;
          color: white;
          font-size: 0.875rem;
          outline: none;
          transition: border-color 0.2s;
        }
        .auth-input:focus { border-color: rgba(34,211,238,0.5); }
        .auth-input::placeholder { color: rgba(255,255,255,0.2); }
        .auth-input:-webkit-autofill {
          -webkit-box-shadow: 0 0 0 100px #0f172a inset;
          -webkit-text-fill-color: white;
        }
      `}</style>
    </div>
  )
}
