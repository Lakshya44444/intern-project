import { Truck, Zap, LogOut, User } from 'lucide-react'

export default function Header({ user, onLogout }) {
  return (
    <header className="py-8 px-4 sm:px-6 max-w-7xl mx-auto mb-4">
      <div className="flex items-center justify-between">
        {/* Logo + title */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Truck className="w-5 h-5 text-white" />
            </div>
            <div className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center">
              <Zap className="w-2 h-2 text-white" />
            </div>
          </div>

          <div>
            <h1 className="text-xl font-bold tracking-tight">
              <span className="text-gradient">FreightIQ</span>
              <span className="text-white/70 font-medium ml-1.5">Elite</span>
              <span className="ml-1.5 text-xs font-mono text-cyan-400/80 bg-cyan-400/10 px-1.5 py-0.5 rounded-md border border-cyan-400/20">
                v8.6
              </span>
            </h1>
            <p className="text-xs text-white/40 mt-0.5">
              AI-Driven Logistics Pricing Engine
            </p>
          </div>
        </div>

        {/* Right side: model stats + user */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-3">
            <StatBadge label="Risk AUC" value="0.91" color="emerald" />
            <StatBadge label="Pricing R²" value="0.93" color="blue" />
          </div>

          {user && (
            <div className="flex items-center gap-2 pl-3 border-l border-white/10">
              {/* User email chip */}
              <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.05] border border-white/10">
                <User className="w-3 h-3 text-white/40" />
                <span className="text-xs text-white/50 max-w-[140px] truncate">
                  {user.email}
                </span>
              </div>

              {/* Logout button */}
              <button
                onClick={onLogout}
                title="Sign out"
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-white/10
                           text-white/40 hover:text-red-400 hover:border-red-500/30 hover:bg-red-500/5
                           transition-all duration-200 text-xs"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Sign out</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Sub-heading */}
      <div className="mt-6 text-center">
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
          Probabilistic Freight Pricing
        </h2>
        <p className="mt-2 text-white/50 text-sm max-w-xl mx-auto">
          Two-stage AI stack predicts delay risk, then converts it into a
          5-layer optimal quote — in real time.
        </p>
      </div>
    </header>
  )
}

function StatBadge({ label, value, color }) {
  const colors = {
    emerald: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
    blue:    'text-blue-400 bg-blue-400/10 border-blue-400/20',
  }
  return (
    <div className={`px-3 py-1.5 rounded-lg border text-xs font-mono ${colors[color]}`}>
      <span className="text-white/50">{label}: </span>
      <span className="font-semibold">{value}</span>
    </div>
  )
}
