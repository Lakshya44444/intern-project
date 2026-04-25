import { Truck, Zap } from 'lucide-react'

export default function Header() {
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

        {/* Status badges */}
        <div className="hidden sm:flex items-center gap-3">
          <StatBadge label="Risk AUC" value="0.88" color="emerald" />
          <StatBadge label="Pricing R²" value="0.91" color="blue" />
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
