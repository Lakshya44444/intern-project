import {
  ArrowRight, Cpu, Fuel, Shield, TrendingUp,
  IndianRupee, CheckCircle2, AlertTriangle,
  Navigation, Thermometer, Gauge, BarChart3,
  Clock, Package2,
} from 'lucide-react'

const fmt  = n => `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
const fpct = n => `${(n * 100).toFixed(1)}%`

export default function ResultDashboard({ result }) {
  if (!result?.pricing?.operational || !result?.abstraction) {
    return (
      <div className="glass p-6 text-red-400/70 text-sm text-center">
        Unexpected response from server — please try again.
      </div>
    )
  }

  const {
    origin, destination, trip_date, weight_tonnes,
    abstraction, efficiency_score, efficiency_label,
    delay_probability, p_return, p_return_model,
    confidence_score, confidence_label,
    pricing, addons, final_price,
  } = result

  return (
    <div className="space-y-4 animate-slide-up">

      {/* Route banner */}
      <div className="glass p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 font-medium">
          <span className="text-white/80">{origin}</span>
          <ArrowRight className="w-4 h-4 text-cyan-400 flex-shrink-0" />
          <span className="text-white/80">{destination}</span>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={result.status} />
          <div className="flex items-center gap-2 text-xs text-white/40 font-mono">
            <span>{abstraction.distance_km} km</span>
            <span>•</span>
            <span>{weight_tonnes}T</span>
            <span>•</span>
            <span>{abstraction.vehicle_type || 'FTL'}</span>
          </div>
        </div>
      </div>

      {/* Intelligence Abstraction Panel */}
      <AbstractionPanel abstraction={abstraction} />

      {/* Confidence + Scores row */}
      <div className="grid grid-cols-3 gap-3">
        <ConfidenceGauge score={confidence_score} label={confidence_label} />
        <MiniScore label="Efficiency" value={fpct(efficiency_score)} sub={efficiency_label}
          color={scoreColor(efficiency_score)} />
        <MiniScore label="Delay Risk" value={fpct(delay_probability)}
          sub={delay_probability < 0.3 ? 'Low Risk' : 'High Risk'}
          color={delay_probability < 0.3 ? '#34d399' : '#f87171'} />
      </div>

      {/* 5-Layer Pricing Breakdown */}
      <div className="glass p-5 space-y-2.5">
        <h4 className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-4">
          5-Layer Pricing Engine (v8.6)
        </h4>

        <Layer1 data={pricing.operational} />
        <Layer2 data={pricing.risk} pReturnModel={p_return_model} mEmpty={pricing.operational.m_empty_kmpl} />
        <Layer3 data={pricing.corridor} />
        <Layer4 data={pricing.corridor} />
        <Layer5 data={pricing.feasibility} />
      </div>

      {/* Add-ons */}
      {addons && addons.selected && addons.selected.length > 0 && (
        <AddonPanel addons={addons} />
      )}

      {/* Final Quote */}
      <FinalQuote
        price={final_price ?? pricing.recommended_price}
        basePrice={pricing.recommended_price}
        addonTotal={addons?.total ?? 0}
        isFeasible={pricing.feasibility.is_feasible}
        note={pricing.feasibility.note}
        floor={pricing.corridor.floor}
        ceiling={pricing.corridor.ceiling}
      />
    </div>
  )
}

/* ── Intelligence Abstraction Panel ────────────────────────────────────────── */
function AbstractionPanel({ abstraction }) {
  const items = [
    { icon: <Navigation className="w-3.5 h-3.5 text-cyan-400" />,
      label: 'Distance',
      value: `${abstraction.distance_km} km`,
      sub: abstraction.distance_method === 'osrm' ? 'road (OSRM)'
         : abstraction.distance_method === 'haversine' ? 'road estimate'
         : abstraction.distance_method === 'historical_avg' ? 'historical avg'
         : 'fallback' },
    { icon: <Thermometer className="w-3.5 h-3.5 text-purple-400" />,
      label: 'Season',
      value: abstraction.season,
      sub: abstraction.day_name },
    { icon: <Gauge className="w-3.5 h-3.5 text-amber-400" />,
      label: 'Loaded Mileage',
      value: `${abstraction.m_loaded_kmpl} km/L`,
      sub: `empty: ${abstraction.m_empty_kmpl} km/L` },
    { icon: <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />,
      label: 'Lane Data',
      value: `${abstraction.lane_trip_count} trips`,
      sub: `pop: ${fpct(abstraction.lane_popularity)} | risk: ${fpct(abstraction.route_risk)}` },
    { icon: <IndianRupee className="w-3.5 h-3.5 text-rose-400" />,
      label: 'Diesel',
      value: `₹${abstraction.diesel_price}/L`,
      sub: abstraction.diesel_source === 'vision' ? 'vision-authenticated' : 'regional default' },
  ]

  return (
    <div className="glass p-4">
      <div className="flex items-center gap-2 mb-3">
        <Cpu className="w-3.5 h-3.5 text-blue-400" />
        <span className="text-xs font-semibold text-white/40 uppercase tracking-widest">
          Intelligence Abstraction
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {items.map(it => (
          <div key={it.label} className="space-y-0.5">
            <div className="flex items-center gap-1.5">
              {it.icon}
              <span className="text-xs text-white/35">{it.label}</span>
            </div>
            <div className="text-sm font-semibold text-white/80">{it.value}</div>
            <div className="text-xs text-white/25">{it.sub}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Confidence Gauge ───────────────────────────────────────────────────────── */
function ConfidenceGauge({ score, label }) {
  const color = confidenceColor(label)
  const pct   = Math.round(score * 100)
  const circ  = 2 * Math.PI * 30
  return (
    <div className="glass p-3 flex flex-col items-center justify-center gap-1">
      <span className="text-xs text-white/35 uppercase tracking-widest">Confidence</span>
      <div className="relative w-16 h-16">
        <svg viewBox="0 0 72 72" className="w-full h-full -rotate-90">
          <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="6" />
          <circle cx="36" cy="36" r="30" fill="none" stroke={color} strokeWidth="6"
            strokeDasharray={`${score * circ} ${circ}`} strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 4px ${color}88)` }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold font-mono" style={{ color }}>{pct}</span>
        </div>
      </div>
      <span className="text-xs font-semibold" style={{ color }}>{label}</span>
    </div>
  )
}

function MiniScore({ label, value, sub, color }) {
  return (
    <div className="glass p-3 flex flex-col items-center justify-center gap-1">
      <span className="text-xs text-white/35 uppercase tracking-widest">{label}</span>
      <span className="text-xl font-bold font-mono" style={{ color }}>{value}</span>
      <span className="text-xs font-medium" style={{ color, opacity: 0.7 }}>{sub}</span>
    </div>
  )
}

/* ── Layer shell ─────────────────────────────────────────────────────────────── */
function LayerShell({ num, icon, label, color, total, children }) {
  const badge = {
    orange: 'text-orange-400 bg-orange-400/10 border-orange-400/20',
    red:    'text-red-400    bg-red-400/10    border-red-400/20',
    blue:   'text-blue-400   bg-blue-400/10   border-blue-400/20',
    violet: 'text-violet-400 bg-violet-400/10 border-violet-400/20',
    emerald:'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  }[color]

  return (
    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-mono px-1.5 py-0.5 rounded border ${badge}`}>L{num}</span>
          {icon}
          <span className="text-xs font-medium text-white/70">{label}</span>
        </div>
        <span className="text-sm font-bold font-mono text-white/80">{fmt(total)}</span>
      </div>
      {children}
    </div>
  )
}

function Row({ label, value, signed }) {
  const num = Number(value)
  const colorClass = signed
    ? (num >= 0 ? 'text-emerald-400' : 'text-red-400')
    : 'text-white/30'
  const prefix = signed && num > 0 ? '+' : ''
  return (
    <div className={`flex justify-between text-xs px-1 ${colorClass}`}>
      <span className={signed ? '' : 'text-white/30'}>{label}</span>
      <span className="font-mono">{prefix}{fmt(num)}</span>
    </div>
  )
}

/* ── Layer 1: Operational Base ──────────────────────────────────────────────── */
function Layer1({ data }) {
  return (
    <LayerShell num="01" color="orange" total={data.c_base} label="Operational Base (C_base)"
      icon={<Fuel className="w-4 h-4 text-orange-400" />}>
      <Row label={`Fuel loaded (${data.m_loaded_kmpl} km/L)`} value={data.fuel_loaded_cost} />
      <Row label="Tolls (round-trip)" value={data.tolls_roundtrip} />
      <Row label="Driver Wages" value={data.driver_wage} />
      <Row label="Maintenance" value={data.maintenance} />
      <Row label="Mobilization" value={data.mobilization} />
    </LayerShell>
  )
}

/* ── Layer 2: Return-Load Risk ──────────────────────────────────────────────── */
function Layer2({ data, pReturnModel, mEmpty }) {
  const model = pReturnModel || {}
  return (
    <LayerShell num="02" color="red" total={data.c_risk} label="Return-Load Risk (C_risk)"
      icon={<Shield className="w-4 h-4 text-red-400" />}>
      <div className="px-1 space-y-1.5">

        {/* p_return bar */}
        <div className="flex items-center justify-between text-xs text-white/30">
          <span>P(return load)</span>
          <div className="flex items-center gap-2">
            <div className="w-20 h-1 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full bg-emerald-500 rounded-full"
                style={{ width: `${data.p_return * 100}%` }} />
            </div>
            <span className="font-mono text-emerald-400">{fpct(data.p_return)}</span>
          </div>
        </div>

        {/* Time-decay details */}
        {model.arrival_time && (
          <div className="flex items-start gap-1.5 p-2 rounded-lg bg-white/[0.03] border border-white/[0.05]">
            <Clock className="w-3 h-3 text-white/25 mt-0.5 flex-shrink-0" />
            <div className="space-y-0.5 text-xs text-white/25 font-mono">
              <div>arrive {model.arrival_time} · dwell ~{model.expected_dwell_h}h</div>
              <div>T_active={model.T_active_h}h · grace={model.T_grace_h}h · λ={model.lambda}</div>
              <div className={model.within_grace ? 'text-emerald-400/60' : 'text-amber-400/60'}>
                {model.within_grace ? 'within grace period — full p_return' : `decay over ${(model.T_active_h - model.T_grace_h).toFixed(1)}h`}
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between text-xs text-white/30">
          <span>P(empty return) = risk exposure</span>
          <span className="font-mono text-red-400">{fpct(data.p_no_return)}</span>
        </div>
        <div className="flex justify-between text-xs text-white/30 px-0.5">
          <span>Empty fuel ({mEmpty ?? '—'} km/L)</span>
          <span className="font-mono">{fmt(data.empty_fuel_val)}</span>
        </div>
        <p className="text-xs text-white/20 italic pt-0.5">
          C_risk = (1 − p_return) × (D ÷ M_empty) × P_diesel
        </p>
      </div>
    </LayerShell>
  )
}

/* ── Layer 3: Pricing Corridor ──────────────────────────────────────────────── */
function Layer3({ data }) {
  const mlFactor = data.ml_multiplier ?? 1.0
  const delayPrem = data.ml_delay_prem_pct ?? 0
  const effAdj    = data.ml_eff_adj_pct ?? 0
  const mlMoved   = Math.abs(mlFactor - 1.0) > 0.001

  return (
    <LayerShell num="03" color="blue" total={data.p_min} label="Pricing Corridor [P_min → P_max]"
      icon={<TrendingUp className="w-4 h-4 text-blue-400" />}>
      <div className="px-1 space-y-1.5">

        {/* Range bar */}
        <div className="space-y-1">
          <div className="relative h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/40 to-emerald-500/30 rounded-full" />
          </div>
          <div className="flex justify-between text-xs font-mono text-white/30">
            <span className="text-blue-400">{fmt(data.p_min)}</span>
            <span className="text-white/15">← corridor →</span>
            <span className="text-emerald-400">{fmt(data.p_max)}</span>
          </div>
        </div>

        <div className="flex justify-between text-xs text-white/30 px-1">
          <span>Demand + urgency premium (15%)</span>
          <span className="font-mono text-emerald-400">+{fmt(data.premium)}</span>
        </div>

        {/* ML Intelligence Multiplier — shows AI influence on price */}
        {mlMoved && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-blue-500/5 border border-blue-500/10">
            <Cpu className="w-3 h-3 text-blue-400/60 flex-shrink-0" />
            <div className="flex-1 space-y-0.5">
              <div className="flex justify-between text-xs text-white/25">
                <span>ML multiplier</span>
                <span className={`font-mono font-semibold ${mlFactor > 1 ? 'text-amber-400/70' : 'text-emerald-400/70'}`}>
                  ×{mlFactor.toFixed(3)}
                </span>
              </div>
              <div className="flex gap-3 text-xs text-white/18 font-mono">
                {delayPrem !== 0 && (
                  <span>delay +{delayPrem.toFixed(1)}%</span>
                )}
                {effAdj !== 0 && (
                  <span>eff {effAdj > 0 ? '+' : ''}{effAdj.toFixed(1)}%</span>
                )}
              </div>
            </div>
          </div>
        )}

        <p className="text-xs text-white/20 italic">
          P_min = (C_base + C_risk) × ML_factor &nbsp;·&nbsp; P_max = P_min × 1.15
        </p>
      </div>
    </LayerShell>
  )
}

/* ── Layer 4: Market Snap ────────────────────────────────────────────────────── */
function Layer4({ data }) {
  return (
    <LayerShell num="04" color="violet" total={data.floor} label="Market Snap (Competitor Bounds)"
      icon={<IndianRupee className="w-4 h-4 text-violet-400" />}>
      <div className="px-1 space-y-1">
        {data.competitor_price ? (
          <>
            <div className="flex justify-between text-xs text-white/30">
              <span>Competitor benchmark</span>
              <span className="font-mono">{fmt(data.competitor_price)}</span>
            </div>
            <div className="flex justify-between text-xs text-white/30">
              <span>Floor (−{data.epsilon_low_pct}%)</span>
              <span className="font-mono text-blue-400">{fmt(data.floor)}</span>
            </div>
            <div className="flex justify-between text-xs text-white/30">
              <span>Ceiling (+{data.epsilon_high_pct}%)</span>
              <span className="font-mono text-emerald-400">{fmt(data.ceiling)}</span>
            </div>
            {data.was_snapped && (
              <p className="text-xs text-amber-400/70 italic">
                ⚠ Corridor snapped to [{fmt(data.floor)}, {fmt(data.ceiling)}]
              </p>
            )}
          </>
        ) : (
          <p className="text-xs text-white/25 italic px-1">
            No benchmark — raw corridor used as-is
          </p>
        )}
      </div>
    </LayerShell>
  )
}

/* ── Layer 5: Feasibility Gate ──────────────────────────────────────────────── */
function Layer5({ data }) {
  return (
    <LayerShell num="05" color="emerald" total={data.recommended} label="Feasibility Gate"
      icon={data.is_feasible
        ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
        : <AlertTriangle className="w-4 h-4 text-amber-400" />}>
      <div className="px-1 space-y-1">
        <div className="flex justify-between text-xs text-white/30">
          <span>Driver survival minimum (C_survival)</span>
          <span className="font-mono">{fmt(data.c_survival)}</span>
        </div>
        <p className={`text-xs italic ${data.is_feasible ? 'text-white/25' : 'text-amber-400/70'}`}>
          {data.note}
        </p>
      </div>
    </LayerShell>
  )
}

/* ── Add-on Panel ────────────────────────────────────────────────────────────── */
function AddonPanel({ addons }) {
  return (
    <div className="glass p-4 space-y-2">
      <div className="flex items-center gap-2">
        <Package2 className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-xs font-semibold text-white/40 uppercase tracking-widest">
          Add-on Services
        </span>
        <span className="ml-auto text-xs font-mono text-emerald-400 font-semibold">
          +{fmt(addons.total)}
        </span>
      </div>
      <div className="space-y-1">
        {Object.entries(addons.breakdown).map(([key, item]) => (
          <div key={key} className="flex justify-between text-xs px-1">
            <div className="flex items-center gap-1.5 text-white/40">
              <CheckCircle2 className="w-3 h-3 text-emerald-400/60" />
              {item.label}
            </div>
            <span className="font-mono text-emerald-400">+{fmt(item.amount)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Final Quote ─────────────────────────────────────────────────────────────── */
function FinalQuote({ price, basePrice, addonTotal, isFeasible, note, floor, ceiling }) {
  const pct = floor < ceiling ? ((basePrice - floor) / (ceiling - floor)) * 100 : 50

  return (
    <div className="glass-strong p-6 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/8 via-transparent to-blue-600/8 pointer-events-none" />

      <div className="relative space-y-4">
        <div className="flex items-center justify-center gap-2">
          <IndianRupee className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-widest">
            Recommended Quote
          </span>
        </div>

        <div className="text-center">
          <div className="text-5xl font-bold font-mono text-gradient">{fmt(price)}</div>
          {addonTotal > 0 && (
            <p className="text-xs text-white/30 mt-1 font-mono">
              {fmt(basePrice)} base + {fmt(addonTotal)} add-ons
            </p>
          )}
          <p className="text-xs text-white/35 mt-2">{note}</p>
        </div>

        {/* Corridor position bar */}
        <div className="space-y-1.5">
          <div className="relative h-2 bg-white/5 rounded-full overflow-visible mx-1">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/15 to-emerald-500/10 rounded-full" />
            <div className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-cyan-400 rounded-full
                           shadow-lg shadow-cyan-400/60 border-2 border-slate-950 transition-all duration-700"
              style={{ left: `calc(${Math.min(95, Math.max(5, pct))}% - 6px)` }}
            />
          </div>
          <div className="flex justify-between text-xs font-mono text-white/25">
            <span>{fmt(floor)}</span>
            <span className="text-white/15">effective corridor</span>
            <span>{fmt(ceiling)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Status Badge ────────────────────────────────────────────────────────────── */
function StatusBadge({ status }) {
  const cfg = {
    'Healthy':    { cls: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25', dot: '#34d399' },
    'Low Margin': { cls: 'text-amber-400  bg-amber-400/10  border-amber-400/25',  dot: '#fbbf24' },
    'Infeasible': { cls: 'text-red-400    bg-red-400/10    border-red-400/25',    dot: '#f87171' },
  }[status] || { cls: 'text-white/40 bg-white/5 border-white/10', dot: '#94a3b8' }

  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold ${cfg.cls}`}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: cfg.dot }} />
      {status}
    </div>
  )
}

/* ── Helpers ──────────────────────────────────────────────────────────────────── */
function scoreColor(s) {
  if (s >= 0.75) return '#34d399'
  if (s >= 0.50) return '#22d3ee'
  if (s >= 0.35) return '#fbbf24'
  return '#f87171'
}

function confidenceColor(label) {
  return { Elite: '#34d399', High: '#22d3ee', Moderate: '#fbbf24', Low: '#f87171' }[label] || '#94a3b8'
}
