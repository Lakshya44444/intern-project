import { useState, useEffect, useRef } from 'react'
import {
  MapPin, CalendarDays, Scale, IndianRupee,
  Loader2, Sparkles, ArrowRight, Info,
  Camera, CheckCircle2, AlertCircle, Truck, Package,
} from 'lucide-react'

const TODAY = new Date().toISOString().split('T')[0]

const VEHICLE_TYPES = [
  { id: 'LCV',     label: 'LCV',    sub: 'Light',     hint: 'Tata Ace · Bolero',  lam: '0.15', maxW: 3,   defW: 1.5 },
  { id: 'MCV',     label: 'MCV',    sub: 'Medium',    hint: '407 · Eicher',       lam: '0.10', maxW: 12,  defW: 5   },
  { id: 'FTL',     label: 'Heavy',  sub: 'HCV / FTL', hint: '18-wheeler',         lam: '0.062',maxW: 30,  defW: 10  },
  { id: 'CARTING', label: 'Carting',sub: 'Intracity', hint: 'Pickup · 3-Wheeler', lam: '0.15', maxW: 5,   defW: 1   },
]

const DEFAULTS = {
  origin:           'Delhi',
  destination:      'Roorkee',
  trip_date:        TODAY,
  weight_tonnes:    10,
  competitor_price: '',
}

const ADDONS = [
  { id: 'helper_required',  label: 'Helper Required',        cost: '₹1,200', hint: 'Load + unload' },
  { id: 'extra_tarpaulin',  label: 'Extra Tarpaulin (Tirpal)',cost: '₹500',   hint: 'Weather cover' },
  { id: 'extra_rope',       label: 'Extra Rope',             cost: '₹150',   hint: 'Cargo securing' },
  { id: 'owner_escort',     label: "Owner's Escort",         cost: '₹900',   hint: 'Send my person' },
  { id: 'express_delivery', label: 'Express Delivery',       cost: '+30%',   hint: 'Priority dispatch' },
]

export default function InputForm({ onPredict, loading }) {
  const [form, setForm]               = useState(DEFAULTS)
  const [vehicleType, setVehicleType] = useState('FTL')
  const [selectedAddons, setSelectedAddons] = useState([])
  const [dieselPrice, setDieselPrice] = useState(null)
  const [vision, setVision]           = useState({ loading: false, result: null, error: null })
  const [originCoords, setOriginCoords] = useState(null)
  const [destCoords,   setDestCoords]   = useState(null)
  const fileRef                       = useRef(null)

  const toggleAddon = (id) =>
    setSelectedAddons(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  // Clamp weight to vehicle type range when type changes
  useEffect(() => {
    const cfg = VEHICLE_TYPES.find(v => v.id === vehicleType)
    if (cfg && Number(form.weight_tonnes) > cfg.maxW) {
      set('weight_tonnes', cfg.defW)
    }
  }, [vehicleType])

  const handleSubmit = (e) => {
    e.preventDefault()
    onPredict({
      ...form,
      weight_tonnes:    Number(form.weight_tonnes),
      competitor_price: form.competitor_price ? Number(form.competitor_price) : null,
      vehicle_type:     vehicleType,
      diesel_price:     dieselPrice,
      addons:           selectedAddons,
      origin_lat: originCoords?.lat ?? null,
      origin_lon: originCoords?.lon ?? null,
      dest_lat:   destCoords?.lat   ?? null,
      dest_lon:   destCoords?.lon   ?? null,
    })
  }

  // ── Vision upload handler ────────────────────────────────────────────────
  const handleVisionFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setVision({ loading: true, result: null, error: null })

    const reader = new FileReader()
    reader.onload = async (e) => {
      const b64       = e.target.result.split(',')[1]
      const mediaType = file.type || 'image/jpeg'
      try {
        const res  = await fetch('/api/extract-vision', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ image_base64: b64, media_type: mediaType }),
        })
        const data = await res.json()

        if (data.error) {
          setVision({ loading: false, result: null, error: data.error })
          return
        }

        setVision({ loading: false, result: data, error: null })

        // Auto-populate extracted values
        if (data.authentic_diesel)     setDieselPrice(data.authentic_diesel)
        if (data.authentic_competitor) set('competitor_price', String(data.authentic_competitor))
      } catch (err) {
        setVision({ loading: false, result: null, error: 'Network error — check the API server.' })
      }
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    handleVisionFile(file)
  }

  // Mirror backend 3-factor formula (distance unknown at form time — base value shown)
  const w = Number(form.weight_tonnes)
  const { mLoadedBase, mFormula } = (() => {
    if (vehicleType === 'LCV')     return { mLoadedBase: Math.max(12.0, 20.0 - 2.5 * w), mFormula: `20.0 − (2.5 × ${w}T)` }
    if (vehicleType === 'MCV')     return { mLoadedBase: Math.max(6.0,  13.0 - 1.0 * w), mFormula: `13.0 − (1.0 × ${w}T)` }
    if (vehicleType === 'CARTING') return { mLoadedBase: Math.max(4.0,  10.0 - 0.35 * w),mFormula: `10.0 − (0.35 × ${w}T)` }
    return { mLoadedBase: Math.max(2.5, 4.5 - 0.06 * w), mFormula: `4.5 − (0.06 × ${w}T)` }
  })()
  const mLoaded = mLoadedBase.toFixed(2)

  return (
    <form onSubmit={handleSubmit} className="glass p-6 space-y-5 animate-fade-in">

      {/* Header */}
      <div className="flex items-center gap-2 pb-1">
        <Sparkles className="w-4 h-4 text-cyan-400" />
        <h3 className="font-semibold text-white/90 text-sm">Trip Details</h3>
        <span className="ml-auto text-xs text-white/25 font-mono">5 inputs</span>
      </div>

      {/* Vehicle type — 4-class selector */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-white/50">Vehicle Type</label>
        <div className="grid grid-cols-2 gap-2">
          {VEHICLE_TYPES.map(v => {
            const active = vehicleType === v.id
            const Icon = v.id === 'FTL' ? Truck : Package
            return (
              <button
                key={v.id}
                type="button"
                onClick={() => setVehicleType(v.id)}
                className={`flex flex-col gap-0.5 px-3 py-2.5 rounded-xl border text-left
                            transition-all duration-200 ${
                  active
                    ? 'bg-cyan-500/15 border-cyan-500/40'
                    : 'bg-white/[0.03] border-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <Icon className={`w-3.5 h-3.5 ${active ? 'text-cyan-400' : 'text-white/30'}`} />
                  <span className={`text-xs font-semibold ${active ? 'text-cyan-400' : 'text-white/50'}`}>
                    {v.label}
                  </span>
                  <span className={`ml-auto font-mono text-xs ${active ? 'text-cyan-400/70' : 'text-white/20'}`}>
                    λ={v.lam}
                  </span>
                </div>
                <div className={`text-xs pl-5 ${active ? 'text-cyan-300/50' : 'text-white/20'}`}>
                  {v.hint}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Route row */}
      <div className="relative">
        <div className="grid grid-cols-2 gap-3">
          <CityInput
            label="Source"
            value={form.origin}
            onChange={v => set('origin', v)}
            onCoordsChange={setOriginCoords}
            iconColor="text-cyan-400"
          />
          <CityInput
            label="Destination"
            value={form.destination}
            onChange={v => set('destination', v)}
            onCoordsChange={setDestCoords}
            iconColor="text-orange-400"
          />
        </div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 mt-2
                        w-7 h-7 rounded-full bg-slate-900 border border-white/10
                        flex items-center justify-center z-10">
          <ArrowRight className="w-3.5 h-3.5 text-white/30" />
        </div>
      </div>

      {/* Date */}
      <Field icon={<CalendarDays className="w-3.5 h-3.5 text-purple-400" />} label="Trip Date">
        <input
          type="date"
          value={form.trip_date}
          min={TODAY}
          onChange={e => set('trip_date', e.target.value)}
          required
          className="input-field"
        />
      </Field>

      {/* Weight */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
            <Scale className="w-3.5 h-3.5 text-amber-400" />
            Load Weight
          </label>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold font-mono text-amber-400">
              {form.weight_tonnes}T
            </span>
            <span className="text-xs text-white/25 font-mono">
              → {mLoaded} km/L
            </span>
          </div>
        </div>

        {(() => {
          const cfg    = VEHICLE_TYPES.find(v => v.id === vehicleType)
          const maxW   = cfg?.maxW ?? 30
          const minW   = vehicleType === 'LCV' ? 0.5 : vehicleType === 'CARTING' ? 0.5 : 1
          const pct    = ((Number(form.weight_tonnes) - minW) / (maxW - minW)) * 100
          return (
            <>
              <input
                type="range" min={minW} max={maxW} step={0.5}
                value={form.weight_tonnes}
                onChange={e => set('weight_tonnes', e.target.value)}
                style={{ background: `linear-gradient(to right, #fbbf24 ${pct}%, rgba(255,255,255,0.1) ${pct}%)` }}
              />
              <div className="flex justify-between text-xs text-white/20">
                <span>{minW}T</span>
                <span>{maxW}T ({cfg?.sub})</span>
              </div>
            </>
          )
        })()}

        <div className="flex items-center gap-1.5 p-2.5 rounded-lg bg-amber-400/5 border border-amber-400/10">
          <Info className="w-3 h-3 text-amber-400/60 flex-shrink-0" />
          <p className="text-xs text-white/30">
            <span className="font-mono">
              M<sub>loaded</sub> = {mFormula} = {mLoaded} km/L
              <span className="text-white/20"> (+0.5 highway / −0.2 city)</span>
            </span>
          </p>
        </div>
      </div>

      {/* Competitor price */}
      <Field
        icon={<IndianRupee className="w-3.5 h-3.5 text-white/30" />}
        label={<span>Competitor Benchmark <span className="text-white/25 font-normal">(optional)</span></span>}
      >
        <input
          type="number"
          min={0}
          step={100}
          value={form.competitor_price}
          onChange={e => set('competitor_price', e.target.value)}
          className="input-field"
          placeholder="Leave blank or scan a quote photo below"
        />
      </Field>

      {/* ── Add-on Services ───────────────────────────────────────────────── */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-white/50">Add-on Services</label>
        <div className="grid grid-cols-1 gap-1.5">
          {ADDONS.map(a => {
            const active = selectedAddons.includes(a.id)
            return (
              <button
                key={a.id}
                type="button"
                onClick={() => toggleAddon(a.id)}
                className={`flex items-center gap-3 px-3 py-2 rounded-xl border text-left
                            transition-all duration-150 ${
                  active
                    ? 'bg-emerald-500/10 border-emerald-500/30'
                    : 'bg-white/[0.02] border-white/8 hover:border-white/15'
                }`}
              >
                <div className={`w-4 h-4 rounded flex items-center justify-center flex-shrink-0
                                border transition-colors ${
                  active
                    ? 'bg-emerald-500 border-emerald-500'
                    : 'border-white/20 bg-white/[0.04]'
                }`}>
                  {active && (
                    <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 10 10">
                      <path d="M1.5 5l2.5 2.5 4.5-4.5" stroke="currentColor" strokeWidth="1.5"
                            strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <span className={`text-xs font-medium ${active ? 'text-emerald-400' : 'text-white/50'}`}>
                    {a.label}
                  </span>
                  <span className={`text-xs ml-1.5 ${active ? 'text-emerald-400/50' : 'text-white/20'}`}>
                    — {a.hint}
                  </span>
                </div>
                <span className={`text-xs font-mono font-semibold flex-shrink-0 ${
                  active ? 'text-emerald-400' : 'text-white/25'
                }`}>
                  {a.cost}
                </span>
              </button>
            )
          })}
        </div>
        {selectedAddons.length > 0 && (
          <p className="text-xs text-white/25 px-1">
            {selectedAddons.length} add-on{selectedAddons.length > 1 ? 's' : ''} selected
            — costs added after base quote
          </p>
        )}
      </div>

      {/* ── Vision Upload ──────────────────────────────────────────────────── */}
      <div className="space-y-2">
        <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
          <Camera className="w-3.5 h-3.5 text-blue-400" />
          Scan Photo
          <span className="text-white/25 font-normal">(diesel board or competitor quote)</span>
        </label>

        {/* Drop zone */}
        <div
          className={`relative rounded-xl border-2 border-dashed p-4 text-center
                      cursor-pointer transition-all duration-200 ${
            vision.loading
              ? 'border-blue-400/40 bg-blue-400/5'
              : 'border-white/10 hover:border-white/25 hover:bg-white/[0.02]'
          }`}
          onClick={() => !vision.loading && fileRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={e => handleVisionFile(e.target.files[0])}
          />

          {vision.loading ? (
            <div className="flex flex-col items-center gap-2 py-1">
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
              <p className="text-xs text-blue-400/70">Extracting with Claude Vision…</p>
            </div>
          ) : vision.result ? (
            <VisionResult result={vision.result} />
          ) : vision.error ? (
            <VisionError error={vision.error} />
          ) : (
            <div className="flex flex-col items-center gap-1.5 py-1">
              <Camera className="w-5 h-5 text-white/20" />
              <p className="text-xs text-white/30">Click or drag a photo here</p>
              <p className="text-xs text-white/15">Diesel board · WhatsApp quote · Rate screenshot</p>
            </div>
          )}
        </div>

        {/* Diesel override badge */}
        {dieselPrice !== null && (
          <div className="flex items-center justify-between px-3 py-1.5 rounded-lg
                          bg-emerald-400/8 border border-emerald-400/20 text-xs">
            <div className="flex items-center gap-1.5 text-emerald-400">
              <CheckCircle2 className="w-3 h-3" />
              <span className="font-mono">Diesel overridden → ₹{dieselPrice}/L</span>
            </div>
            <button
              type="button"
              onClick={() => { setDieselPrice(null); setVision({ loading: false, result: null, error: null }) }}
              className="text-white/25 hover:text-white/50 text-xs"
            >
              reset
            </button>
          </div>
        )}
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading || !form.origin || !form.destination || !form.trip_date}
        className="w-full py-3.5 rounded-xl font-semibold text-sm
                   bg-gradient-to-r from-cyan-500 to-blue-600
                   hover:from-cyan-400 hover:to-blue-500
                   disabled:opacity-40 disabled:cursor-not-allowed
                   transition-all duration-200 shadow-lg shadow-blue-500/20
                   flex items-center justify-center gap-2"
      >
        {loading
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Generating Quote…</>
          : <><Sparkles className="w-4 h-4" /> Generate AI Quote</>
        }
      </button>

      <style>{`
        .input-field {
          width: 100%;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 0.75rem;
          padding: 0.625rem 0.875rem;
          color: white;
          font-size: 0.875rem;
          outline: none;
          transition: border-color 0.2s;
          color-scheme: dark;
        }
        .input-field:focus { border-color: rgba(34,211,238,0.5); }
        .input-field option { background: #0f172a; }
        input[type='range'] {
          width: 100%;
          height: 6px;
          border-radius: 9999px;
          appearance: none;
          cursor: pointer;
          outline: none;
        }
        input[type='range']::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #fbbf24;
          cursor: pointer;
          box-shadow: 0 0 8px rgba(251,191,36,0.6);
        }
      `}</style>
    </form>
  )
}

/* ── Sub-components ──────────────────────────────────────────────────────────── */

function Field({ icon, label, children }) {
  return (
    <div className="space-y-1.5">
      <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
        {icon}{label}
      </label>
      {children}
    </div>
  )
}

// ── City autocomplete via Photon (Komoot OSM CDN) — proxied through backend ──
async function searchCities(query) {
  try {
    const res = await fetch(`/api/autocomplete?q=${encodeURIComponent(query)}`)
    if (!res.ok) return []
    return await res.json()
  } catch {
    return []
  }
}

function CityInput({ label, value, onChange, onCoordsChange, iconColor }) {
  const [suggestions, setSuggestions] = useState([])
  const [open, setOpen]               = useState(false)
  const [searching, setSearching]     = useState(false)
  const debounceRef                   = useRef(null)

  const onInput = (v) => {
    onChange(v)
    onCoordsChange(null)
    clearTimeout(debounceRef.current)

    if (v.length < 2) {
      setSuggestions([])
      setOpen(false)
      setSearching(false)
      return
    }

    setSearching(true)
    debounceRef.current = setTimeout(async () => {
      const results = await searchCities(v)
      setSuggestions(results)
      setOpen(results.length > 0)
      setSearching(false)
    }, 220)
  }

  const select = (s) => {
    onChange(s.city)
    onCoordsChange(s.lat != null ? { lat: s.lat, lon: s.lon } : null)
    setSuggestions([])
    setOpen(false)
  }

  return (
    <div className="space-y-1.5 relative">
      <label className="flex items-center gap-1.5 text-xs font-medium text-white/50">
        <MapPin className={`w-3.5 h-3.5 ${iconColor}`} />
        {label}
        {searching && <Loader2 className="w-3 h-3 animate-spin ml-auto opacity-40" />}
      </label>

      <input
        type="text"
        value={value}
        onChange={e => onInput(e.target.value)}
        onFocus={() => suggestions.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        required
        className="input-field"
        placeholder={`Type ${label === 'Source' ? 'origin' : 'destination'} city…`}
        autoComplete="off"
      />

      {open && suggestions.length > 0 && (
        <div
          className="absolute z-50 w-full mt-0.5 rounded-xl border border-white/15
                     overflow-hidden shadow-2xl"
          style={{ background: 'rgba(10,18,36,0.97)', backdropFilter: 'blur(16px)' }}
        >
          {suggestions.map(s => (
            <button
              key={s.id}
              type="button"
              onMouseDown={() => select(s)}
              className="w-full text-left px-3 py-2.5 flex items-center gap-2.5
                         hover:bg-white/[0.06] transition-colors
                         border-b border-white/[0.05] last:border-0"
            >
              <MapPin className={`w-3 h-3 ${iconColor} flex-shrink-0 opacity-50`} />
              <div className="min-w-0">
                <div className="text-sm text-white/85">{s.city}</div>
                {s.state && <div className="text-xs text-white/30 truncate">{s.state}</div>}
              </div>
            </button>
          ))}
          <div className="px-3 py-1.5 text-right">
            <span className="text-xs text-white/15">© OpenStreetMap</span>
          </div>
        </div>
      )}
    </div>
  )
}

function VisionResult({ result }) {
  const hasValues = result.authentic_diesel || result.authentic_competitor
  const isHigh    = result.confidence_level === 'High'
  const srcLabel  = { fuel_board: 'Fuel Board', chat_screenshot: 'Chat Screenshot', rate_chart: 'Rate Chart' }[result.source_type] || result.source_type || 'Image'
  return (
    <div className="space-y-1.5 text-left">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs text-emerald-400">
          <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="font-medium">
            {isHigh ? 'Extraction successful' : 'Extracted (low confidence)'}
          </span>
        </div>
        {result.source_type && (
          <span className="text-xs font-mono text-white/25">{srcLabel}</span>
        )}
      </div>
      {result.authentic_diesel && (
        <div className="flex justify-between text-xs text-white/40 px-1">
          <span>Diesel (cash rate)</span>
          <span className="font-mono text-emerald-400">₹{result.authentic_diesel}/L</span>
        </div>
      )}
      {result.authentic_competitor && (
        <div className="flex justify-between text-xs text-white/40 px-1">
          <span>Competitor rate</span>
          <span className="font-mono text-emerald-400">₹{Number(result.authentic_competitor).toLocaleString('en-IN')}</span>
        </div>
      )}
      {result.notes && (
        <p className="text-xs text-white/20 italic px-1">{result.notes}</p>
      )}
      {!hasValues && (
        <p className="text-xs text-amber-400/70 px-1">No pricing data found in image.</p>
      )}
    </div>
  )
}

function VisionError({ error }) {
  const isNoKey = error.includes('ANTHROPIC_API_KEY')
  return (
    <div className="flex flex-col items-center gap-1.5 py-1 text-center">
      <AlertCircle className="w-5 h-5 text-red-400/60" />
      <p className="text-xs text-red-400/70 max-w-xs">
        {isNoKey
          ? 'Vision disabled — set ANTHROPIC_API_KEY in the backend environment.'
          : error
        }
      </p>
    </div>
  )
}
