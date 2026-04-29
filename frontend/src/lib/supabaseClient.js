import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL  = import.meta.env.VITE_SUPABASE_URL
const SUPABASE_ANON = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!SUPABASE_URL || !SUPABASE_ANON) {
  console.warn(
    '[FreightIQ] Supabase credentials missing.\n' +
    'Create frontend/.env with VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.\n' +
    'See frontend/.env.example for instructions.'
  )
}

export const supabase = createClient(SUPABASE_URL ?? '', SUPABASE_ANON ?? '')
