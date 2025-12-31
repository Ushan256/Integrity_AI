import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://ntxcydlhdrtmzacclivk.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im50eGN5ZGxoZHJ0bXphY2NsaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjcyMDYyMTEsImV4cCI6MjA4Mjc4MjIxMX0.Iu58jj8ntkfMj_x6AWxvLu_WIZvAzJLSHvhm0Ww9_r0'

export const supabase = createClient(supabaseUrl, supabaseKey)