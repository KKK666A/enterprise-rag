import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post surfaces a string error detail returned by the backend', async t => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: 'Synthetic document was rejected' }),
    {
      status: 409,
      headers: { 'Content-Type': 'application/json' },
    },
  )

  await assert.rejects(
    post('/api/documents', { filename: 'synthetic.txt' }),
    { message: 'Synthetic document was rejected' },
  )
})
