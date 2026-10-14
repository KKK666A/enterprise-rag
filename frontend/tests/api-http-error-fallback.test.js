import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post reports the HTTP status when an error response has no JSON detail', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => new Response('<html>Bad Gateway</html>', {
    status: 502,
    headers: { 'Content-Type': 'text/html' },
  })

  await assert.rejects(
    post('/api/query', { question: 'synthetic policy question' }),
    error => {
      assert.match(error.message, /502/)
      return true
    },
  )
})
