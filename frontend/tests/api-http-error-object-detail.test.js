import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'


test('falls back to the HTTP status for object-shaped error details', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async () => ({
    ok: false,
    status: 418,
    json: async () => ({
      detail: { code: 'synthetic_error', retryable: false },
    }),
  })

  try {
    await assert.rejects(
      post('/api/query', { question: 'synthetic question' }),
      error => {
        assert.match(error.message, /418/)
        assert.doesNotMatch(error.message, /\[object Object\]/)
        assert.doesNotMatch(error.message, /synthetic_error/)
        return true
      },
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})
