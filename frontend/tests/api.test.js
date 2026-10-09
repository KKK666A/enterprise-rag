import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post sends JSON requests and returns the parsed response', async (t) => {
  const originalFetch = globalThis.fetch
  const requests = []

  globalThis.fetch = async (path, options) => {
    requests.push({ path, options })
    return new Response(
      JSON.stringify({ question: 'synthetic question', sources: [] }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )
  }

  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const result = await post('/api/query', {
    question: 'synthetic question',
  })

  assert.deepEqual(result, {
    question: 'synthetic question',
    sources: [],
  })
  assert.equal(requests.length, 1)
  assert.equal(requests[0].path, '/api/query')
  assert.equal(requests[0].options.method, 'POST')
  assert.deepEqual(requests[0].options.headers, {
    'Content-Type': 'application/json',
  })
  assert.equal(
    requests[0].options.body,
    JSON.stringify({ question: 'synthetic question' }),
  )
})
