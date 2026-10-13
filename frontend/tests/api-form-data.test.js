import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post passes FormData through without overriding multipart headers', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  const form = new FormData()
  form.append('file', new Blob(['synthetic document']), 'synthetic.txt')

  let request
  globalThis.fetch = async (path, options) => {
    request = { path, options }
    return new Response(JSON.stringify({ filename: 'synthetic.txt', chunks: 1 }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const result = await post('/api/documents', form)

  assert.deepEqual(result, { filename: 'synthetic.txt', chunks: 1 })
  assert.equal(request.path, '/api/documents')
  assert.equal(request.options.method, 'POST')
  assert.equal(request.options.body, form)
  assert.equal('headers' in request.options, false)
})
