import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post reports a successful response with invalid JSON', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    json: async () => { throw new SyntaxError('synthetic malformed JSON') },
  })

  await assert.rejects(
    post('/api/query', { question: 'synthetic question' }),
    { message: '服务返回了无效数据。' },
  )
})
