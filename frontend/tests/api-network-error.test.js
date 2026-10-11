import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post reports a stable connection error when fetch rejects', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => {
    throw new TypeError('synthetic network failure with internal details')
  }

  await assert.rejects(
    post('/api/query', { question: 'synthetic policy question' }),
    {
      message: '无法连接服务，请检查前后端是否启动。',
    },
  )
})
