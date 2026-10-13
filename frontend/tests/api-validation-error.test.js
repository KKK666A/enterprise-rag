import assert from 'node:assert/strict'
import test from 'node:test'

import { post } from '../src/api.js'

test('post combines FastAPI validation messages into a readable error', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({
      detail: [
        { loc: ['body', 'question'], msg: 'Field required' },
        { loc: ['body', 'question'], msg: 'Must be at most 200 characters' },
      ],
    }),
  })

  await assert.rejects(
    post('/api/query', { question: '' }),
    { message: 'Field required；Must be at most 200 characters' },
  )
})
