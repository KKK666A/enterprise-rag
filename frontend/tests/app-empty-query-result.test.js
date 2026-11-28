import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const appSource = await readFile(
  new URL('../src/App.vue', import.meta.url),
  'utf8',
)

test('an empty query result uses the API message and renders the empty state', () => {
  assert.match(
    appSource,
    /bestMatch\.value\s*=\s*data\.sources\[0\]\s*\?\?\s*null/,
    'an empty sources array should clear the previous best match',
  )
  assert.match(
    appSource,
    /emptyMessage\.value\s*=\s*data\.message\s*\|\|\s*['"][^'"]+['"]/,
    'the API message should be retained with a user-facing fallback',
  )
  assert.match(
    appSource,
    /<el-empty\s+v-else-if="queried"\s+:description="emptyMessage"/,
    'completed queries without a best match should render the empty state',
  )
})
