import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const appSource = readFileSync(
  new URL('../src/App.vue', import.meta.url),
  'utf8',
)

test('retrieved document content is rendered as text, not raw HTML', () => {
  assert.match(appSource, /\{\{\s*bestMatch\.text\s*\}\}/)
  assert.doesNotMatch(appSource, /v-html\s*=\s*["'][^"']*bestMatch\.text/)
})
