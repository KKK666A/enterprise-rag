import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const appSource = readFileSync(
  new URL('../src/App.vue', import.meta.url),
  'utf8',
)

test('retrieved document sources are rendered as escaped text', () => {
  assert.match(
    appSource,
    /<span class="muted small">\s*{{\s*bestMatch\.source\s*}}\s*<\/span>/,
  )
  assert.doesNotMatch(
    appSource,
    /v-html\s*=\s*["']bestMatch\.source["']/,
  )
})
