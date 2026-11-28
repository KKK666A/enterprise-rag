import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const appSource = await readFile(
  new URL('../src/App.vue', import.meta.url),
  'utf8',
)

test('a successful upload clears its selection and invalidates stale query results', () => {
  const importDocument = appSource.match(
    /async function importDocument\(\) \{(?<body>[\s\S]*?)\n\}/,
  )?.groups?.body

  assert.ok(importDocument, 'expected to find importDocument')

  const successState = importDocument.match(
    /uploadResult\.value = await post\('\/api\/documents', form\)(?<state>[\s\S]*?)ElMessage\.success/,
  )?.groups?.state

  assert.ok(successState, 'expected state updates after a successful upload')
  assert.match(successState, /uploadRef\.value\?\.clearFiles\(\)/)
  assert.match(successState, /selected\.value = null/)
  assert.match(successState, /bestMatch\.value = null/)
  assert.match(successState, /queried\.value = false/)
})
