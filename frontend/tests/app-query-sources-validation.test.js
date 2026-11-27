import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const component = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
const script = component.match(/<script setup>([\s\S]*?)<\/script>/)?.[1]

function loadSearch(post) {
  const executable = script
    .replace(/^import .*$/gm, '')
    .concat('\nreturn { search, queryError, querying, question, queried, bestMatch }')

  const ref = value => ({ value })
  const computed = getter => ({ get value() { return getter() } })
  const ElMessage = { success() {}, warning() {} }

  return Function('ref', 'computed', 'ElMessage', 'post', executable)(
    ref,
    computed,
    ElMessage,
    post,
  )
}

test('search rejects a successful response whose sources field is not an array', async () => {
  const state = loadSearch(async () => ({ sources: null }))
  state.question.value = 'synthetic policy question'

  await state.search()

  assert.match(state.queryError.value, /sources/)
  assert.equal(state.querying.value, false)
  assert.equal(state.queried.value, false)
  assert.equal(state.bestMatch.value, null)
})
