import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

function loadUploadSelector() {
  const component = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
  const setup = component.match(/<script setup>([\s\S]*?)<\/script>/)?.[1]

  assert.ok(setup, 'App.vue must contain a script setup block')

  const executable = setup
    .replace(/^import .*$/gm, '')
    .concat('\nglobalThis.uploadSelector = { choose, selected, uploadError, uploadResult, uploadRef }')

  const context = {
    computed: callback => ({ get value() { return callback() } }),
    ref: value => ({ value }),
  }
  vm.runInNewContext(executable, context)
  return context.uploadSelector
}

test('upload selector rejects a non-TXT file and clears stale state', () => {
  const selector = loadUploadSelector()
  const previousFile = { name: 'synthetic-policy.txt', size: 128 }
  let clearCalls = 0

  selector.selected.value = previousFile
  selector.uploadResult.value = { filename: previousFile.name }
  selector.uploadRef.value = { clearFiles: () => { clearCalls += 1 } }

  selector.choose({ raw: { name: 'synthetic-policy.pdf', size: 128 } })

  assert.equal(selector.selected.value, null)
  assert.equal(selector.uploadResult.value, null)
  assert.match(selector.uploadError.value, /TXT/)
  assert.equal(clearCalls, 1)
})
