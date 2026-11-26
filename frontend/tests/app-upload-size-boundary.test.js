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

test('upload selector accepts a non-empty TXT file at the 200 KiB limit', () => {
  const selector = loadUploadSelector()
  const file = { name: 'synthetic-policy.txt', size: 200 * 1024 }
  let clearCalls = 0

  selector.uploadRef.value = { clearFiles: () => { clearCalls += 1 } }
  selector.uploadError.value = 'stale error'
  selector.uploadResult.value = { filename: 'previous.txt' }

  selector.choose({ raw: file })

  assert.equal(selector.selected.value, file)
  assert.equal(selector.uploadError.value, '')
  assert.equal(selector.uploadResult.value, null)
  assert.equal(clearCalls, 0)
})
