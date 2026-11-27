import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
const match = source.match(/const score = value => ([^\r\n]+)/)

assert.ok(match, 'App.vue should define the score formatter')

const score = vm.runInNewContext(`value => ${match[1]}`)

test('retrieval scores are displayed with stable precision', () => {
  assert.equal(score(0.0163934426), '0.0164')
  assert.equal(score(1), '1.0000')
})

test('unavailable retrieval scores use a placeholder', () => {
  for (const value of [null, undefined, '0.5', Number.NaN, Infinity]) {
    assert.equal(score(value), '—')
  }
})
