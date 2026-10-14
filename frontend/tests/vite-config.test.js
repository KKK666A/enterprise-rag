import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFile } from 'node:fs/promises'

test('development server proxies API requests to the local backend', async () => {
  const source = await readFile(new URL('../vite.config.js', import.meta.url), 'utf8')
  const executableSource = source
    .replace(/^import .*$/gm, '')
    .replace('export default defineConfig(', 'globalThis.config = defineConfig(')

  const context = {
    defineConfig: config => config,
    vue: () => ({ name: 'vue' }),
  }
  vm.runInNewContext(executableSource, context)

  assert.equal(context.config.server.host, '127.0.0.1')
  assert.equal(context.config.server.port, 5173)
  assert.equal(context.config.server.strictPort, true)
  assert.deepEqual(
    { ...context.config.server.proxy['/api'] },
    {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      proxyTimeout: 600000,
      timeout: 600000,
    },
  )
})
