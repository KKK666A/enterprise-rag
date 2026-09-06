// 相对路径通过 Vite 代理转发。模型密钥只保存在后端。
export async function post(path, body) {
  let response
  try {
    response = await fetch(path, {
      method: 'POST',
      ...(body instanceof FormData ? { body } : {
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      }),
    })
  } catch { throw new Error('无法连接服务，请检查前后端是否启动。') }
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = data?.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => item.msg).join('；') : `请求失败（${response.status}），请检查后端。`)
  }
  if (!data) throw new Error('服务返回了无效数据。')
  return data
}
