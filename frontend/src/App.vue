<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { post } from './api'

const uploadRef = ref()
const selected = ref(null)
const uploading = ref(false)
const querying = ref(false)
const busy = computed(() => uploading.value || querying.value)
const uploadResult = ref(null)
const uploadError = ref('')
const question = ref('')
const submittedQuestion = ref('')
const queryError = ref('')
const bestMatch = ref(null)
const queried = ref(false)
const emptyMessage = ref('')
const examples = ['年假需要提前多久申请？', '差旅报销需要哪些材料？', '远程办公如何申请？']

function choose(file) {
  uploadError.value = ''
  uploadResult.value = null
  const raw = file.raw
  if (!raw || !raw.name.toLowerCase().endsWith('.txt') || !raw.size || raw.size > 200 * 1024) {
    uploadError.value = '请选择非空 TXT 文件，大小不超过 200KB。'
    selected.value = null
    uploadRef.value?.clearFiles()
    return
  }
  selected.value = raw
}

async function importDocument() {
  if (!selected.value || busy.value) return
  uploading.value = true
  uploadError.value = ''
  uploadResult.value = null
  try {
    const form = new FormData()
    form.append('file', selected.value)
    uploadResult.value = await post('/api/documents', form)
    uploadRef.value?.clearFiles()
    selected.value = null
    bestMatch.value = null
    queried.value = false
    ElMessage.success('文档导入成功')
  } catch (error) { uploadError.value = error.message }
  finally { uploading.value = false }
}

async function search() {
  const text = question.value.trim()
  if (!text || busy.value) return
  querying.value = true
  queryError.value = ''
  bestMatch.value = null
  queried.value = false
  submittedQuestion.value = text
  try {
    const data = await post('/api/query', { question: text })
    if (!Array.isArray(data.sources)) throw new Error('服务返回的 sources 格式不正确。')
    // 后端已按 RRF 排序，只展示第一条，不把分数当作百分比。
    bestMatch.value = data.sources[0] ?? null
    emptyMessage.value = data.message || '没有找到匹配文档。'
    queried.value = true
  } catch (error) { queryError.value = error.message }
  finally { querying.value = false }
}
const score = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—'
</script>

<template>
  <header><div class="brand"><span class="logo">知</span><strong>知页</strong><span class="eyebrow">KNOWLEDGE SPACE</span></div><span class="muted small">企业知识库 · 学习版</span></header>
  <main>
    <section class="hero"><span class="eyebrow">YOUR KNOWLEDGE, WITHIN REACH</span><h1>让知识，随问而得。</h1><p class="muted">导入文档，用一个问题找到最相关的原文。</p></section>
    <div class="workspace">
      <aside class="panel">
        <span class="eyebrow">01 / 知识导入</span><h2>从一份文档开始</h2><p class="muted small">文本切分、向量化与入库由后端完成。</p>
        <el-upload ref="uploadRef" drag :auto-upload="false" accept=".txt" :limit="1" :disabled="busy" :on-change="choose" :on-remove="() => selected = null" :on-exceed="() => ElMessage.warning('请先移除已选文件')">
          <div class="upload-icon">↑</div><div>拖拽文件到这里，或 <em>点击选择</em></div><template #tip><div class="muted small">UTF-8 编码 TXT · 最大 200KB</div></template>
        </el-upload>
        <el-button class="wide" type="primary" :loading="uploading" :disabled="!selected || querying" @click="importDocument">{{ uploading ? '正在向量化并入库…' : '导入知识库' }}</el-button>
        <el-alert v-if="uploadError" :title="uploadError" type="error" :closable="false" show-icon />
        <div v-if="uploadResult" class="success"><strong>已导入 {{ uploadResult.filename }}</strong><p>新增 {{ uploadResult.chunks }} 个文本块</p><span>知识库共 {{ uploadResult.total_chunks }} 个文本块</span></div>
        <p class="note">同名文件暂不支持覆盖。文档将发送至外部向量模型，请勿上传未获授权的敏感资料。</p>
      </aside>
      <section class="panel">
        <span class="eyebrow">02 / 混合检索</span><h2>你想了解什么？</h2><label for="question">输入你的问题</label>
        <el-input id="question" v-model="question" type="textarea" :rows="3" maxlength="200" show-word-limit :disabled="busy" placeholder="例如：年假需要提前多久申请？" />
        <div class="query-actions"><span class="muted small">BM25 关键词 + 向量语义检索</span><el-button type="primary" :loading="querying" :disabled="!question.trim() || uploading" @click="search">检索知识库 →</el-button></div>
        <div class="examples"><button v-for="example in examples" :key="example" :disabled="busy" @click="question = example">{{ example }}</button></div>
        <el-alert v-if="queryError" :title="queryError" type="error" :closable="false" show-icon />
        <div class="result" aria-live="polite" :aria-busy="querying">
          <el-skeleton v-if="querying" :rows="4" animated />
          <article v-else-if="bestMatch"><div class="result-heading"><el-tag type="success">TOP 1 · 最相关原文</el-tag><span class="muted small">{{ bestMatch.source }}</span></div><p class="muted small">{{ submittedQuestion }}</p><div class="result-text">{{ bestMatch.text }}</div><p class="small">RRF 融合分数 {{ score(bestMatch.rrf_score) }} · 向量分数 {{ score(bestMatch.vector_score) }}</p><p class="muted small">分数用于排序，不代表答案正确率；本页展示原文，不生成回答。</p></article>
          <el-empty v-else-if="queried" :description="emptyMessage" :image-size="70" />
          <div v-else class="placeholder"><span>⌕</span><h3>答案的线索，就在文档里</h3><p>提出一个问题，这里只展示排名最高的文本块。</p></div>
        </div>
      </section>
    </div>
    <footer>Vue 3 + Element Plus <span>Milvus · BM25 · RRF</span></footer>
  </main>
</template>
