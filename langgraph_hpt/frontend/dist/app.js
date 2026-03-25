const taskForm = document.getElementById('register-form');
const runForm = document.getElementById('run-form');
const taskIdInput = document.getElementById('task-id');
const taskMeta = document.getElementById('task-meta');
const eventsBox = document.getElementById('events');
const phaseEl = document.getElementById('phase');
const bestValueEl = document.getElementById('best-value');
const iterInfoEl = document.getElementById('iter-info');
const fetchTaskBtn = document.getElementById('fetch-task-btn');

function appendEvent(obj) {
  const now = new Date().toLocaleTimeString();
  eventsBox.textContent += `[${now}] ${JSON.stringify(obj)}\n`;
  eventsBox.scrollTop = eventsBox.scrollHeight;
}

function setError(msg) {
  appendEvent({ phase: 'error', message: msg });
}

async function registerTask(evt) {
  evt.preventDefault();
  const plugin = document.getElementById('plugin-file').files[0];
  const space = document.getElementById('space-file').files[0];
  const taskName = document.getElementById('task-name').value.trim();

  if (!plugin || !space) {
    setError('请先选择插件文件和参数空间文件。');
    return;
  }

  const formData = new FormData();
  formData.append('plugin_file', plugin);
  formData.append('search_space_file', space);
  if (taskName) formData.append('task_name', taskName);

  const resp = await fetch('/hpt/tasks/register', { method: 'POST', body: formData });
  const data = await resp.json();
  if (!resp.ok) {
    setError(data.detail || '任务注册失败');
    return;
  }

  taskIdInput.value = data.task_id;
  taskMeta.textContent = JSON.stringify(data, null, 2);
  appendEvent({ phase: 'task_registered', ...data });
}

async function fetchTaskMeta() {
  const taskId = taskIdInput.value.trim();
  if (!taskId) {
    setError('请先输入 task_id');
    return;
  }
  const resp = await fetch(`/hpt/tasks/${taskId}`);
  const data = await resp.json();
  if (!resp.ok) {
    setError(data.detail || '获取 task 失败');
    return;
  }
  taskMeta.textContent = JSON.stringify(data, null, 2);
}

function parseSSEChunk(block) {
  const lines = block.split('\n');
  let event = 'message';
  let data = '';
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    return { event, payload: JSON.parse(data) };
  } catch {
    return { event, payload: { raw: data } };
  }
}

function updateStats(payload) {
  if (payload.phase) phaseEl.textContent = payload.phase;
  if (payload.best_value !== undefined && payload.best_value !== null) {
    bestValueEl.textContent = Number(payload.best_value).toFixed(6);
  }
  const rep = payload.rep_idx ?? '-';
  const iter = payload.iter_idx ?? '-';
  iterInfoEl.textContent = `${rep}/${iter}`;
}

async function runStream(evt) {
  evt.preventDefault();
  eventsBox.textContent = '';
  bestValueEl.textContent = '-';
  phaseEl.textContent = 'starting';
  iterInfoEl.textContent = '-';

  const taskId = taskIdInput.value.trim();
  if (!taskId) {
    setError('请先注册任务并获得 task_id。');
    return;
  }

  const payload = {
    task_id: taskId,
    method: document.getElementById('method').value,
    T: Number(document.getElementById('T').value),
    T_ini: Number(document.getElementById('T_ini').value),
    T_rep: Number(document.getElementById('T_rep').value),
    seed: Number(document.getElementById('seed').value),
    objective_timeout_s: Number(document.getElementById('timeout').value),
  };

  const resp = await fetch('/hpt/runs/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!resp.ok || !resp.body) {
    const data = await resp.json().catch(() => ({}));
    setError(data.detail || '启动流式运行失败');
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 2);
      if (!block) continue;

      const parsed = parseSSEChunk(block);
      if (!parsed) continue;

      updateStats(parsed.payload);
      appendEvent(parsed.payload);

      if (parsed.event === 'run_completed' && parsed.payload.result) {
        taskMeta.textContent = JSON.stringify(parsed.payload.result, null, 2);
      }
    }
  }
}

taskForm.addEventListener('submit', registerTask);
runForm.addEventListener('submit', runStream);
fetchTaskBtn.addEventListener('click', fetchTaskMeta);
