// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// The Showrunner JSON API and its event stream. Paths are relative: the page works wherever WanGP mounts it.

async function call(method, path, body) {
  const options = { method, headers: {} };
  if (body instanceof FormData) options.body = body;
  else if (body !== undefined) { options.body = JSON.stringify(body); options.headers['Content-Type'] = 'application/json'; }
  const response = await fetch(`api/${path}`, options);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!response.ok) {
    const error = new Error(data?.detail || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.current = data?.current;
    throw error;
  }
  return data;
}

export const api = {
  state: () => call('GET', 'state'),
  create: (title, kind, brief = {}) => call('POST', 'productions', { title, kind, brief }),
  get: id => call('GET', `productions/${id}`),
  ops: (id, ops, base_revision = null) => call('POST', `productions/${id}/ops`, { ops, base_revision }),
  undo: id => call('POST', `productions/${id}/undo`),
  redo: id => call('POST', `productions/${id}/redo`),
  duplicate: id => call('POST', `productions/${id}/duplicate`),
  version: (id, version, against = 'current') => call('GET', `productions/${id}/versions/${version}?against=${encodeURIComponent(against)}`),
  bringBack: (id, version, kind, item) => call('POST', `productions/${id}/versions/${version}/bring`, { kind, item: String(item) }),
  restoreVersion: (id, version) => call('POST', `productions/${id}/versions/${version}/restore`),
  duplicateVersion: (id, version) => call('POST', `productions/${id}/versions/${version}/duplicate`),
  trash: id => call('DELETE', `productions/${id}`),
  archive: (id, archived = true) => call('POST', `productions/${id}/archive`, { archived }),
  analysis: (id, refresh = false) => call('GET', `productions/${id}/analysis${refresh ? '?refresh=true' : ''}`),
  renderGroups: (id, shots) => call('GET', `productions/${id}/render-groups?shots=${encodeURIComponent(shots.join(','))}`),
  waveform: (id, path) => call('GET', `productions/${id}/waveform?path=${encodeURIComponent(path)}`),
  subtitles: (id, path) => call('GET', `productions/${id}/subtitles?path=${encodeURIComponent(path)}`),
  rates: id => call('GET', `productions/${id}/rates`),
  sizes: id => call('GET', `productions/${id}/sizes`),
  director: id => call('GET', `productions/${id}/director`),
  direct: (id, task, message = '', shots = [], newVersion = true) => call('POST', `productions/${id}/director`, { task, message, shots, new_version: newVersion }),
  steer: (id, text) => call('POST', `productions/${id}/director/steer`, { text }),
  stop: id => call('POST', `productions/${id}/director/stop`),
  pause: (id, now = false) => call('POST', `productions/${id}/director/pause${now ? '?now=true' : ''}`),
  resume: (id, message = '') => call('POST', `productions/${id}/director/resume`, { message }),
  rerun: (id, shot) => call('POST', `productions/${id}/shots/${shot}/rerun`),
  renderNow: (id, shots, final = false, newSeed = null) => call('POST', `productions/${id}/render`, { shots, final, new_seed: newSeed }),
  songVoice: (id, singers) => call('POST', `productions/${id}/song-voice`, { singers }),
  singerVoice: () => call('GET', 'singer-voice'),
  detachSound: (id, shot) => call('POST', `productions/${id}/shots/${shot}/detach-sound`),
  cancelRender: (id, shot) => call('POST', `productions/${id}/shots/${shot}/cancel`),
  assemble: id => call('POST', `productions/${id}/assemble`),
  templates: id => call('GET', `productions/${id}/templates`),
  chooseTemplate: (id, role, template) => call('POST', `productions/${id}/templates`, { role, template }),
  defaultTemplates: () => call('GET', 'templates'),
  chooseDefaultTemplate: (role, template) => call('POST', 'templates', { role, template }),
  setPromptRules: (template, text) => call('POST', 'prompt-rules', { template, text }),
  guide: () => call('GET', 'guide'),
  settings: () => call('GET', 'settings'),
  conversation: (id, after) => call('GET', `productions/${id}/conversation?after=${after}`),
  saveSettings: settings => call('POST', 'settings', settings),
  castLoras: id => call('GET', `productions/${id}/loras`),
  addText: (id, text) => call('POST', `productions/${id}/text`, text),
  changeText: (id, overlay, fields) => call('POST', `productions/${id}/text/${overlay}`, { fields }),
  card: (id, shot, text) => call('POST', `productions/${id}/shots/${shot}/card`, text),
  upload(id, file, role = 'bin', target = null) {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('role', role);
    if (target != null && target !== '') form.append('target', target);  // a position can be 0
    return call('POST', `productions/${id}/upload`, form);
  },
  exportUrl: id => `api/productions/${id}/export`,
  importZip(file) {
    const form = new FormData();
    form.append('file', file, file.name);
    return call('POST', 'productions/import', form);
  },
  sourcePart: (id, shot) => `api/productions/${id}/source_part?shot=${encodeURIComponent(shot)}`,
  sourceFrame: (id, shot, frame) => `api/productions/${id}/source_frame?shot=${encodeURIComponent(shot)}&frame=${encodeURIComponent(frame)}`,
  takePart: (id, shot, take) => `api/productions/${id}/take_part?shot=${encodeURIComponent(shot)}&take=${encodeURIComponent(take)}`,
  file: (id, path, download = false) => path ? `api/productions/${id}/file?path=${encodeURIComponent(path)}${download ? '&download=true' : ''}` : '',
};

// One event stream for the open production (or the catalog), on a WebSocket: an HTTP stream would hold one of the few connections the
// browser shares between all the tabs of a server. After a drop it reconnects, then calls handlers.reconnected to catch up.
export function listen(production, handlers) {
  const url = new URL(`api/events${production ? `?production=${encodeURIComponent(production)}` : ''}`, location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  let socket = null, timer = null, dropped = false;
  const connect = () => {
    socket = new WebSocket(url);
    socket.onopen = () => { if (dropped) { dropped = false; handlers.reconnected?.(); } };
    socket.onmessage = message => {
      const event = JSON.parse(message.data);
      try { handlers[event.type || 'message']?.(event); } catch (error) { console.error(error); }
    };
    socket.onclose = () => { dropped = true; timer = setTimeout(connect, 2000); };
  };
  connect();
  return () => { clearTimeout(timer); socket.onclose = null; socket.close(); };
}
