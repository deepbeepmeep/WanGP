// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Storyboard: every shot as a card, grouped by section; drag to reorder, drop media to add takes, review at a glance.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, art, formatDuration, sectionColor, menu, confirmDialog, promptDialog, toast, count, shootEstimate } from './ui.js';
import { segments } from './sequencer.js';
import { templateLabel } from './models.js';

// A shot's length in the cut, as the timeline and the export play it: its planned length or trim, not the whole take.
const cutLengths = doc => new Map(segments(doc, actions.take).map(item => [item.shot.id, item.duration]));
// What Shoot shoots: the included shots without a take and the ones marked for a redo.
export const toShoot = doc => doc.shots.filter(shot => shot.included && !shot.locked && (!shot.takes.length || ['rejected', 'failed'].includes(shot.status)));

// Roughly how long shooting these shots takes, from the measured times of their models, fetched again when the takes or the format change
// (ready: called when new times arrive).
const rates = { key: '', value: null, ready: new Set() };
export function shootTime(doc, shots, ready) {
  const key = [doc.id, doc.brief.aspect, doc.brief.resolution, doc.shots.reduce((sum, shot) => sum + shot.takes.length, 0)].join('|');
  if (rates.key !== key) {
    if (rates.key.split('|')[0] !== doc.id) rates.value = null;  // never another production's times
    Object.assign(rates, { key, ready: new Set() });
    api.rates(doc.id).then(result => { if (rates.key === key) { rates.value = result.rates; rates.ready.forEach(call => call()); } }).catch(() => {});
  }
  rates.ready.add(ready);
  return rates.value ? shootEstimate(shots, rates.value) : '';
}

const FILTERS = [['all', 'All'], ['todo', 'To shoot'], ['review', 'To review'], ['approved', 'Approved'], ['excluded', 'Excluded']];
const STATUS_LABEL = { planned: 'Not shot', queued: 'Queued', rendering: 'Rendering', rendered: 'To review', approved: 'Approved', rejected: 'Redo', failed: 'Failed' };
export const BIN_TYPE = 'application/x-showrunner-bin';
export const SHOT_TYPE = 'application/x-showrunner-shots';

export function shotPoster(shot) {
  const take = actions.take(shot);
  return take?.poster ? actions.file(take.poster) : (take?.kind === 'image' ? actions.file(take.path) : null);
}

export function castAvatars(ids, doc) {
  return `<span class="avatars">${ids.map(id => doc.cast.find(member => member.id === id)).filter(Boolean).slice(0, 4).map(member => {
    const image = member.master || member.images[0];
    return `<span class="avatar-xs" title="${esc(member.name)}" style="${image ? `background-image:url('${actions.file(image)}')` : ''}">${image ? '' : esc((member.name || '?')[0])}</span>`;
  }).join('')}</span>`;
}

function speakerName(speaker, doc) {
  return doc.cast.find(member => member.id === speaker)?.name || (speaker === 'NARRATOR' ? 'Narrator' : speaker || '?');
}

export function mountStoryboard(container, doc) {
  container.innerHTML = `<div class="toolbar sb-toolbar" id="sb-tools"></div><div class="storyboard" id="board"></div>`;
  const tools = container.querySelector('#sb-tools');
  const board = container.querySelector('#board');
  const cards = new Map();
  let dragIds = null, dropTarget = null;

  function visible(shot) {
    switch (state.filter) {
      case 'todo': return shot.included && ['planned', 'failed', 'rejected', 'queued', 'rendering'].includes(shot.status);
      case 'review': return shot.status === 'rendered';
      case 'approved': return shot.status === 'approved';
      case 'excluded': return !shot.included;
      default: return true;
    }
  }

  function renderTools() {
    const doc = state.doc;
    const total = Math.max(0, ...segments(doc, actions.take).map(item => item.start + item.duration));  // where the cut ends: transitions overlap
    const counts = status => doc.shots.filter(shot => shot.status === status).length;
    const working = actions.busy(doc) || state.live.running;
    const selected = [...state.selection];
    const shooting = toShoot(doc), todo = shooting.length;
    const estimate = todo ? shootTime(doc, shooting, renderTools) : '';
    const review = doc.brief.review_before_edit, approved = shot => actions.take(shot)?.verdict?.by === 'user' && actions.take(shot).verdict.ok;
    const waiting = review && !todo ? doc.shots.filter(shot => shot.included && !approved(shot)).length : 0;  // the showrunner waits for them before the edit
    tools.innerHTML = `<div class="col" style="gap:0;min-width:0"><h2>Storyboard</h2><span class="stat" style="white-space:nowrap">${doc.shots.length} shots · ${formatDuration(total)} · <span style="color:var(--green)">${counts('approved')} approved</span> · <span style="color:var(--amber)">${counts('rendered')} to review</span></span></div>
      <span class="sep"></span><div class="filters">${FILTERS.map(([key, label]) => `<button class="chip clickable ${state.filter === key ? 'on' : ''}" data-filter="${key}">${label}</button>`).join('')}</div>
      <div class="row" style="margin-left:auto;flex-wrap:nowrap">
      <input type="range" class="range" id="density" min="180" max="420" step="10" value="${state.cardWidth}" style="width:96px" title="Card size">
      ${selected.length > 1 ? `<span class="chip on">${selected.length} selected</span>` : ''}
      ${selected.length ? `<button class="btn small" id="reshoot" ${working ? 'disabled' : ''}>${icon('refresh')} Re-shoot ${count(selected.length, 'shot')}</button>` : ''}
      <button class="btn small" id="add">${icon('plus')} Shot</button>
      ${estimate ? `<span class="stat" style="white-space:nowrap" title="Roughly how long generating the ${count(todo, 'shot')} to shoot takes, measured on earlier takes of the same models (this production's first, then your other productions' at the same picture size). Checks and retakes add to it.">${icon('clock')} ${esc(estimate)}</span>` : ''}
      ${waiting ? `<button class="chip clickable" id="review-wait" title="The showrunner waits for your review before the edit: approve each take, or mark it for a redo. Assemble still makes a cut now.">${icon('clock')}Waiting for your review: ${count(waiting, 'shot')}</button>` : ''}
      ${doc.shots.length ? `<button class="btn primary small" id="shoot" ${working ? 'disabled' : ''} title="${todo ? `The showrunner shoots the shots without a take and the ones marked for a redo, then ${review ? 'waits for your review before the edit' : 'assembles the edit'}` : 'Every shot has a take: assemble the edit now (texts, sound, subtitles); the showrunner panel can also assemble and check it'}">${icon(todo ? 'clapper' : 'film')} ${todo ? `Shoot ${count(todo, 'shot')}` : 'Assemble'}</button>`
        : `<button class="btn primary small" id="plan" ${working ? 'disabled' : ''}>${icon('wand')} Plan with the showrunner</button>`}</div>`;
    tools.querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
      state.filter = button.dataset.filter;
      const shown = new Set(visibleIds());
      state.selection = new Set([...state.selection].filter(id => shown.has(id)));  // a selection never hides behind the filter
      renderTools(); renderBoard(); actions.select(state.focus && shown.has(state.focus) ? state.focus : [...state.selection][0] || null, { additive: false, open: false, keep: true });
    });
    tools.querySelector('#density').oninput = event => { state.cardWidth = +event.target.value; localStorage.setItem('sr.card', state.cardWidth); board.style.setProperty('--card-w', `${state.cardWidth}px`); };
    tools.querySelector('#add').onclick = () => addShot();
    tools.querySelector('#plan')?.addEventListener('click', () => actions.direct('plan'));
    tools.querySelector('#shoot')?.addEventListener('click', () => todo ? actions.direct('shoot') : actions.assemble());  // as the timeline's Assemble: at once, by you
    tools.querySelector('#reshoot')?.addEventListener('click', () => actions.direct('reshoot', '', selected));
    tools.querySelector('#review-wait')?.addEventListener('click', () => tools.querySelector('[data-filter="review"]').click());
  }

  async function addShot(position = null) {
    const doc = state.doc;
    const last = doc.shots.at(-1);
    const result = await actions.ops([{ op: 'add_shots', position, shots: [{ title: 'New shot', section: last?.section || '', duration_s: 4, generation: { template: last?.generation?.template || 'gen_video', settings: { prompt: '' } } }] }]);
    const id = result?.[0]?.shots?.[0];
    if (id) actions.select(id);
  }

  function signature(shot, index) {
    const take = actions.take(shot);
    return JSON.stringify([index, shot.title, shot.intent, shot.dialogue, shot.section, shot.status, shot.locked, shot.included, shot.duration_s, shot.trim, shot.camera, shot.generation?.template, shot.cast, take?.id, take?.poster, take?.verdict, shot.takes.length, state.selection.has(shot.id), state.renders[shot.id], state.flash?.has(shot.id)]);
  }

  function card(shot, index) {
    const doc = state.doc;
    const take = actions.take(shot);
    const poster = shotPoster(shot);
    const verdict = take?.verdict;
    const rendering = shot.status === 'rendering' || state.renders[shot.id] != null;
    const element = document.createElement('article');
    element.className = `shot ${state.selection.has(shot.id) ? 'selected' : ''} ${shot.included ? '' : 'excluded'} ${rendering ? 'shot-rendering' : ''} ${state.flash?.has(shot.id) ? 'flash' : ''}`;
    element.dataset.id = shot.id;
    element.draggable = true;
    element.innerHTML = `<div class="shot-media">
        ${poster ? `<img src="${poster}" alt="" loading="lazy" draggable="false">` : `<div class="ph-art" style="${art(shot.id + shot.section, shot.intent)}"></div><div class="ph"><span>${esc(shot.intent || shot.generation?.settings?.prompt || 'No description yet')}</span></div>`}
        <span class="shot-num">${index + 1}</span>
        <span class="shot-status">${shot.locked ? `<span class="tag" title="${shot.takes.length ? 'Locked: the showrunner leaves it as it is' : 'Locked before it was shot: it will not be shot nor change'}">${icon('lock')}${shot.takes.length ? '' : ' not shot'}</span>` : ''}<span class="tag ${shot.status}">${rendering ? '<span class="spinner" style="width:10px;height:10px;border-width:1.5px"></span>' : ''}${STATUS_LABEL[shot.status]}</span></span>
        <span class="shot-badges">${shot.takes.length > 1 ? `<span class="tag" title="Takes">${icon('layers')}${shot.takes.length}</span>` : ''}${verdict ? `<span class="tag" title="${esc((verdict.by === 'user' ? 'You' : 'Showrunner') + ': ' + (verdict.note || (verdict.ok ? 'good take' : 'needs a redo')))}" style="background:${verdict.ok ? 'rgba(61,220,151,.9)' : 'rgba(255,93,108,.9)'};color:${verdict.ok ? '#002b19' : '#fff'}">${icon(verdict.ok ? 'check' : 'x')}${verdict.by === 'user' ? 'You' : 'Showrunner'}</span>` : ''}</span>
        <span class="shot-duration" title="Its length in the cut${take?.duration_s && Math.abs(take.duration_s - (cutLengths(state.doc).get(shot.id) || 0)) > 0.05 ? ` (the take lasts ${formatDuration(take.duration_s)})` : ''}">${formatDuration(cutLengths(state.doc).get(shot.id) ?? shot.duration_s)}</span>
        ${rendering ? `<span class="shot-progress"><i style="width:${state.renders[shot.id] ?? 30}%"></i></span>` : ''}
      </div>
      <div class="shot-body"><div class="shot-title">${esc(shot.title || `Shot ${index + 1}`)}</div>
        ${shot.dialogue?.length ? `<div class="shot-line" title="${esc(shot.dialogue.map(item => `${speakerName(item.speaker, doc)}: ${item.line}`).join('\n'))}">${icon('chat')}<span><b>${esc(speakerName(shot.dialogue[0].speaker, doc))}:</b> ${esc(shot.dialogue[0].line)}${shot.dialogue.length > 1 ? ` <span class="muted">+${shot.dialogue.length - 1}</span>` : ''}</span></div>` : ''}
        <div class="shot-sub">${shot.camera ? `${icon('video')}<span>${esc(shot.camera)}</span>` : `<span>${esc(templateLabel(shot.generation?.template))}</span>`}<span class="grow"></span>${castAvatars(shot.cast, doc)}</div></div>`;
    if (take?.kind === 'video') {
      element.addEventListener('mouseenter', () => {
        if (element.querySelector('video')) return;
        const video = Object.assign(document.createElement('video'), { src: actions.file(take.path), muted: true, loop: true, playsInline: true, autoplay: true });
        video.draggable = false;
        element.querySelector('.shot-media').prepend(video);
        video.play().catch(() => {});
      });
      element.addEventListener('mouseleave', () => element.querySelector('video')?.remove());
    }
    return element;
  }

  function renderBoard() {
    const doc = state.doc;
    board.style.setProperty('--card-w', `${state.cardWidth}px`);
    if (!doc.shots.length) {
      const working = actions.busy(doc) || state.live.running;
      board.innerHTML = `<div class="sb-empty"><div class="big-icon">${icon(working ? 'director' : 'wand')}</div>
        <h2>${working ? 'The showrunner is planning…' : 'No shots yet'}</h2>
        <p>${working ? 'Shots appear here as soon as the plan is written. Follow along in the Showrunner panel.' : 'Let the showrunner turn your brief into a shot list, or add shots yourself.'}</p>
        <div class="row">${working ? '<span class="spinner"></span>' : `<button class="btn primary lg" id="empty-plan">${icon('wand')} Plan with the showrunner</button><button class="btn lg" id="empty-add">${icon('plus')} Add a shot</button>`}</div>
        <button class="btn ghost small" id="empty-brief">${icon('brief')} Review the brief</button></div>`;
      board.querySelector('#empty-plan')?.addEventListener('click', () => actions.direct('plan'));
      board.querySelector('#empty-add')?.addEventListener('click', () => addShot());
      board.querySelector('#empty-brief').onclick = () => actions.go('brief');
      cards.clear();
      return;
    }
    const groups = [], lengths = cutLengths(doc);
    doc.shots.forEach((shot, index) => {
      if (!visible(shot)) return;
      const name = shot.section || '';
      if (!groups.length || groups.at(-1).name !== name) groups.push({ name, items: [] });
      groups.at(-1).items.push([shot, index]);
    });
    const fragment = document.createDocumentFragment();
    const seen = new Set();
    for (const group of groups) {
      const section = document.createElement('section');
      section.className = 'sb-section';
      const duration = group.items.reduce((sum, [shot]) => sum + (lengths.get(shot.id) || 0), 0);
      const outline = (doc.outline || []).find(item => item.name === group.name);
      const done = group.items.filter(([shot]) => shot.status === 'approved').length, shot = group.items.filter(([shot]) => actions.take(shot)).length;
      const folded = collapsed.has(group.name) && groups.length > 1;  // a lone sequence never folds away the whole board
      if (groups.length > 1 || group.name) section.innerHTML = `<div class="sb-section-head" data-fold="${esc(group.name)}" title="${folded ? 'Show' : 'Hide'} this sequence's shots">
        <span class="fold">${icon(folded ? 'chevron' : 'chevronDown')}</span><span class="swatch" style="background:${sectionColor(group.name)}"></span><b>${esc(group.name || 'Shots')}</b>
        <span class="muted">${group.items.length} shots · ${formatDuration(duration)}${outline?.duration_s ? ` of ${formatDuration(outline.duration_s)}` : ''}</span>
        <span class="seq-progress" title="${shot} shot · ${done} approved"><i style="width:${100 * shot / group.items.length}%"></i><i class="ok" style="width:${100 * done / group.items.length}%"></i></span>
        ${outline?.summary ? `<span class="muted seq-summary">${esc(outline.summary)}</span>` : ''}</div>`;
      if (folded) { fragment.append(section); group.items.forEach(([item]) => seen.add(item.id)); continue; }
      const grid = document.createElement('div');
      grid.className = 'sb-grid';
      for (const [shot, index] of group.items) {
        const key = signature(shot, index);
        let element = cards.get(shot.id);
        if (!element || element.dataset.sig !== key) {
          element = card(shot, index);
          element.dataset.sig = key;
          cards.set(shot.id, element);
        }
        seen.add(shot.id);
        grid.append(element);
      }
      if (group === groups.at(-1) && state.filter === 'all') grid.append(Object.assign(document.createElement('div'), { className: 'add-shot', innerHTML: `<div class="col" style="align-items:center">${icon('plus')}<span>Add a shot</span></div>`, onclick: () => addShot() }));
      section.append(grid);
      fragment.append(section);
    }
    for (const id of [...cards.keys()]) if (!seen.has(id)) cards.delete(id);
    board.replaceChildren(fragment);
    if (!groups.length) {
      board.innerHTML = `<div class="empty-state" style="margin-top:30px">No shot is ${esc(FILTERS.find(([key]) => key === state.filter)?.[1].toLowerCase() || 'shown')}. <button class="btn small" id="show-all">Show all ${doc.shots.length} shots</button></div>`;
      board.querySelector('#show-all').onclick = () => tools.querySelector('[data-filter="all"]').click();
    }
  }

  // The shots on screen, in order: ranges, arrow keys and bulk actions never reach shots hidden by the filter or a folded sequence.
  function visibleIds() { return state.doc.shots.filter(shot => visible(shot) && !collapsed.has(shot.section || '')).map(shot => shot.id); }

  // Long productions: sequences fold to their header (remembered per production in this browser).
  const foldKey = () => `sr.fold.${state.pid}`;
  let collapsed = new Set();
  try { collapsed = new Set(JSON.parse(localStorage.getItem(foldKey()) || '[]')); } catch { collapsed = new Set(); }

  // Selection, menus and drag and drop (delegated on the board) -----------------------------------------------------------------
  board.addEventListener('click', event => {
    const head = event.target.closest('[data-fold]');
    if (head) {
      const name = head.dataset.fold;
      collapsed.has(name) ? collapsed.delete(name) : collapsed.add(name);
      try { localStorage.setItem(foldKey(), JSON.stringify([...collapsed])); } catch { /* the fold is a convenience */ }
      renderBoard();
      return;
    }
    const element = event.target.closest('.shot');
    if (!element) { if (event.target === board) actions.select(null, { open: false }); return; }
    actions.select(element.dataset.id, { additive: event.ctrlKey || event.metaKey, range: event.shiftKey });
  });
  board.addEventListener('contextmenu', event => {
    const element = event.target.closest('.shot');
    if (!element) return;
    event.preventDefault();
    if (!state.selection.has(element.dataset.id)) actions.select(element.dataset.id, { open: false });
    shotMenu({ x: event.clientX, y: event.clientY }, [...state.selection]);
  });
  board.addEventListener('dragstart', event => {
    const element = event.target.closest('.shot');
    if (!element) return;
    dragIds = state.selection.has(element.dataset.id) ? state.doc.shots.map(shot => shot.id).filter(id => state.selection.has(id)) : [element.dataset.id];
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData(SHOT_TYPE, JSON.stringify(dragIds));
    const image = element.querySelector('img');
    if (image) event.dataTransfer.setDragImage(image, 40, 30);
    requestAnimationFrame(() => dragIds.forEach(id => cards.get(id)?.classList.add('dragging')));
  });
  board.addEventListener('dragend', () => { cards.forEach(element => element.classList.remove('dragging')); clearDrop(); dragIds = null; });
  board.addEventListener('dragover', event => {
    const element = event.target.closest('.shot');
    const types = event.dataTransfer.types;
    if (!element || !(types.includes(SHOT_TYPE) || types.includes('Files') || types.includes(BIN_TYPE))) return;
    event.preventDefault();
    clearDrop();
    if (types.includes(SHOT_TYPE)) {
      const rect = element.getBoundingClientRect();
      const after = event.clientX > rect.left + rect.width / 2;
      element.classList.add(after ? 'drop-after' : 'drop-before');
      dropTarget = { id: element.dataset.id, after };
    } else {
      element.classList.add('file-over');
      dropTarget = { id: element.dataset.id, media: true };
    }
  });
  board.addEventListener('drop', async event => {
    const target = dropTarget;
    clearDrop();
    if (!target) return;
    event.preventDefault();
    event.stopPropagation();
    document.getElementById('veil')?.classList.remove('on');
    const types = event.dataTransfer.types;
    if (types.includes(SHOT_TYPE)) {
      const ids = JSON.parse(event.dataTransfer.getData(SHOT_TYPE));
      if (ids.includes(target.id)) return;
      const order = state.doc.shots.map(shot => shot.id).filter(id => !ids.includes(id));
      const index = order.indexOf(target.id) + (target.after ? 1 : 0);
      const before = order[index] ?? null;
      const section = actions.shot(target.id)?.section || '';  // dropped among another sequence's shots, they join it
      await actions.ops([{ op: 'move_shots', shots: ids, before },
        ...ids.filter(id => (actions.shot(id)?.section || '') !== section).map(id => ({ op: 'update_shot', shot: id, fields: { section } }))]);
    } else if (types.includes(BIN_TYPE)) {
      await actions.ops([{ op: 'bin_to_take', item: event.dataTransfer.getData(BIN_TYPE), shot: target.id }]);
      toast('Added as a take.', 'ok');
    } else if (event.dataTransfer.files.length) {
      await actions.upload([...event.dataTransfer.files].slice(0, 1), 'take', target.id);
      toast('Added as a take.', 'ok');
    }
  });
  function clearDrop() {
    dropTarget = null;
    board.querySelectorAll('.drop-before,.drop-after,.file-over').forEach(element => element.classList.remove('drop-before', 'drop-after', 'file-over'));
  }

  renderTools();
  renderBoard();
  return {
    update() { renderTools(); renderBoard(); },
    selection() { renderTools(); renderBoard(); },
    progress() { renderBoard(); },
    key(event) { boardKeys(event, addShot, visibleIds()); },
    visibleIds,
  };
}

export function shotMenu(anchor, ids) {
  const shots = ids.map(id => actions.shot(id)).filter(Boolean);
  if (!shots.length) return;
  const one = shots.length === 1 ? shots[0] : null;
  const locked = shots.every(shot => shot.locked);
  const included = shots.every(shot => shot.included);
  const take = one && actions.take(one);
  menu(anchor, [
    ...(take ? [{ label: 'Approve take', icon: 'check', action: () => verdict(shots, true) }, { label: 'Needs a redo…', icon: 'x', action: () => verdictWithNote(shots) }, '-'] : []),
    { label: locked ? 'Unlock' : 'Lock (the showrunner keeps it)', icon: locked ? 'unlock' : 'lock', action: () => actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { locked: !locked } }))) },
    { label: included ? 'Exclude from the edit' : 'Include in the edit', icon: included ? 'eyeOff' : 'eye', action: () => actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { included: !included } }))) },
    '-',
    ...(one && take?.settings && Object.keys(take.settings).length ? [{ label: 'Render a new take', icon: 'refresh', action: () => import('./inspector.js').then(module => module.rerun(one.id)) }] : []),
    { label: `Ask the showrunner to re-shoot${shots.length > 1 ? ` ${shots.length} shots` : ''}`, icon: 'director', action: () => actions.direct('reshoot', '', shots.map(shot => shot.id)) },
    { label: 'Tell the showrunner about it…', icon: 'chat', action: async () => { const text = await promptDialog({ title: 'Note for the showrunner', placeholder: 'Make it a slow push-in, keep her red coat…', multiline: true, ok: 'Send' }); if (text?.trim()) actions.steer(`${shots.map(shot => shot.id).join(', ')}: ${text.trim()}`); } },
    '-',
    ...(one ? [{ label: 'Duplicate', icon: 'copy', action: () => duplicate(one) }] : []),
    { label: shots.length > 1 ? `Delete ${shots.length} shots` : 'Delete shot', icon: 'trash', danger: true, action: () => remove(shots) },
  ]);
}

async function verdict(shots, ok, note = '') {
  await actions.ops(shots.filter(shot => actions.take(shot)).map(shot => ({ op: 'verdict', shot: shot.id, ok, note })));
}

async function verdictWithNote(shots) {
  const note = await promptDialog({ title: 'What should change?', text: 'Your note goes into the shot; the showrunner reads it when it re-shoots (Re-shoot, or Shoot for every shot marked for a redo).', placeholder: 'Her face changes in the last second; keep the camera still…', multiline: true, ok: 'Mark for a redo' });
  if (note === null) return;
  await verdict(shots, false, note);
}

async function duplicate(shot) {
  const index = state.doc.shots.findIndex(item => item.id === shot.id);
  const fields = { title: `${shot.title} (copy)`, section: shot.section, intent: shot.intent, camera: shot.camera, cast: shot.cast, duration_s: shot.duration_s, generation: shot.generation, transition: shot.transition, notes: shot.notes };
  await actions.ops([{ op: 'add_shots', position: index + 1, shots: [fields] }]);
}

async function remove(shots) {
  if (!(await confirmDialog({ title: shots.length > 1 ? `Delete ${shots.length} shots?` : `Delete "${shots[0].title || shots[0].id}"?`, text: shots.length > 1 ? 'Their takes stay in the production folder. Undo brings the shots back.' : 'Its takes stay in the production folder. Undo brings the shot back.', ok: 'Delete', danger: true }))) return;
  await actions.ops(shots.map(shot => ({ op: 'remove_shot', shot: shot.id })));
  state.selection.clear();
  actions.select(null, { open: false });
}

function boardKeys(event, addShot, ids) {
  const current = state.focus && ids.includes(state.focus) ? state.focus : null;
  const key = event.key;
  if (['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(key)) {
    event.preventDefault();
    const step = key === 'ArrowRight' || key === 'ArrowDown' ? 1 : -1;
    const next = ids[Math.max(0, Math.min(ids.length - 1, (current ? ids.indexOf(current) : -1) + step))];
    if (next) actions.select(next, { range: event.shiftKey });
    document.querySelector(`.shot[data-id="${next}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    return;
  }
  const shots = [...state.selection].map(id => actions.shot(id)).filter(Boolean);
  if (!shots.length) { if (key.toLowerCase() === 'n') addShot(); return; }
  if (['a', 'x', 'l', 'e', 'n'].includes(key.toLowerCase())) event.preventDefault();  // the key must not reach the note field it opens
  if (key.toLowerCase() === 'a') verdict(shots, true);
  else if (key.toLowerCase() === 'x') verdictWithNote(shots);
  else if (key.toLowerCase() === 'l') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { locked: !shots.every(item => item.locked) } })));
  else if (key.toLowerCase() === 'e') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { included: !shots.every(item => item.included) } })));
  else if (key === 'Delete' || key === 'Backspace') remove(shots);
  else if (key.toLowerCase() === 'n') addShot(state.doc.shots.findIndex(shot => shot.id === shots.at(-1).id) + 1);
  else if (key === 'Escape') actions.select(null, { open: false });
}
