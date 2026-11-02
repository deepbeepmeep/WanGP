// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Shot inspector: watch and compare takes, judge them, and edit everything the director will use.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, art, fieldSaver, formatDuration, toast, promptDialog, menu, confirmDialog, shotRefs, shotText } from './ui.js';
import { shotMenu } from './storyboard.js';
import { choiceLabel, templateLabel } from './models.js';

const TRANSITIONS = [['cut', 'Cut'], ['crossfade', 'Dissolve'], ['fade_black', 'Fade black'], ['fade_white', 'Fade white']];

export async function rerun(shotId) {
  try { await api.rerun(state.pid, shotId); toast('Rendering a new take…'); } catch (error) { toast(error.message, 'error', 7000); }
}

export function mountInspector(root) {
  let shownId = null, compareId = null, signatureShown = '', promptOpen = false;  // the prompt is the showrunner's: folded unless opened
  const saveShot = fieldSaver((shotId, fields) => actions.ops([{ op: 'update_shot', shot: shotId, fields }], { quiet: true }).catch(error => toast(error.message, 'error')), 500);

  function render() {
    const scroll = root.scrollTop, same = state.selection.size === 1 && [...state.selection][0] === shownId;
    paint();
    if (same) root.scrollTop = scroll;  // a change to the shot keeps the panel where the user was
  }

  function paint() {
    const doc = state.doc;
    const ids = [...state.selection];
    if (ids.length > 1) return renderMany(ids.map(id => actions.shot(id)).filter(Boolean));
    const shot = ids.length ? actions.shot(ids[0]) : null;
    if (!shot) return renderOverview(doc);
    if (shot.id !== shownId) { compareId = null; saveShot.flush(); root.scrollTop = 0; }
    shownId = shot.id;
    const index = doc.shots.indexOf(shot);
    const take = actions.take(shot);
    const compare = compareId && shot.takes.find(item => item.id === compareId && item.id !== take?.id);
    const verdict = take?.verdict;
    const rendering = shot.status === 'rendering' || state.renders[shot.id] != null;
    const sections = [...new Set(doc.shots.map(item => item.section).filter(Boolean))];
    const settings = shot.generation?.settings || {};
    root.innerHTML = `<div class="inspector">
      <div class="insp-head"><span class="shot-num">${index + 1}</span><input class="input bare grow" id="i-title" value="${esc(shot.title)}" placeholder="Shot ${index + 1}" style="font-weight:650;font-size:15px">
        <button class="btn small icon ${shot.locked ? 'active' : ''}" id="i-lock" title="${shot.locked ? 'Locked: the showrunner keeps it. Click to unlock' : 'Lock: the showrunner will not change it'}">${icon(shot.locked ? 'lock' : 'unlock')}</button>
        <button class="btn small icon" id="i-more" title="More">${icon('more')}</button></div>
      <div class="player" id="i-player">${playerHtml(shot, take, compare)}</div>
      ${shot.takes.length ? `<div><div class="label">${icon('layers')} Takes <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">click to use · right-click to compare or remove</span></div>
        <div class="takes">${shot.takes.map((item, number) => `<div class="take ${item.id === take?.id ? 'on' : ''} ${item.id === compare?.id ? 'cmp' : ''}" data-take="${item.id}" title="${esc(shotText(item.note || "", doc.shots, doc.cast))} · ${item.by === 'deepy' ? 'the showrunner' : item.by === 'user' ? 'you' : item.by}">
          ${item.poster ? `<img src="${actions.file(item.poster)}" alt="">` : `<div style="aspect-ratio:16/9;${art(item.id)}"></div>`}<span class="take-label">T${number + 1}${item.duration_s ? ` · ${(+item.duration_s).toFixed(1)}s` : ''}</span>
          ${item.verdict ? `<span class="verdict-dot" style="background:${item.verdict.ok ? 'var(--green)' : 'var(--red)'}"></span>` : ''}</div>`).join('')}</div></div>` : ''}
      ${shot.status === 'failed' ? failure(shot, doc) : ''}
      ${take ? `<div class="verdict-row">
          <button class="btn approve ${shot.status === 'approved' ? 'active' : ''}" id="i-approve">${icon('check')} Approve</button>
          <button class="btn reject ${shot.status === 'rejected' && verdict?.by === 'user' ? 'active' : ''}" id="i-reject" title="Mark the take for a redo, with a note for the showrunner">${icon('x')} Redo…</button>
          ${take.settings && Object.keys(take.settings).length ? `<button class="btn" id="i-rerun" title="Render a new take now from this shot's settings and prompt, without the showrunner" ${rendering ? 'disabled' : ''}>${icon('refresh')} New take</button>` : ''}</div>
        ${[verdict, verdict?.by === 'user' && take.showrunner_verdict].filter(Boolean).map(item => `<div class="verdict-note ${item.ok ? 'ok' : 'bad'}">${icon(item.ok ? 'check' : 'alert')}<span><b>${item.by === 'user' ? 'You' : 'Showrunner'}:</b> ${shotRefs(item.note || (item.by === 'user' ? (item.ok ? 'Approved.' : 'Marked for a redo.') : item.ok ? 'Good take.' : 'Needs a retry.'), doc.shots, doc.cast)}</span></div>`).join('')}
        ${checksHtml(take)}` : `<div class="verdict-row"><button class="btn primary" id="i-shoot" ${rendering ? 'disabled' : ''}>${icon('clapper')} ${rendering ? 'Rendering…' : 'Shoot this shot'}</button></div>`}
      <div class="field"><div class="label">${icon('sparkles')} What happens</div><textarea class="textarea" id="i-intent" rows="3" placeholder="${doc.brief.kind === 'music_video' ? 'The singer turns to the camera as the chorus starts…' : 'She hears the radio crackle and turns to the window…'}">${esc(shot.intent)}</textarea></div>
      <div class="field"><div class="label">${icon('chat')} Dialogue <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">words spoken or sung in this shot</span></div>
        <div class="dialogue" id="i-dialogue">${(shot.dialogue || []).map(item => lineRow(item, doc)).join('')}</div>
        <button class="btn small ghost" id="i-add-line" style="align-self:flex-start">${icon('plus')} Line</button></div>
      <details class="details" id="i-prompt-box" ${promptOpen ? 'open' : ''}><summary>${icon('chevron')} Prompt <span class="muted" id="i-count" style="font-weight:500">written by the showrunner from what happens, the camera and the cast · ${(settings.prompt || '').length} chars</span></summary>
        <textarea class="textarea" id="i-prompt" rows="6" style="margin-top:8px" placeholder="The showrunner writes the full generation prompt here.">${esc(settings.prompt || '')}</textarea></details>
      <div class="insp-grid">
        <div class="field"><div class="label">${icon('clock')} Duration</div><input class="input" id="i-duration" type="number" min="0.5" max="60" step="0.1" value="${shot.duration_s ?? ''}"></div>
        <div class="field"><div class="label">${icon('layers')} Section</div><input class="input" id="i-section" list="i-sections" value="${esc(shot.section)}"><datalist id="i-sections">${sections.map(name => `<option value="${esc(name)}">`).join('')}</datalist></div>
        <div class="field" style="grid-column:1/-1"><div class="label">${icon('video')} Camera</div><input class="input" id="i-camera" value="${esc(shot.camera)}" placeholder="Slow dolly-in, low angle…"></div>
        <div class="field" style="grid-column:1/-1"><div class="label">${icon('bolt')} Model</div><select class="select" id="i-model" title="The Deepy template that renders this shot">
          <option value="${esc(shot.generation?.template || 'gen_video')}">${esc(templateLabel(shot.generation?.template))}</option></select></div>
      </div>
      ${doc.cast.length ? `<div class="field"><div class="label">${icon('users')} In this shot</div><div class="cast-toggle">${doc.cast.map(member => {
        const image = member.master || member.images[0];
        return `<button class="chip ${shot.cast.includes(member.id) ? 'on' : ''}" data-cast="${member.id}"><span class="avatar-xs" style="${image ? `background-image:url('${actions.file(image)}')` : ''}">${image ? '' : esc((member.name || '?')[0])}</span>${esc(member.name || member.id)}</button>`;
      }).join('')}</div></div>` : ''}
      <div class="field"><div class="label">${icon('transition')} Transition into this shot</div>
        <div class="row"><div class="seg" id="i-transition">${TRANSITIONS.map(([key, label]) => `<button class="${(shot.transition?.type || 'cut') === key ? 'on' : ''}" data-tr="${key}">${label}</button>`).join('')}</div>
        ${(shot.transition?.type || 'cut') !== 'cut' ? `<input class="input" id="i-tr-d" type="number" min="0.1" max="3" step="0.1" value="${shot.transition?.duration_s || 0.5}" style="width:74px" title="Seconds">` : ''}</div></div>
      ${take?.duration_s ? `<div class="field"><div class="label">${icon('scissors')} Trim <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">seconds of the take</span></div>
        <div class="row"><input class="input" id="i-in" type="number" min="0" step="0.05" value="${shot.trim?.[0] ?? 0}" style="width:90px"><span class="muted">to</span><input class="input" id="i-out" type="number" min="0" step="0.05" value="${shot.trim?.[1] ?? ''}" placeholder="${(+take.duration_s).toFixed(2)}" style="width:90px">
        <button class="btn small ghost" id="i-trim-reset">Reset</button><span class="grow"></span><label class="row muted" style="font-size:12.5px;cursor:pointer"><span class="toggle ${shot.included ? 'on' : ''}" id="i-included"></span>In the edit</label></div></div>` : ''}
      <div class="field"><div class="label">${icon('note')} Notes for the showrunner</div><textarea class="textarea" id="i-notes" rows="2" placeholder="Anything to keep in mind for this shot…">${esc(shot.notes)}</textarea></div>
      <div class="row"><button class="btn grow" id="i-ask">${icon('director')} Ask the showrunner</button><button class="btn grow" id="i-reshoot">${icon('refresh')} Re-shoot</button></div>
      <details class="details"><summary>${icon('chevron')} Generation settings · ${esc(templateLabel(shot.generation?.template))}</summary>
        <div class="col" style="margin-top:10px"><input class="input mono" id="i-template" value="${esc(shot.generation?.template || 'gen_video')}">
        <textarea class="textarea json" id="i-settings" spellcheck="false">${esc(JSON.stringify(settings, null, 2))}</textarea><div class="muted" style="font-size:12px">JSON merged into the shot's settings when you leave the field.</div></div></details>
    </div>`;
    signatureShown = JSON.stringify(shot);
    bindShot(shot, take);
  }

  function playerHtml(shot, take, compare) {
    if (!take) return `<div class="ph-art" style="position:absolute;inset:0;${art(shot.id + shot.section, shot.intent)}"></div><div class="empty" style="color:rgba(255,255,255,.85)">${esc(shot.intent || 'Not shot yet')}</div>`;
    if (take.kind === 'image') return `<img src="${actions.file(take.path)}" alt="">`;
    if (take.kind === 'audio') return `<div class="empty"><audio controls src="${actions.file(take.path)}" style="width:90%"></audio></div>`;
    if (!compare) return `<video src="${actions.file(take.path)}" controls loop playsinline preload="metadata" poster="${take.poster ? actions.file(take.poster) : ''}"></video>`;
    return `<video id="cmp-a" src="${actions.file(take.path)}" loop playsinline muted></video>
      <div class="compare" id="cmp-clip" style="clip-path:inset(0 0 0 50%)"><video id="cmp-b" src="${actions.file(compare.path)}" loop playsinline muted></video></div>
      <div class="compare-handle" id="cmp-handle" style="left:50%"></div>
      <span class="tag compare-label" style="left:8px">A · used</span><span class="tag compare-label" style="right:8px;left:auto;background:rgba(56,225,208,.85);color:#002b28">B</span>
      <div class="row" style="position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:4;flex-wrap:nowrap"><button class="btn small" id="cmp-play">${icon('play')} Play both</button>
        <button class="btn small" id="cmp-use" title="Use take B in the edit">${icon('check')} Use B</button><button class="btn small" id="cmp-done">${icon('x')} Done</button></div>`;
  }

  function checksHtml(take) {
    const checks = take.checks || {};
    const chips = [];
    if (take.duration_s) chips.push(`${icon('clock')}${(+take.duration_s).toFixed(2)} s`);
    if (take.width) chips.push(`${icon('ratio')}${take.width}×${take.height}`);
    if (take.fps) chips.push(`${take.fps} fps`);
    if (Array.isArray(checks.black_ranges_seconds)) chips.push(checks.black_ranges_seconds.length ? `<span style="color:var(--red)">${icon('alert')}Black frames</span>` : `${icon('check')}No black frames`);
    if (Array.isArray(checks.cut_times_seconds)) chips.push(checks.cut_times_seconds.length ? `<span style="color:var(--amber)">${icon('scissors')}${checks.cut_times_seconds.length} cut(s) inside</span>` : `${icon('check')}One continuous shot`);
    if (checks.lufs != null) chips.push(`${icon('audio')}${checks.lufs} LUFS`);
    return chips.length ? `<div class="checks">${chips.map(chip => `<span class="chip">${chip}</span>`).join('')}</div>` : '';
  }

  function bindShot(shot, take) {
    const id = shot.id;
    const field = (selector, key, transform = value => value) => root.querySelector(selector)?.addEventListener('input', event => saveShot(id, { [key]: transform(event.target.value) }));
    field('#i-title', 'title');
    field('#i-intent', 'intent');
    field('#i-camera', 'camera');
    field('#i-section', 'section');
    field('#i-notes', 'notes');
    field('#i-duration', 'duration_s', value => Math.max(0.5, +value || 0.5));
    root.querySelector('#i-prompt-box').addEventListener('toggle', event => { promptOpen = event.target.open; });
    root.querySelector('#i-prompt').addEventListener('input', event => {
      root.querySelector('#i-count').textContent = `written by the showrunner from what happens, the camera and the cast · ${event.target.value.length} chars`;
      saveShot(id, { generation: { settings: { prompt: event.target.value } } });
    });
    const saveDialogue = () => saveShot(id, { dialogue: [...root.querySelectorAll('#i-dialogue .line-row')].map(row => ({
      speaker: row.querySelector('select').value, line: row.querySelector('textarea').value, note: row.querySelector('input').value })) });
    root.querySelector('#i-dialogue').addEventListener('input', saveDialogue);
    root.querySelector('#i-dialogue').addEventListener('change', saveDialogue);
    root.querySelector('#i-dialogue').addEventListener('click', event => { if (event.target.closest('[data-remove-line]')) { event.target.closest('.line-row').remove(); saveDialogue(); } });
    root.querySelector('#i-add-line').onclick = () => {  // saved once it has words
      const box = root.querySelector('#i-dialogue');
      box.insertAdjacentHTML('beforeend', lineRow({ speaker: shot.cast[0] || 'NARRATOR', line: '', note: '' }, state.doc));
      box.lastElementChild.querySelector('textarea').focus();
    };
    root.querySelector('#i-lock').onclick = () => actions.ops([{ op: 'update_shot', shot: id, fields: { locked: !shot.locked } }]);
    root.querySelector('#i-more').onclick = event => shotMenu(event.currentTarget, [id]);
    root.querySelectorAll('[data-cast]').forEach(button => button.onclick = () => {
      const cast = new Set(shot.cast);
      cast.has(button.dataset.cast) ? cast.delete(button.dataset.cast) : cast.add(button.dataset.cast);
      actions.ops([{ op: 'update_shot', shot: id, fields: { cast: [...cast] } }]);
    });
    root.querySelectorAll('[data-tr]').forEach(button => button.onclick = () => actions.ops([{ op: 'update_shot', shot: id, fields: { transition: { type: button.dataset.tr, duration_s: shot.transition?.duration_s || 0.5 } } }]));
    root.querySelector('#i-tr-d')?.addEventListener('change', event => actions.ops([{ op: 'update_shot', shot: id, fields: { transition: { duration_s: +event.target.value } } }]));
    const trim = () => {
      const start = +root.querySelector('#i-in').value || 0;
      const endValue = root.querySelector('#i-out').value;
      saveShot(id, { trim: [Math.max(0, start), endValue === '' ? null : Math.max(start + 0.1, +endValue)] });
    };
    root.querySelector('#i-in')?.addEventListener('input', trim);
    root.querySelector('#i-out')?.addEventListener('input', trim);
    root.querySelector('#i-trim-reset')?.addEventListener('click', () => actions.ops([{ op: 'update_shot', shot: id, fields: { trim: [0, null] } }]));
    root.querySelector('#i-included')?.parentElement.addEventListener('click', () => actions.ops([{ op: 'update_shot', shot: id, fields: { included: !shot.included } }]));
    root.querySelectorAll('[data-take]').forEach(element => {
      element.onclick = () => actions.ops([{ op: 'update_shot', shot: id, fields: { chosen_take: element.dataset.take } }]);
      element.oncontextmenu = event => {
        event.preventDefault();
        menu({ x: event.clientX, y: event.clientY }, [
          { label: 'Use this take', icon: 'check', action: () => actions.ops([{ op: 'update_shot', shot: id, fields: { chosen_take: element.dataset.take } }]) },
          { label: 'Compare with the used take', icon: 'compare', action: () => { compareId = element.dataset.take; render(); } },
          '-',
          { label: 'Remove take', icon: 'trash', danger: true, action: async () => { if (await confirmDialog({ title: 'Remove this take?', text: 'The file stays in the production folder.', ok: 'Remove', danger: true })) actions.ops([{ op: 'remove_take', shot: id, take: element.dataset.take }]); } },
        ]);
      };
    });
    root.querySelector('#i-approve')?.addEventListener('click', () => actions.ops([{ op: 'verdict', shot: id, take: take.id, ok: true, note: '' }]));
    root.querySelector('#i-reject')?.addEventListener('click', async () => {
      const note = await promptDialog({ title: 'What should change?', text: 'The showrunner reads your note before re-shooting.', placeholder: 'The hands look wrong; make the move slower…', multiline: true, ok: 'Mark for a redo' });
      if (note !== null) actions.ops([{ op: 'verdict', shot: id, take: take.id, ok: false, note }]);
    });
    root.querySelector('#i-rerun')?.addEventListener('click', () => rerun(id));
    root.querySelector('#i-shoot')?.addEventListener('click', () => actions.direct('reshoot', '', [id]));
    root.querySelector('#i-reshoot').onclick = () => actions.direct('reshoot', '', [id]);
    root.querySelector('#i-ask').onclick = async () => {
      const text = await promptDialog({ title: `Note for the showrunner about "${shot.title || id}"`, placeholder: 'Make it a slow push-in and keep her red coat…', multiline: true, ok: 'Send' });
      if (text?.trim()) actions.steer(`${id} (${shot.title}): ${text.trim()}`);
    };
    fillModels(root.querySelector('#i-model'), shot.generation?.template || 'gen_video');
    // arrow keys walk the list: the model is applied once the choice settles (a pause or leaving the list), not at every step
    const model = root.querySelector('#i-model');
    let modelTimer = null;
    const applyModel = async () => {
      clearTimeout(modelTimer);
      modelTimer = null;
      if (model.value === (shot.generation?.template || 'gen_video')) return;
      await actions.ops([{ op: 'update_shot', shot: id, fields: { generation: { template: model.value } } }]);
      if (shot.takes.length) toast('Model changed: New take or Re-shoot renders the shot with it.', 'info', 5000);
    };
    model.addEventListener('change', () => { clearTimeout(modelTimer); modelTimer = setTimeout(applyModel, 900); });
    model.addEventListener('blur', () => { if (modelTimer) applyModel(); });
    root.querySelector('#i-template').addEventListener('change', event => actions.ops([{ op: 'update_shot', shot: id, fields: { generation: { template: event.target.value.trim() || 'gen_video' } } }]));
    root.querySelector('#i-settings').addEventListener('change', event => {
      try { actions.ops([{ op: 'update_shot', shot: id, fields: { generation: { settings: JSON.parse(event.target.value || '{}') } } }]); }
      catch (error) { toast(`Invalid JSON: ${error.message}`, 'error'); }
    });
    if (root.querySelector('#cmp-handle')) bindCompare();
  }

  // The shot's model: Deepy's templates grouped by the kind of shot they make (the brief sets each kind's default).
  async function fillModels(select, current) {
    let roles;
    try { roles = await actions.templateRoles(); } catch { return; }
    if (!root.contains(select)) return;
    const seen = new Set([current]);
    const groups = roles.filter(role => ['cast_shots', 'cast_shots_lip_synced', 'shots_without_cast', 'images'].includes(role.id))
      .map(role => ({ ...role, choices: role.choices.filter(choice => choice.value === current || !seen.has(choice.value) && seen.add(choice.value)).map(choice => ({ ...choice, label: choiceLabel(role, choice) })) }));  // each model once, named as in Default models
    const known = current === 'draw_svg' || groups.some(role => role.choices.some(choice => choice.value === current));
    select.innerHTML = (known ? '' : `<option value="${esc(current)}" selected>${esc(templateLabel(current))}</option>`)
      + groups.map(role => `<optgroup label="${esc(role.label)}">${role.choices.map(choice => `<option value="${esc(choice.value)}" ${choice.value === current ? 'selected' : ''}>${esc(choice.label)}${choice.value === (role.value || role.auto) ? ' (the brief\'s choice)' : ''}</option>`).join('')}</optgroup>`).join('')
      + `<optgroup label="Other"><option value="draw_svg" ${current === 'draw_svg' ? 'selected' : ''}>${templateLabel('draw_svg')}</option></optgroup>`;
  }

  function bindCompare() {
    const player = root.querySelector('#i-player'), handle = root.querySelector('#cmp-handle'), clip = root.querySelector('#cmp-clip');
    const a = root.querySelector('#cmp-a'), b = root.querySelector('#cmp-b');
    const move = event => {
      const rect = player.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      handle.style.left = `${ratio * 100}%`;
      clip.style.clipPath = `inset(0 0 0 ${ratio * 100}%)`;
    };
    handle.addEventListener('pointerdown', event => { handle.setPointerCapture(event.pointerId); handle.onpointermove = move; });
    handle.addEventListener('pointerup', () => { handle.onpointermove = null; });
    root.querySelector('#cmp-done').onclick = () => { compareId = null; render(); };
    root.querySelector('#cmp-use').onclick = () => { const id = compareId; compareId = null; actions.ops([{ op: 'update_shot', shot: shownId, fields: { chosen_take: id } }]); };
    root.querySelector('#cmp-play').onclick = event => {
      const playing = !a.paused;
      if (playing) { a.pause(); b.pause(); } else { b.currentTime = a.currentTime = 0; a.play(); b.play(); }
      event.currentTarget.innerHTML = `${icon(playing ? 'play' : 'pause')} ${playing ? 'Play both' : 'Pause'}`;
    };
    a.ontimeupdate = () => { if (Math.abs(b.currentTime - a.currentTime) > .08) b.currentTime = a.currentTime; };
  }

  function renderMany(shots) {
    const total = shots.reduce((sum, shot) => sum + (actions.take(shot)?.duration_s || shot.duration_s || 0), 0);
    root.innerHTML = `<div class="inspector"><div class="insp-head"><b style="font-size:15px">${shots.length} shots selected</b><span class="muted">· ${formatDuration(total)}</span></div>
      <div class="refs" style="grid-template-columns:repeat(3,1fr)">${shots.slice(0, 9).map(shot => { const take = actions.take(shot); return `<div class="ref" style="aspect-ratio:16/9">${take?.poster ? `<img src="${actions.file(take.poster)}" alt="">` : `<div style="position:absolute;inset:0;${art(shot.id + shot.section, shot.intent)}"></div>`}</div>`; }).join('')}</div>
      <div class="col">
        <button class="btn approve" data-bulk="approve">${icon('check')} Approve their takes</button>
        <button class="btn" data-bulk="lock">${icon('lock')} Lock / unlock</button>
        <button class="btn" data-bulk="include">${icon('eye')} Include / exclude</button>
        <button class="btn" data-bulk="reshoot">${icon('refresh')} Ask the showrunner to re-shoot</button>
        <div class="field"><div class="label">${icon('layers')} Move to section</div><input class="input" id="bulk-section" placeholder="Chorus 2"></div>
        <div class="field"><div class="label">${icon('transition')} Transition</div><div class="seg" id="bulk-tr">${TRANSITIONS.map(([key, label]) => `<button data-tr="${key}">${label}</button>`).join('')}</div></div>
        <button class="btn danger" data-bulk="delete">${icon('trash')} Delete</button></div></div>`;
    root.querySelectorAll('[data-bulk]').forEach(button => button.onclick = () => {
      const kind = button.dataset.bulk;
      if (kind === 'approve') actions.ops(shots.filter(shot => actions.take(shot)).map(shot => ({ op: 'verdict', shot: shot.id, ok: true, note: '' })));
      if (kind === 'lock') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { locked: !shots.every(item => item.locked) } })));
      if (kind === 'include') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { included: !shots.every(item => item.included) } })));
      if (kind === 'reshoot') actions.direct('reshoot', '', shots.map(shot => shot.id));
      if (kind === 'delete') shotMenu(button, shots.map(shot => shot.id));
    });
    root.querySelector('#bulk-section').addEventListener('change', event => actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { section: event.target.value } }))));
    root.querySelectorAll('#bulk-tr [data-tr]').forEach(button => button.onclick = () => actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { transition: { type: button.dataset.tr } } }))));
    shownId = null;
  }

  function renderOverview(doc) {
    shownId = null;
    const count = status => doc.shots.filter(shot => shot.status === status).length;
    const minutes = (doc.budget?.gpu_seconds || 0) / 60;
    root.innerHTML = `<div class="inspector">
      <div class="panel" style="padding:16px"><div class="label">${icon('target')} Production</div>
        <div class="checklist">
          ${check(doc.brief.text || doc.brief.script, 'Brief written')}${check(doc.shots.length, `${doc.shots.length} shots planned`)}
          ${check(doc.shots.length && doc.shots.every(shot => shot.takes.length || !shot.included), `${doc.shots.filter(shot => shot.takes.length).length} shots shot`)}
          ${check(doc.shots.length && count('approved') === doc.shots.length, `${count('approved')} approved by you`)}${check(doc.deliverables.length, doc.deliverables.length ? 'Edit assembled' : 'Edit not assembled yet')}
        </div></div>
      <div class="three" style="gap:8px">${stat(count('rendered'), 'to review', 'var(--amber)')}${stat(count('rejected') + count('failed'), 'to redo', 'var(--red)')}${stat(minutes.toFixed(1), 'GPU min', 'var(--cyan)')}</div>
      <details class="panel details" style="padding:12px 16px"><summary class="label" style="margin:0;cursor:pointer">${icon('keyboard')} Keyboard shortcuts</summary>
        <div class="col" style="gap:6px;font-size:12.5px;margin-top:10px">${[['←→', 'Select shot'], ['A', 'Approve'], ['X', 'Redo, with a note'], ['L', 'Lock'], ['E', 'Include / exclude'], ['N', 'New shot'], ['Del', 'Delete'], ['Space', 'Play (timeline)'], ['1-5', 'Views'], ['D', 'Showrunner'], ['Ctrl+Z', 'Undo'], ['?', 'Guide']].map(([key, text]) => `<div class="row"><span class="kbd">${key}</span><span class="muted">${text}</span></div>`).join('')}</div></details>
      <p class="muted" style="font-size:12.5px;margin:0">Select a shot to watch its takes, judge them and edit what the showrunner will use. Drop media from your computer or the bin onto a shot to make it a take.</p>
    </div>`;
  }

  return {
    update() {
      const active = document.activeElement;
      if (active && root.contains(active) && active.matches('textarea, input, select')) return;  // no redraw under the user's hands
      const id = state.selection.size === 1 ? [...state.selection][0] : null;
      const shot = id && actions.shot(id);
      if (shot && shot.id === shownId && JSON.stringify(shot) === signatureShown && !root.querySelector('#cmp-handle')) return;
      render();
    },
  };
}

// Why the shot's last render failed: the newest log line about it (New take and render write the error there).
function failure(shot, doc) {
  const entry = [...doc.log].reverse().find(item => new RegExp(`\\b${shot.id}\\b`).test(item.text || '') && /fail/i.test(item.text || ''));
  return `<div class="verdict-note bad">${icon('alert')}<span><b>Failed:</b> ${entry ? shotRefs(entry.text, doc.shots, doc.cast) : 'the render returned nothing; ask the showrunner or try again.'}</span></div>`;
}

function check(ok, text) { return `<div class="check ${ok ? 'ok' : 'todo'}">${icon(ok ? 'check' : 'clock')}<span>${esc(text)}</span></div>`; }
function stat(value, label, color) { return `<div class="panel" style="padding:12px;text-align:center"><div style="font-size:22px;font-weight:750;color:${color}">${value}</div><div class="muted" style="font-size:11.5px">${label}</div></div>`; }

function lineRow(item, doc) {
  const speakers = [...doc.cast.map(member => [member.id, member.name || member.id]), ['NARRATOR', 'Narrator']];
  if (item.speaker && !speakers.some(([value]) => value === item.speaker)) speakers.push([item.speaker, item.speaker]);
  return `<div class="line-row"><select class="select">${speakers.map(([value, label]) => `<option value="${esc(value)}" ${value === item.speaker ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select>
    <button class="btn small ghost icon" data-remove-line title="Remove this line">${icon('x')}</button>
    <textarea class="textarea" rows="2" placeholder="The exact words">${esc(item.line)}</textarea>
    <input class="input" value="${esc(item.note || '')}" placeholder="Delivery: whispered, smiling, angry…"></div>`;
}
