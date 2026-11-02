// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Deliver: the assembled versions of the edit, what is left before the final cut, and the files to take away.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, ago, formatDuration, toast, pickFiles, h, confirmDialog, count } from './ui.js';

export function mountDeliver(container) {
  let selected = null, compareWith = null, changes = null;
  const number = id => state.doc.deliverables.findIndex(item => item.id === id) + 1;  // versions are numbered in the order they were made
  const subtitles = new Map();  // each export's subtitles for the player: a WebVTT object URL, '' without any (read once)
  let showSubtitles = false;
  try { showSubtitles = localStorage.getItem('sr.subtitles') === '1'; } catch { /* a convenience */ }
  function render() {
    const doc = state.doc;
    const versions = [...doc.deliverables].reverse();
    const current = versions.find(item => item.id === selected) || versions[0];
    const other = compareWith && compareWith !== current?.id ? versions.find(item => item.id === compareWith) : null;
    const included = doc.shots.filter(shot => shot.included);
    const shot = included.filter(item => item.takes.length).length;
    const approved = included.filter(item => item.status === 'approved').length;
    const review = included.filter(item => item.status === 'rendered').length;
    const latest = versions[0];
    const stale = latest && latest.edit_key && latest.edit_key !== doc.edit_key;  // the edit changed since the last export
    const method = item => ({ 'stream copy': 'joined without re-encoding', 'single encode': 'encoded once' }[item.method] || item.method || '');
    const name = item => item.path.split('/').pop();
    container.innerHTML = `<div class="deliver">
      <div class="deliver-hero">
        <div class="col" style="gap:10px"><div class="player" id="d-player" style="--aspect:${doc.brief.aspect.replace(':', '/')}">${other ? compareHtml(current, other)
          : current ? `<video id="d-video" data-path="${esc(current.path)}" src="${actions.file(current.path)}" controls playsinline poster="${current.poster ? actions.file(current.poster) : ''}">${subtitleTrack(current.path)}</video>${subtitleSwitch(current.path)}`
          : `<div class="empty"><div class="col" style="align-items:center">${icon('film')}<b>No version yet</b><span>Assemble the edit to watch it here.</span></div></div>`}</div>
          ${other ? `<div class="panel" style="padding:14px 16px"><div class="label">${icon('compare')} From version ${Math.min(number(current.id), number(other.id))} to version ${Math.max(number(current.id), number(other.id))}</div>
            ${changes ? (changes.length ? `<ul class="changes">${changes.map(line => `<li>${esc(line)}</li>`).join('')}</ul>` : '<div class="muted">No change in the edit.</div>') : '<div class="muted">Reading both versions…</div>'}</div>` : ''}</div>
        <div class="col" style="gap:14px">
          ${stale ? `<div class="stale-note">${icon('alert')}<span><b>The edit changed since the last version.</b> Assemble again to include your latest changes.</span></div>` : ''}
          <div class="panel" style="padding:16px"><div class="label">${icon('target')} Before the final cut</div><div class="checklist">
            ${check(shot === included.length && included.length, `${shot}/${included.length} shots shot`)}${check(approved === included.length && included.length, `${approved}/${included.length} approved by you`, review ? `${review} waiting for your review` : '')}
            ${check(doc.brief.song || doc.edit.audio_mode === 'clips', doc.edit.audio_mode === 'song' ? 'Soundtrack: the song' : doc.edit.audio_mode === 'mix' ? 'Soundtrack: song and shot sound' : 'Soundtrack: shot sound')}</div>
            <div class="row" style="margin-top:12px"><button class="btn ${latest && !stale ? '' : 'primary'} grow" id="assemble" ${shot ? '' : 'disabled title="Shoot at least one shot first"'}>${icon('film')} ${latest ? 'Assemble again' : 'Assemble'}</button>
              <button class="btn" id="ask" title="The showrunner assembles the edit, watches it and reports">${icon('director')} Showrunner review</button></div></div>
          ${current ? `<div class="panel" style="padding:16px"><div class="label">${icon('info')} Version ${number(current.id)} · ${esc(clock(current.time))}</div><div class="col" style="gap:6px;font-size:13px">
            <div class="row"><span class="muted grow">Length</span><b>${formatDuration(current.duration_s)}</b></div>
            <div class="row"><span class="muted grow">Picture</span><b>${current.width || '?'}×${current.height || '?'}${current.fps ? ` · ${current.fps} fps` : ''}</b></div>
            <div class="row"><span class="muted grow">Made by</span><b>${current.by === 'deepy' ? 'the showrunner' : 'you'} · ${esc(method(current))}</b></div>
            <div class="row"><span class="muted grow">When</span><b>${ago(current.time)}</b></div></div>
            <a class="btn ${stale ? '' : 'primary'}" style="margin-top:12px;width:100%" href="${actions.file(current.path, true)}" download="${esc(name(current))}">${icon('download')} Download${current === latest && stale ? ' (previous edit)' : ''}</a></div>` : ''}
        </div></div>
      ${soundPanel(doc)}
      ${versions.length ? `<div class="panel"><div class="panel-head"><h3>Versions</h3><span class="hint">click to watch · compare two · open one to bring back what it used</span></div><div class="versions">${versions.map(item => `<div class="version ${item.id === current?.id ? 'on' : ''} ${item.id === other?.id ? 'cmp' : ''}" data-version="${item.id}">
          ${item.poster ? `<img src="${actions.file(item.poster)}" alt="">` : ''}<div class="grow"><b>Version ${number(item.id)}</b> <span class="muted">· ${esc(clock(item.time))}</span><div class="muted" style="font-size:12px">${formatDuration(item.duration_s)} · ${item.width || '?'}×${item.height || '?'} · ${esc(method(item))}</div></div><span class="muted" style="font-size:12px">${ago(item.time)}</span>
          ${item.id !== current?.id ? `<button class="btn small ghost" data-compare="${item.id}" title="Watch it side by side with the version above, and see what changed">${icon('compare')} Compare</button>` : ''}
          <button class="btn small ghost" data-open="${item.id}" title="See what this version used and bring pieces of it back">${icon('layers')} Open</button></div>`).join('')}</div></div>` : ''}
    </div>`;
    container.querySelector('#assemble').onclick = async event => {
      const button = event.currentTarget;
      button.disabled = true;
      button.innerHTML = '<span class="spinner"></span> Assembling…';
      if (await actions.assemble()) selected = null;
      render();
    };
    container.querySelector('#ask').onclick = () => actions.direct('assemble');
    bindSound();
    container.querySelectorAll('[data-version]').forEach(element => element.onclick = () => { selected = element.dataset.version; compareWith = null; render(); });
    container.querySelectorAll('[data-compare]').forEach(button => button.onclick = async event => {
      event.stopPropagation();
      selected = current.id;
      compareWith = button.dataset.compare;
      changes = null;
      render();
      container.querySelector('#d-player').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      try { changes = (await api.version(state.pid, compareWith, current.id)).changes; } catch (error) { changes = [error.message]; }
      render();
    });
    container.querySelectorAll('[data-open]').forEach(button => button.onclick = event => { event.stopPropagation(); openVersion(button.dataset.open); });
    if (other) bindCompare();
    else if (current) subtitles.has(current.path) ? bindSubtitles() : readSubtitles(current.path);
  }

  // The subtitles a version carries, over the player on request: the export keeps them as a track the viewer turns on.
  const subtitleTrack = path => subtitles.get(path) ? `<track kind="subtitles" label="Subtitles" srclang="${esc(state.doc.brief.language || 'en')}" src="${subtitles.get(path)}">` : '';
  const subtitleSwitch = path => subtitles.get(path) ? `<button class="btn small player-subtitles" id="d-subtitles" role="switch" aria-checked="false" title="Show or hide the subtitles this version carries">${icon('captions')}<span>Subtitles: off</span></button>` : '';
  async function readSubtitles(path) {
    subtitles.set(path, '');
    try {
      const { vtt } = await api.subtitles(state.pid, path);
      if (vtt.includes('-->')) subtitles.set(path, URL.createObjectURL(new Blob([vtt], { type: 'text/vtt' })));
    } catch { return; }  // the player plays without them
    const video = container.querySelector('#d-video');
    if (!subtitles.get(path) || video?.dataset.path !== path) return;
    video.insertAdjacentHTML('beforeend', subtitleTrack(path));  // added in place: a version already playing goes on
    video.insertAdjacentHTML('afterend', subtitleSwitch(path));
    bindSubtitles();
  }
  function bindSubtitles() {
    const video = container.querySelector('#d-video'), button = container.querySelector('#d-subtitles'), track = video?.textTracks[0];
    if (!track || !button) return;
    const paint = () => {
      const on = track.mode === 'showing';
      button.classList.toggle('active', on);
      button.setAttribute('aria-checked', String(on));
      button.lastElementChild.textContent = `Subtitles: ${on ? 'on' : 'off'}`;
    };
    button.onclick = () => {
      showSubtitles = track.mode !== 'showing';
      try { localStorage.setItem('sr.subtitles', showSubtitles ? '1' : '0'); } catch { /* a convenience */ }
      track.mode = showSubtitles ? 'showing' : 'hidden';
      paint();
    };
    video.textTracks.onchange = () => { showSubtitles = track.mode === 'showing'; paint(); };  // the player's own captions menu switches them too
    track.mode = showSubtitles ? 'showing' : 'hidden';
    paint();
  }

  // Two versions in one frame, a handle to slide between them, played in sync (as takes are compared in the shot panel).
  function compareHtml(a, b) {
    const poster = item => item.poster ? ` poster="${actions.file(item.poster)}"` : '';
    return `<video id="cmp-a" src="${actions.file(a.path)}"${poster(a)} preload="auto" playsinline muted></video>
      <div class="compare" id="cmp-clip" style="clip-path:inset(0 0 0 50%)"><video id="cmp-b" src="${actions.file(b.path)}"${poster(b)} preload="auto" playsinline muted></video></div>
      <div class="compare-handle" id="cmp-handle" style="left:50%"></div>
      <span class="tag compare-label" style="left:8px">A · version ${number(a.id)}</span><span class="tag compare-label" style="right:8px;left:auto;background:rgba(56,225,208,.85);color:#002b28">B · version ${number(b.id)}</span>
      <div class="row" style="position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:4;flex-wrap:nowrap"><button class="btn small" id="cmp-play">${icon('play')} Play both</button>
        <button class="btn small" id="cmp-sound" title="Hear the other version">${icon('audio')} Sound: A</button><button class="btn small" id="cmp-done">${icon('x')} Done</button></div>`;
  }
  function bindCompare() {
    const player = container.querySelector('#d-player'), handle = container.querySelector('#cmp-handle'), clip = container.querySelector('#cmp-clip');
    const a = container.querySelector('#cmp-a'), b = container.querySelector('#cmp-b');
    a.muted = false;
    handle.addEventListener('pointerdown', event => {
      handle.setPointerCapture(event.pointerId);
      handle.onpointermove = move => {
        const rect = player.getBoundingClientRect(), ratio = Math.max(0, Math.min(1, (move.clientX - rect.left) / rect.width));
        handle.style.left = `${ratio * 100}%`;
        clip.style.clipPath = `inset(0 0 0 ${ratio * 100}%)`;
      };
    });
    handle.addEventListener('pointerup', () => { handle.onpointermove = null; });
    container.querySelector('#cmp-play').onclick = event => {
      if (a.paused) { b.currentTime = a.currentTime; a.play(); b.play(); event.currentTarget.innerHTML = `${icon('pause')} Pause`; }
      else { a.pause(); b.pause(); event.currentTarget.innerHTML = `${icon('play')} Play both`; }
    };
    container.querySelector('#cmp-sound').onclick = event => { a.muted = !a.muted; b.muted = !a.muted; event.currentTarget.innerHTML = `${icon('audio')} Sound: ${a.muted ? 'B' : 'A'}`; };
    container.querySelector('#cmp-done').onclick = () => { compareWith = null; render(); };
  }

  // A version opened: what it used (shots and their takes, cast, text, sound), each piece one click from coming back, and the whole
  // version to restore or to start a new production from.
  async function openVersion(id) {
    let data;
    try { data = await api.version(state.pid, id); } catch (error) { toast(error.message, 'error', 8000); return; }
    const back = h('<div class="modal-back"><div class="modal version-view" role="dialog" aria-modal="true"></div></div>');
    const modal = back.querySelector('.modal');
    const keys = event => { if (event.key === 'Escape' && !document.querySelector('.modal-back ~ .modal-back')) { event.stopPropagation(); close(); } };  // a confirm on top closes first
    const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
    back.addEventListener('click', event => { if (event.target === back) close(); });
    document.addEventListener('keydown', keys);
    document.getElementById('overlay-root').append(back);
    let now = state.doc;
    const paint = () => { modal.innerHTML = versionHtml(number(id), data, now); bind(); };
    const bind = () => {
      modal.querySelector('[data-close]').onclick = close;
      modal.querySelectorAll('[data-bring]').forEach(button => button.onclick = async () => {
        button.disabled = true;
        try {
          await api.bringBack(state.pid, id, button.dataset.bring, button.dataset.item);
          [now, data] = await Promise.all([api.get(state.pid), api.version(state.pid, id)]);
          toast('Brought back. Undo: Ctrl+Z', 'ok');
          paint();
        } catch (error) { toast(error.message, 'error', 7000); button.disabled = false; }
      });
      modal.querySelector('[data-restore]').onclick = async () => {
        if (!(await confirmDialog({ title: `Restore version ${number(id)}?`, text: 'The production goes back to what this version used: shots, takes, cast, text and sound. Nothing is deleted: takes made since stay on their shots and shots added since stay, out of the edit. Undo brings today\'s state back.', ok: 'Restore' }))) return;
        try { await api.restoreVersion(state.pid, id); toast(`Version ${number(id)} restored. Undo: Ctrl+Z`, 'ok', 6000); close(); } catch (error) { toast(error.message, 'error', 7000); }
      };
      modal.querySelector('[data-branch]').onclick = async () => {
        try { const copy = await api.duplicateVersion(state.pid, id); toast(`"${copy.title}" created`, 'ok', 6000); close(); location.hash = `#/p/${copy.id}/storyboard`; } catch (error) { toast(error.message, 'error', 7000); }
      };
    };
    paint();
  }

  // Sound and subtitles: the sound in one line (placed on the Timeline, where it plays against the picture), and the subtitle files.
  function bindSound() {
    const edit = () => state.doc.edit;
    container.querySelector('#edit-sound').onclick = () => actions.go('timeline');
    container.querySelectorAll('[data-subtitle]').forEach(button => button.onclick = () => actions.ops([{ op: 'edit', fields: { subtitles: edit().subtitles.filter((_, number) => number !== +button.dataset.subtitle) } }]));
    container.querySelector('#add-srt')?.addEventListener('click', async () => {
      const files = await pickFiles('.srt', false);
      if (files.length) try { await api.upload(state.pid, files[0], 'subtitles'); toast('Subtitles added.', 'ok'); } catch (error) { toast(error.message, 'error'); }
    });
  }

  render();
  return { update: render, assembling: render, destroy() { subtitles.forEach(url => url && URL.revokeObjectURL(url)); } };
}

// When a version was made: the time, with the day when it was not today.
function clock(timestamp) {
  const date = new Date(timestamp * 1000), today = date.toDateString() === new Date().toDateString();
  return date.toLocaleString(undefined, today ? { hour: '2-digit', minute: '2-digit' } : { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

// What a version used, beside today's production: a piece already in use is marked, the others get a "Bring back".
function versionHtml(number, data, now) {
  const doc = data.doc, version = data.deliverable;
  const used = shot => shot.takes.find(take => take.id === shot.chosen_take) || null;
  const today = id => now.shots.find(shot => shot.id === id);
  const shots = doc.shots.filter(shot => shot.included).map((shot, index) => {
    const take = used(shot), current = today(shot.id);
    const state_ = !current ? ['shot', 'Bring back the shot'] : !take ? null : used(current)?.path === take.path ? null : ['take', 'Use this take again'];
    return `<div class="vshot">${take?.poster ? `<img src="${actions.file(take.poster)}" alt="">` : '<div class="vshot-empty"></div>'}
      <div class="col" style="gap:2px;min-width:0"><b>${index + 1}. ${esc(shot.title || 'Untitled')}</b><span class="muted" style="font-size:11.5px">${take ? `take ${take.id.slice(1)}` : 'no take'}${current ? '' : ' · removed since'}</span>
      ${state_ ? `<button class="btn small" data-bring="${state_[0]}" data-item="${shot.id}">${icon('undo')} ${state_[1]}</button>` : '<span class="tag" style="align-self:flex-start">In the edit</span>'}</div></div>`;
  }).join('');
  const sameMember = (old, member) => member && ['name', 'master', 'voice'].every(key => (old[key] || null) === (member[key] || null)) && JSON.stringify(old.looks || []) === JSON.stringify(member.looks || []);
  const cast = doc.cast.map(member => `<div class="vcast">${member.master ? `<img src="${actions.file(member.master)}" alt="">` : '<div class="vshot-empty"></div>'}
    <div class="col" style="gap:4px;min-width:0"><b>${esc(member.name || member.id)}</b>${member.voice ? `<audio controls preload="none" src="${actions.file(member.voice)}"></audio>` : ''}
    ${sameMember(member, now.cast.find(item => item.id === member.id)) ? '<span class="tag" style="align-self:flex-start">As today</span>' : `<button class="btn small" data-bring="cast" data-item="${member.id}">${icon('undo')} Bring back as in this version</button>`}</div></div>`).join('');
  const texts = doc.edit.overlays.map(item => `<div class="vrow"><span class="chip">${esc(item.style || 'picture')}</span><span class="grow">${esc(item.text || item.path.split('/').pop())}</span>
    ${now.edit.overlays.some(other => other.path === item.path) ? '<span class="tag">In the edit</span>' : `<button class="btn small" data-bring="overlay" data-item="${item.id}">${icon('undo')} Bring back</button>`}</div>`).join('');
  const tracks = doc.edit.audio_tracks.map((item, index) => `<div class="vrow"><span class="grow">${esc(item.path.split('/').pop())}</span><audio controls preload="none" src="${actions.file(item.path)}"></audio>
    ${now.edit.audio_tracks.some(other => other.path === item.path) ? '<span class="tag">In the edit</span>' : `<button class="btn small" data-bring="track" data-item="${index}">${icon('undo')} Bring back</button>`}</div>`).join('');
  const subtitles = doc.edit.subtitles.map((item, index) => `<div class="vrow"><span class="grow">${esc(item.path.split('/').pop())}</span>
    ${now.edit.subtitles.some(other => other.path === item.path) ? '<span class="tag">In the edit</span>' : `<button class="btn small" data-bring="subtitle" data-item="${index}">${icon('undo')} Bring back</button>`}</div>`).join('');
  return `<div class="row"><h3 class="grow">Version ${number} <span class="muted" style="font-weight:500;font-size:13px">${esc(clock(version.time))} · ${formatDuration(version.duration_s)}</span></h3>
      <button class="btn small ghost icon" data-close title="Close">${icon('x')}</button></div>
    <div class="version-body">
      <div><div class="label">${icon('compare')} Changed since</div>${data.changes.length ? `<ul class="changes">${data.changes.map(line => `<li>${esc(line)}</li>`).join('')}</ul>` : '<div class="muted">Nothing: today\'s edit is this version.</div>'}</div>
      <div><div class="label">${icon('film')} Shots</div><div class="vshots">${shots || '<div class="muted">No shot.</div>'}</div></div>
      ${cast ? `<div><div class="label">${icon('users')} Cast</div><div class="vshots">${cast}</div></div>` : ''}
      ${texts ? `<div><div class="label">${icon('note')} Text</div>${texts}</div>` : ''}
      ${tracks ? `<div><div class="label">${icon('audio')} Sound tracks</div>${tracks}</div>` : ''}
      ${subtitles ? `<div><div class="label">${icon('note')} Subtitle files</div>${subtitles}</div>` : ''}
    </div>
    <div class="actions"><button class="btn" data-branch title="A new production starting from this version; this one stays as it is">${icon('layers')} Duplicate from this version</button>
      <button class="btn primary" data-restore title="Go back to this version, as one change you can undo">${icon('undo')} Restore this version</button></div>`;
}

function soundPanel(doc) {
  const edit = doc.edit, name = path => path.split('/').pop().replace(/-[0-9a-f]{6}(\.\w+)$/, '$1');
  const lines = doc.shots.filter(shot => shot.included && shot.takes.length).reduce((count, shot) => count + (shot.dialogue || []).filter(line => (line.line || '').trim()).length, 0);  // what assemble subtitles
  const soundtrack = { song: 'Song only', clips: 'Shot sound only', mix: 'Song and shot sound' }[doc.brief.song ? edit.audio_mode : 'clips'];
  const tracks = edit.audio_tracks.length ? count(edit.audio_tracks.length, 'track') + ' over the edit' : 'no track over the edit';
  return `<div class="panel"><div class="panel-head"><h3>Sound and subtitles</h3><span class="hint">What the export plays and shows, besides the picture</span></div>
    <div class="col" style="gap:14px">
      <div class="row" style="flex-wrap:wrap"><span class="label" style="margin:0">${icon('music')} Sound</span><span>${soundtrack} · ${tracks}</span><span class="grow"></span>
        <button class="btn small" id="edit-sound" title="The soundtrack and the tracks (narration, score, effects) are placed on the Timeline">${icon('timeline')} Edit on the Timeline</button></div>
      <div><div class="label">${icon('note')} Subtitles <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">a track the viewer turns on, never burnt into the picture</span></div>
        ${edit.subtitles.length ? edit.subtitles.map((item, index) => `<div class="row" style="font-size:13px"><b>${esc(name(item.path))}</b><span class="chip">${esc(item.language || 'eng')}</span><span class="grow"></span>
          <a class="btn small ghost" href="${actions.file(item.path, true)}" download="${esc(name(item.path))}">${icon('download')}</a><button class="btn small ghost icon" data-subtitle="${index}" title="Remove">${icon('x')}</button></div>`).join('')
          : `<div class="muted" style="font-size:12.5px">${lines ? `From the shots' dialogue: ${lines} line${lines > 1 ? 's' : ''}, timed on the cut and written as in the script. An SRT file you add replaces them.` : 'None: the words written in the shots\' dialogue become subtitles, or add an SRT file.'}</div>`}
        <button class="btn small" id="add-srt" style="margin-top:8px">${icon('plus')} SRT file</button></div>
    </div></div>`;
}

function check(ok, text, extra = '') {
  return `<div class="check ${ok ? 'ok' : 'todo'}">${icon(ok ? 'check' : 'clock')}<span>${esc(text)}${extra ? ` <span class="muted">· ${esc(extra)}</span>` : ''}</span></div>`;
}
