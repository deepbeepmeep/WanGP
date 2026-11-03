// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Timeline: the edit over the song. Preview it live, drag shots to reorder, drag their edges to trim (snapping to beats),
// set transitions, and see sections, bars and the waveform the cuts should follow.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, art, h, formatTime, formatDuration, sectionColor, drawWave, toast, menu, confirmDialog, pickFiles, availability, AUDIO_ACCEPT } from './ui.js';
import { Sequencer, segments } from './sequencer.js';
import { shotMenu } from './storyboard.js';

const TRANSITION_ORDER = ['cut', 'crossfade', 'fade_black', 'fade_white'];
const TRANSITION_LABEL = { cut: 'Cut', crossfade: 'Dissolve', fade_black: 'Fade through black', fade_white: 'Fade through white' };
const LANES = { ruler: [0, 30], sections: [30, 24], video: [56, 84], text: [140, 32], words: [172, 24], audio: [172, 64] };
const TEXT_STYLES = [['title', 'Title'], ['lower_third', 'Lower third'], ['caption', 'Caption'], ['chapter', 'Chapter card'], ['end_card', 'End card']];
const trackLengths = new Map(), trackProbes = new Map();  // a sound track's length, read from the file (its probe kept until then)
const trackName = track => track.name || track.path.split('/').pop().replace(/^\d{4}-\d\d-\d\d-\d\dh\d\dm\d\ds_seed\d+_/, '').replace(/\.\w+$/, '');
const trackLength = track => Math.max(0.05, (track.out_s ?? trackLengths.get(track.path) ?? 3) - (+track.in_s || 0));  // what it plays of its file
const TEXT_DESIGNS = [['classic', 'Classic'], ['bold', 'Bold'], ['elegant', 'Elegant'], ['minimal', 'Minimal'], ['typewriter', 'Typewriter'], ['storybook', 'Storybook']];

export function mountTimeline(container, doc) {
  container.innerHTML = `<div class="tl">
    <div class="monitor-wrap"><div class="monitor"><div class="stage-box" id="stage"><div class="text-preview" id="text-preview"></div></div><div class="monitor-overlay" id="mon-info"></div></div>
      <div class="monitor-side"><div class="now-card" id="now"></div><div class="now-card" id="edit-card"></div></div></div>
    <div class="transport" id="transport"></div>
    <div class="tracks" id="tracks"><div class="tracks-inner" id="inner">
      <div class="track-labels" id="labels"></div>
      <div class="ruler" id="ruler"></div><div class="sections-lane" id="sections"></div><div class="video-lane" id="video"></div><div class="text-lane" id="text"></div><div class="words-lane" id="words"></div>
      <div class="audio-lane" id="audio"><canvas id="wave"></canvas><div class="beats-lane" id="beats"></div><div class="sound-lane" id="sound-tracks"></div></div>
      <div class="playhead" id="playhead"></div><div class="snap-guide hidden" id="guide"></div></div></div>
  </div>`;
  const $ = selector => container.querySelector(selector);
  const tracks = $('#tracks'), inner = $('#inner'), stage = $('#stage');
  let pxPerSecond = 0, snap = localStorage.getItem('sr.snap') !== '0', segs = [], sequencer = null, songKey = '', drag = null, fitted = false;
  let selectedText = null, selectedTrack = null, lastClick = { id: null, time: 0 };
  const doubleClick = id => { const now = performance.now(), double = lastClick.id === id && now - lastClick.time < 400; lastClick = { id: double ? null : id, time: now }; return double; };

  function songUrl(doc) { return doc.brief.song ? actions.file(doc.brief.song) : null; }
  function analysis() { return state.doc.audio?.analysis; }
  function totalSeconds() { return Math.max(segs.length ? segs.at(-1).start + segs.at(-1).duration : 0, 1); }
  function x(seconds) { return 96 + seconds * pxPerSecond; }
  function secondsAt(clientX) { return Math.max(0, (clientX - inner.getBoundingClientRect().left - 96) / pxPerSecond); }

  function setupSequencer() {
    const doc = state.doc;
    const key = `${doc.brief.song}|${doc.edit.audio_mode}|${doc.edit.song_start_s || 0}`;
    if (sequencer && key === songKey) return;
    const time = sequencer?.time || 0;
    sequencer?.destroy();
    songKey = key;
    sequencer = new Sequencer(stage, { songUrl: doc.edit.audio_mode !== 'clips' ? songUrl(doc) : null, songOffset: +(doc.edit.song_start_s || 0), songMode: doc.edit.audio_mode, fileUrl: actions.file,
      onTime: time => { positionPlayhead(time); renderTime(time); previewText(time); }, onSegment: seg => renderNow(seg) });
    sequencer.time = time;
  }

  function fit() {
    const width = tracks.clientWidth - 96 - 24;
    const total = Math.max(totalSeconds(), analysis()?.duration_s || 0, 5);
    pxPerSecond = Math.max(1.5, width / total);
    fitted = true;
  }

  function render() {
    const doc = state.doc;
    segs = segments(doc, actions.take);
    setupSequencer();
    sequencer.setTracks(doc.edit.audio_tracks);
    sequencer.setSegments(segs);
    if (!fitted) fit();
    const song = analysis();
    const total = Math.max(totalSeconds(), song?.duration_s || 0);
    inner.style.width = `${x(total) + 40}px`;
    renderTransport();
    renderLabels();
    renderRuler(total);
    renderSections(song);
    renderClips();
    renderText();
    renderWords();
    renderAudio(song, total);
    renderEditCard();
    positionPlayhead(sequencer.time);
  }

  function renderTransport() {
    const doc = state.doc;
    const empty = segs.length ? '' : 'No shot in the edit yet: plan or add shots on the storyboard.';
    $('#transport').innerHTML = `<button class="play-btn" id="play" ${availability(empty, 'Play / pause (Space)')}>${icon(sequencer?.playing ? 'pause' : 'play')}</button>
      <button class="btn ghost small icon" id="back" ${availability(empty, 'To the start (Home)')}>${icon('undo')}</button>
      <span class="time" id="time"></span>
      <span class="sep"></span>
      <div class="seg" id="mode" title="Soundtrack of the preview and of the assembled edit">${[['song', 'Song'], ['clips', 'Shot sound'], ['mix', 'Mix']].map(([key, label]) => `<button class="${doc.edit.audio_mode === key ? 'on' : ''}" data-mode="${key}">${label}</button>`).join('')}</div>
      <button class="btn small ${snap ? 'active' : ''}" id="snap" title="Snap trims to beats and bars">${icon('magnet')} Snap</button>
      <div class="grow"></div>
      <button class="btn ghost small icon" id="zoom-out" title="Zoom out">${icon('zoomOut')}</button>
      <button class="btn ghost small" id="zoom-fit">Fit</button>
      <button class="btn ghost small icon" id="zoom-in" title="Zoom in">${icon('zoomIn')}</button>
      <span class="sep"></span>
      <button class="btn small" id="add-text" title="Lay a title, a lower third, a caption or a card over the edit at the playhead">${icon('plus')} Text</button>
      <button class="btn small" id="add-track" title="Narration, score or effects over the edit: it starts at 0 s, drag it on the Sound lane into place and click it for its level">${icon('plus')} Audio</button>
      <button class="btn primary small" id="assemble" ${availability(actions.assembleBlocked(doc))}>${icon('film')} Assemble</button>`;
    $('#play').onclick = togglePlay;
    $('#back').onclick = () => sequencer.seek(0);
    $('#snap').onclick = () => { snap = !snap; localStorage.setItem('sr.snap', snap ? '1' : '0'); renderTransport(); };
    $('#zoom-in').onclick = () => zoom(1.4);
    $('#zoom-out').onclick = () => zoom(1 / 1.4);
    $('#zoom-fit').onclick = () => { fit(); render(); };
    container.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => actions.ops([{ op: 'edit', fields: { audio_mode: button.dataset.mode } }]));
    $('#assemble').onclick = assemble;
    $('#add-text').onclick = () => editText(null);
    $('#add-track').onclick = async () => { const files = await pickFiles(AUDIO_ACCEPT, false); if (files.length) actions.upload(files, 'track'); };
    renderTime(sequencer?.time || 0);
  }

  async function assemble() {
    const button = $('#assemble');
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span> Assembling…';
    if (await actions.assemble()) actions.go('deliver');
    else { button.disabled = false; button.innerHTML = `${icon('film')} Assemble`; }
  }

  function renderTime(time) {
    const element = $('#time');
    if (element) element.textContent = `${formatTime(time, true)} / ${formatTime(sequencer?.total || 0, true)}`;
  }

  function renderLabels() {
    const spoken = state.doc.shots.some(shot => shot.dialogue?.length);  // no words, no lane: the sound comes up
    const lane = (key, iconName, label, shift = 0) => `<div style="top:${LANES[key][0] + shift}px;height:${LANES[key][1]}px">${icon(iconName)}${label}</div>`;
    $('#labels').innerHTML = lane('ruler', 'clock', 'Time') + lane('sections', 'layers', 'Parts') + lane('video', 'video', 'Shots') + lane('text', 'note', 'Text')
      + (spoken ? lane('words', 'chat', 'Words') : '') + lane('audio', 'music', state.doc.brief.song ? 'Song' : 'Sound', spoken ? LANES.words[1] : 0);
  }

  function renderRuler(total) {
    const step = [1, 2, 5, 10, 15, 30, 60].find(value => value * pxPerSecond >= 70) || 120;
    let html = '';
    for (let second = 0; second <= total + step; second += step) {
      html += `<div class="tick major" style="left:${x(second)}px"><label>${formatTime(second)}</label></div>`;
      for (let minor = 1; minor < 5; minor++) html += `<div class="tick minor" style="left:${x(second + minor * step / 5)}px"></div>`;
    }
    $('#ruler').innerHTML = html;
  }

  // The parts as the board names them (the shots' sections, on the cut); before the plan has any, the parts heard in the song.
  function renderSections(song) {
    const planned = sectionsFromShots(), offset = +(state.doc.edit.song_start_s || 0);
    const sections = planned.length ? planned : (song?.sections || []).map(section => ({ ...section, start_s: section.start_s - offset, end_s: section.end_s - offset }));
    $('#sections').innerHTML = sections.map(section => `<div class="sec" style="left:${x(section.start_s)}px;width:${Math.max(4, (section.end_s - section.start_s) * pxPerSecond - 3)}px;background:${sectionColor(section.name)}" title="${esc(section.name)} ${formatTime(section.start_s)}–${formatTime(section.end_s)}">${esc(section.name)}${section.label ? ` · ${esc(section.label)}` : ''}</div>`).join('');
  }

  function sectionsFromShots() {
    const result = [];
    for (const seg of segs) {
      const name = seg.shot.section || '';
      if (!name) continue;
      if (result.length && result.at(-1).name === name) result.at(-1).end_s = seg.start + seg.duration;
      else result.push({ name, start_s: seg.start, end_s: seg.start + seg.duration });
    }
    return result;
  }

  function renderClips() {
    const lane = $('#video');
    lane.innerHTML = segs.map((seg, index) => {
      const take = seg.take;
      const left = x(seg.start), width = Math.max(14, seg.duration * pxPerSecond - 2);
      const strip = take?.strip ? `background-image:url('${actions.file(take.strip)}')` : take?.poster ? `background-image:url('${actions.file(take.poster)}');background-size:auto 100%` : '';
      const selected = state.selection.has(seg.shot.id);
      const pin = index ? `<div class="transition-pin ${seg.transition ? 'fx' : ''}" data-pin="${seg.shot.id}" style="left:${left + (seg.transition ? seg.transition.duration * pxPerSecond / 2 : 0)}px" title="${TRANSITION_LABEL[seg.shot.transition?.type || 'cut']} into shot ${index + 1}: click to change">${icon(seg.transition ? 'transition' : 'scissors')}</div>` : '';
      return `${pin}<div class="clip ${take ? '' : 'planned'} ${selected ? 'selected' : ''}" data-clip="${seg.shot.id}" style="left:${left}px;width:${width}px;${strip}">
        ${take ? '' : `<div class="clip-art" style="${art(seg.shot.id + seg.shot.section, seg.shot.intent)}"></div>`}
        <span class="clip-label" title="${index + 1} · ${esc(seg.shot.title || '')}${seg.speed !== 1 ? ` · plays at ${seg.speed}×` : ''}">${width < 46 ? index + 1 : `${index + 1} · ${esc(seg.shot.title || 'Untitled')}${seg.speed !== 1 ? ` · ${seg.speed}×` : ''}`}</span>
        <span class="handle l" data-edge="l"></span><span class="handle r" data-edge="r"></span></div>`;
    }).join('') || `<div class="muted" style="position:absolute;left:110px;top:30px">No shot in the edit yet: plan or add shots on the storyboard.</div>`;
  }

  // The spoken lines where their subtitles sit: a line starts with its shot and lasts its recorded length, several share their shot by their number of characters.
  function renderWords() {
    const name = id => state.doc.cast.find(member => member.id === id)?.name || '';
    $('#words').innerHTML = segs.flatMap(seg => {
      const lines = (seg.shot.dialogue || []).filter(item => item.line?.trim()), total = lines.reduce((sum, item) => sum + item.line.length, 0);
      const span = lines.length === 1 && lines[0].duration_s ? Math.min(seg.duration, lines[0].duration_s / seg.speed + 0.3) : seg.duration;
      let start = seg.start;
      return lines.map(item => {
        const length = span * item.line.length / total, left = x(start), width = Math.max(14, length * pxPerSecond - 2), who = name(item.speaker);
        start += length;
        return `<div class="word-clip" style="left:${left}px;width:${width}px" title="${esc(who ? `${who}: ${item.line}` : item.line)} · ${length.toFixed(1)} s">${esc(item.line)}</div>`;
      });
    }).join('');
    $('#words').classList.toggle('hidden', !segs.some(seg => seg.shot.dialogue?.length));
  }

  // Text over the edit: titles, lower thirds, captions and cards, drawn at the production's format and faded in and out.
  function overlays() { return state.doc.edit.overlays || []; }

  function renderText() {
    const lane = $('#text');
    const rows = [], row = {};
    for (const item of [...overlays()].sort((a, b) => a.start_s - b.start_s)) {  // a text that overlaps another goes on the next row
      const index = rows.findIndex(end => end <= item.start_s + 0.01);
      row[item.id] = index < 0 ? rows.push(item.end_s) - 1 : (rows[index] = item.end_s, index);
    }
    const height = rows.length > 1 ? (LANES.text[1] - 6) / rows.length : 0;
    lane.innerHTML = overlays().map(item => `<div class="text-clip ${item.by === 'user' ? '' : 'by-director'} ${selectedText === item.id ? 'selected' : ''}" data-text="${item.id}" style="left:${x(item.start_s)}px;width:${Math.max(18, (item.end_s - item.start_s) * pxPerSecond - 2)}px${height ? `;top:${3 + row[item.id] * height}px;height:${height - 1}px;bottom:auto` : ''}"
        title="${esc(item.text || 'Picture')}${item.subtitle ? ` · ${esc(item.subtitle)}` : ''} · ${item.by === 'user' ? 'yours' : 'by the showrunner'} · double-click to edit">
        <span>${icon('note')}${esc(item.text || 'Picture')}</span><span class="handle l" data-edge="l"></span><span class="handle r" data-edge="r"></span></div>`).join('')
      || `<span class="muted lane-hint" style="left:${x(0) + 12}px">Titles, names and cards: + Text in the bar above</span>`;
    previewText(sequencer?.time || 0);
  }

  function previewText(time) {
    const box = $('#text-preview');
    if (!box) return;
    const closes = item => item.style === 'end_card' && totalSeconds() - item.end_s < 0.25;  // it stays to the last frame, as assemble.py draws it
    const active = overlays().filter(item => time >= item.start_s - 0.02 && (time < item.end_s || closes(item)) && item.path);  // a text added at the playhead shows at once
    const key = active.map(item => item.id + item.path).join('|');
    if (box.dataset.key !== key) {
      box.dataset.key = key;
      box.innerHTML = active.map(item => `<img data-ov="${item.id}" src="${actions.file(item.path)}" alt="">`).join('');
    }
    for (const item of active) {
      const fade = Math.min(item.fade_s || 0, (item.end_s - item.start_s) / (closes(item) ? 1 : 3));
      const opacity = fade > 0.01 && sequencer?.playing ? Math.min(1, (time - item.start_s) / fade, closes(item) ? 1 : (item.end_s - time) / fade) : 1;  // fades while playing, full while paused
      const image = box.querySelector(`[data-ov="${item.id}"]`);
      if (image) image.style.opacity = Math.max(0, opacity).toFixed(3);
    }
  }

  async function editText(item) {
    const start = item ? item.start_s : +(sequencer?.time || 0).toFixed(2);
    const back = h(`<div class="modal-back"><div class="modal" role="dialog" aria-modal="true" style="width:min(520px,94vw)"><h3>${item ? 'Edit the text' : 'Text over the edit'}</h3>
      <div class="col" style="gap:10px">
        <div class="seg" id="t-style">${TEXT_STYLES.map(([key, label]) => `<button class="${(item?.style || 'title') === key ? 'on' : ''}" data-style="${key}">${label}</button>`).join('')}</div>
        <input class="input" id="t-text" placeholder="The words, as they should appear" value="${esc(item?.text || '')}">
        <input class="input" id="t-sub" placeholder="Second line (optional): a role, a place, a date" value="${esc(item?.subtitle || '')}">
        <label class="row" style="gap:8px"><span class="muted">Design</span><select class="input" id="t-design" style="flex:1">
          <option value="">The production's (${esc((TEXT_DESIGNS.find(([key]) => key === state.doc.style.text_design) || TEXT_DESIGNS[0])[1])})</option>
          ${TEXT_DESIGNS.map(([key, label]) => `<option value="${key}" ${item?.design === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
        <div class="row"><span class="muted">From</span><input class="input" id="t-start" type="number" min="0" step="0.1" value="${start}" style="width:90px"><span class="muted">for</span>
          <input class="input" id="t-dur" type="number" min="0.5" step="0.1" value="${item ? +(item.end_s - item.start_s).toFixed(2) : 3}" style="width:80px"><span class="muted">s</span></div>
        <div class="muted" style="font-size:12px">Chapter and end cards cover the picture; the other styles sit over it.</div></div>
      <div class="actions">${item ? '<button class="btn ghost danger" data-r="del">Delete</button><span class="grow"></span>' : ''}<button class="btn ghost" data-r="0">Cancel</button><button class="btn primary" data-r="1">${item ? 'Save' : 'Add'}</button></div></div></div>`);
    let style = item?.style || 'title';
    back.querySelectorAll('[data-style]').forEach(button => button.onclick = () => { style = button.dataset.style; back.querySelectorAll('[data-style]').forEach(other => other.classList.toggle('on', other === button)); });
    const close = () => back.remove();
    back.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
    back.addEventListener('click', async event => {
      if (event.target === back) return close();
      const button = event.target.closest('[data-r]');
      if (!button) return;
      if (button.dataset.r === '0') return close();
      if (button.dataset.r === 'del') { close(); return removeText(item); }
      const text = back.querySelector('#t-text').value.trim(), subtitle = back.querySelector('#t-sub').value.trim(), design = back.querySelector('#t-design').value;
      const startS = Math.max(0, +back.querySelector('#t-start').value || 0), duration = Math.max(0.5, +back.querySelector('#t-dur').value || 3);
      if (!text) { toast('Write the text first.', 'error'); return; }
      button.disabled = true;
      try {
        if (item) await api.changeText(state.pid, item.id, { text, subtitle, style, design, start_s: startS, end_s: startS + duration });
        else await api.addText(state.pid, { text, subtitle, style, design, start_s: startS, duration_s: duration });
        close();
      } catch (error) { toast(error.message, 'error', 6000); button.disabled = false; }
    });
    document.getElementById('overlay-root').append(back);
    back.querySelector('#t-text').focus();
  }

  async function removeText(item) {
    if (await confirmDialog({ title: `Delete "${item.text || 'this picture'}"?`, text: 'Undo brings it back.', ok: 'Delete', danger: true })) actions.ops([{ op: 'remove_overlay', overlay: item.id }]);
  }

  function renderAudio(song, total) {
    const canvas = $('#wave');
    const audioLane = $('#audio');
    const duration = song?.duration_s || 0;
    audioLane.style.width = `${Math.max(x(Math.max(total, duration)), 200)}px`;
    if (song?.waveform && state.doc.edit.audio_mode !== 'clips') {
      const offset = +(state.doc.edit.song_start_s || 0);
      canvas.style.left = `${x(-offset)}px`;
      canvas.style.width = `${duration * pxPerSecond}px`;
      drawWave(canvas, song.waveform, { color: '#38e1d0', color2: '#5aa7ff', progress: (sequencer.time + offset) / duration });
      const bars = new Set((song.bars || []).map(value => value.toFixed(2)));
      const showBeats = pxPerSecond * 60 / (song.bpm || 120) > 6;
      $('#beats').innerHTML = (showBeats ? song.beats : song.bars || []).map(beat => `<i class="${bars.has(beat.toFixed(2)) ? 'bar' : ''}" style="left:${x(beat - offset)}px"></i>`).join('');
    } else {
      canvas.style.width = '0px';
      $('#beats').innerHTML = state.doc.edit.audio_tracks.length ? '' : `<span class="muted" style="position:absolute;left:110px;top:22px;font-size:12px">${state.doc.brief.song ? 'Shot sound: each take plays its own audio.' : 'Each shot plays its own sound; narration and music laid over the edit appear here.'}</span>`;
    }
    renderTracks();
  }

  // The sound tracks laid over the edit (narration, score, a shot's detached sound): a block each, as long as what it plays of its file,
  // dragged to move it and by its edges to play less or more of the file.
  function renderTracks() {
    $('#sound-tracks').innerHTML = state.doc.edit.audio_tracks.map((track, index) => {
      if (!trackLengths.has(track.path)) {
        trackLengths.set(track.path, 3);
        const probe = new Audio();
        probe.preload = 'metadata';
        probe.onloadedmetadata = () => { trackLengths.set(track.path, probe.duration || 3); trackProbes.delete(track.path); renderTracks(); };
        trackProbes.set(track.path, probe);
        probe.src = actions.file(track.path);
      }
      return `<div class="track-clip ${selectedTrack === index ? 'selected' : ''}" data-track="${index}" style="left:${x(+track.start_s || 0)}px;width:${Math.max(18, trackLength(track) * pxPerSecond - 2)}px;top:${4 + (index % 2) * 28}px"
        title="${esc(trackName(track))} · from ${formatTime(+track.start_s || 0)} · ${(+track.gain_db || 0).toFixed(0)} dB · click for its level and fades, drag to move, drag its edges to shorten or extend it">${icon('audio')}${esc(trackName(track))}<span class="handle l" data-edge="l"></span><span class="handle r" data-edge="r"></span></div>`;
    }).join('');
  }

  function renderNow(seg) {
    const take = seg?.take;
    const index = segs.indexOf(seg);
    $('#now').innerHTML = seg ? `<div class="label">${icon('target')} Now playing</div><div class="big">${index + 1}<span class="muted" style="font-size:14px;font-weight:500"> / ${segs.length}</span></div>
      <b>${esc(seg.shot.title || seg.shot.id)}</b><div class="muted" style="font-size:12.5px;margin-top:4px">${esc(seg.shot.intent || '')}</div>
      <div class="row" style="margin-top:8px;flex-wrap:wrap"><span class="tag ${seg.shot.status}">${esc(seg.shot.status)}</span><span class="chip">${formatDuration(seg.duration)}</span>${seg.speed !== 1 ? `<span class="chip">${seg.speed}×</span>` : ''}${take ? '' : '<span class="chip">Animatic</span>'}</div>` : '';
    $('#mon-info').innerHTML = seg ? `<span class="tag">${index + 1}/${segs.length}</span>${take ? '' : '<span class="tag planned">Not shot yet</span>'}` : '';
  }

  // Beside the monitor: the edit at a glance with its sound (song start, tracks), or the track clicked on the Sound lane.
  function renderEditCard() {
    const doc = state.doc;
    const card = $('#edit-card');
    const track = selectedTrack != null ? doc.edit.audio_tracks[selectedTrack] : null;
    if (track) {
      card.innerHTML = `<div class="row"><div class="label grow" style="margin:0">${icon('audio')} Sound track</div><button class="btn small ghost icon" data-close title="Close (Escape)">${icon('x')}</button></div>
        <b class="track-title" title="${esc(track.path)}">${esc(trackName(track))}</b>
        <audio controls preload="metadata" src="${actions.file(track.path)}"></audio>
        <div class="col" style="gap:6px;font-size:12.5px">
          <label class="row"><span class="muted grow">Starts at</span><input class="input" data-start type="number" min="0" step="0.1" value="${+track.start_s || 0}" style="width:76px"><span class="muted">s</span></label>
          <div class="row" title="The part of the file it plays, in seconds"><span class="muted grow">Plays</span><input class="input" data-in type="number" min="0" step="0.1" value="${+track.in_s || 0}" style="width:64px" aria-label="From"><span class="muted">to</span><input class="input" data-out type="number" min="0" step="0.1" value="${track.out_s ?? ''}" placeholder="${(trackLengths.get(track.path) ?? 0).toFixed(1)}" style="width:64px" aria-label="To"></div>
          <div class="row" title="Seconds of fade at its start and at its end"><span class="muted grow">Fades</span><input class="input" data-fade-in type="number" min="0" step="0.1" value="${+track.fade_in_s || 0}" style="width:64px" aria-label="Fade in"><span class="muted">in</span><input class="input" data-fade-out type="number" min="0" step="0.1" value="${+track.fade_out_s || 0}" style="width:64px" aria-label="Fade out"><span class="muted">out</span></div>
          <label class="row"><span class="muted grow">Level</span><input type="range" class="range" data-gain min="-30" max="6" step="1" value="${+track.gain_db || 0}" style="width:104px"><span class="mono" data-gain-v style="width:42px;text-align:right">${(+track.gain_db || 0).toFixed(0)} dB</span></label></div>
        <button class="btn small danger" data-remove style="margin-top:8px">${icon('trash')} Remove from the edit</button>`;
      const setTrack = fields => actions.ops([{ op: 'edit', fields: { audio_tracks: state.doc.edit.audio_tracks.map((item, index) => index === selectedTrack ? { ...item, ...fields } : item) } }]);
      card.querySelector('[data-close]').onclick = () => selectTrack(null);
      card.querySelector('[data-start]').onchange = event => setTrack({ start_s: Math.max(0, +event.target.value || 0) });
      card.querySelector('[data-in]').onchange = event => setTrack({ in_s: Math.max(0, Math.min(+event.target.value || 0, (track.out_s ?? Infinity) - 0.05)) });
      card.querySelector('[data-out]').onchange = event => setTrack({ out_s: event.target.value === '' ? null : Math.max((+track.in_s || 0) + 0.05, +event.target.value) });
      card.querySelector('[data-fade-in]').onchange = event => setTrack({ fade_in_s: Math.max(0, +event.target.value || 0) });
      card.querySelector('[data-fade-out]').onchange = event => setTrack({ fade_out_s: Math.max(0, +event.target.value || 0) });
      card.querySelector('[data-gain]').oninput = event => { card.querySelector('[data-gain-v]').textContent = `${(+event.target.value).toFixed(0)} dB`; };
      card.querySelector('[data-gain]').onchange = event => setTrack({ gain_db: +event.target.value });
      card.querySelector('[data-remove]').onclick = () => removeTrack(selectedTrack);
      return;
    }
    const shot = segs.filter(seg => seg.take).length;
    card.innerHTML = `<div class="label">${icon('film')} Edit</div><div class="col" style="gap:6px;font-size:12.5px">
      <div class="row"><span class="muted grow">Length</span><b>${formatDuration(sequencer.total || 0)}</b></div>
      <div class="row"><span class="muted grow">Shots shot</span><b>${shot}/${segs.length}</b></div>
      ${analysis() ? `<div class="row"><span class="muted grow">Song</span><b>${formatDuration(analysis().duration_s)} · ${Math.round(analysis().bpm || 0)} BPM</b></div>` : ''}
      ${analysis() && doc.edit.audio_mode !== 'clips' && analysis().duration_s - (doc.edit.song_start_s || 0) - (sequencer.total || 0) > 0.5
        ? `<div class="row" style="color:var(--amber)">${icon('alert')}<span>The picture ends ${formatDuration(analysis().duration_s - (doc.edit.song_start_s || 0) - sequencer.total)} before the song</span></div>` : ''}
      <div class="row"><span class="muted grow">Versions</span><b>${doc.deliverables.length}</b></div>
      ${doc.brief.song ? `<label class="row" title="Where the edit starts in the song"><span class="muted grow">Song starts at</span><input class="input" id="song-start" type="number" min="0" step="0.1" value="${doc.edit.song_start_s || 0}" style="width:70px"><span class="muted">s</span></label>` : ''}
      <div class="row"><span class="muted grow">Tracks over the edit</span><b>${doc.edit.audio_tracks.length}</b></div></div>`;
    card.querySelector('#song-start')?.addEventListener('change', event => actions.ops([{ op: 'edit', fields: { song_start_s: Math.max(0, +event.target.value || 0) } }]));
  }

  function selectTrack(index) {
    selectedTrack = index;
    if (index != null) { selectedText = null; renderText(); }
    renderTracks();
    renderEditCard();
    $('.tl').classList.toggle('sound-open', index != null);  // a narrow window shows the column beside the monitor for the track's card only
    const card = $('#edit-card'), side = card.parentElement;
    if (index != null) side.scrollTop += card.getBoundingClientRect().top - side.getBoundingClientRect().top;  // the track's card is below Now playing: in view even beside a short monitor
  }

  function removeTrack(index) {
    selectTrack(null);
    return actions.ops([{ op: 'edit', fields: { audio_tracks: state.doc.edit.audio_tracks.filter((_, number) => number !== index) } }]);
  }

  function positionPlayhead(time) {
    const head = $('#playhead');
    if (!head) return;
    head.style.left = `${x(time)}px`;
    if (sequencer?.playing) {
      const left = x(time) - tracks.scrollLeft;
      if (left > tracks.clientWidth - 80 || left < 96) tracks.scrollLeft = x(time) - 200;
    }
  }

  function togglePlay() {
    sequencer.toggle();
    $('#play').innerHTML = icon(sequencer.playing ? 'pause' : 'play');
  }

  function zoom(factor, anchorSeconds = sequencer.time) {
    const before = x(anchorSeconds) - tracks.scrollLeft;
    pxPerSecond = Math.max(1.5, Math.min(400, pxPerSecond * factor));
    render();
    tracks.scrollLeft = x(anchorSeconds) - before;
  }

  // Snapping: beats and bars of the song, and the edges of the other shots.
  function snapTo(seconds, ignore) {
    if (!snap) return { value: seconds, guide: null };
    const song = analysis();
    const offset = +(state.doc.edit.song_start_s || 0);
    const points = [...(song?.beats || []).map(value => value - offset), ...segs.filter(seg => seg.shot.id !== ignore).flatMap(seg => [seg.start, seg.start + seg.duration])];
    const threshold = 8 / pxPerSecond;
    let best = null;
    for (const point of points) if (Math.abs(point - seconds) < threshold && (best === null || Math.abs(point - seconds) < Math.abs(best - seconds))) best = point;
    return best === null ? { value: seconds, guide: null } : { value: best, guide: best };
  }

  function showGuide(seconds) {
    const guide = $('#guide');
    guide.classList.toggle('hidden', seconds == null);
    if (seconds != null) guide.style.left = `${x(seconds)}px`;
  }

  // Interactions --------------------------------------------------------------------------------------------------------------
  inner.addEventListener('pointerdown', event => {
    const handle = event.target.closest('.handle');
    const clip = event.target.closest('.clip');
    const pin = event.target.closest('.transition-pin');
    if (pin) return;
    const trackClip = event.target.closest('.track-clip');
    if (trackClip) {
      const index = +trackClip.dataset.track, track = state.doc.edit.audio_tracks[index];
      drag = { kind: 'track', index, edge: handle?.dataset.edge || null, element: trackClip, startX: event.clientX, start: +track.start_s || 0, skip: +track.in_s || 0, length: trackLength(track), file: trackLengths.get(track.path) ?? Infinity, moved: false };
      inner.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    const textClip = event.target.closest('.text-clip');
    if (textClip) {
      const item = overlays().find(entry => entry.id === textClip.dataset.text);
      drag = { kind: 'text', edge: handle?.dataset.edge || null, item, element: textClip, startX: event.clientX, moved: false };
      inner.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (handle && clip) {
      const seg = segs.find(item => item.shot.id === clip.dataset.clip);
      drag = { kind: 'trim', edge: handle.dataset.edge, seg, clip, startX: event.clientX, original: { start: seg.start, duration: seg.duration, srcIn: seg.srcIn } };
      handle.classList.add('on');
      inner.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (clip) {
      drag = { kind: 'move', id: clip.dataset.clip, clip, startX: event.clientX, moved: false, left: parseFloat(clip.style.left) };
      inner.setPointerCapture(event.pointerId);
      return;
    }
    if (event.target.closest('.ruler, .sections-lane, .audio-lane, .video-lane, .words-lane')) {
      sequencer.seek(secondsAt(event.clientX));
      drag = { kind: 'scrub' };
      inner.setPointerCapture(event.pointerId);
    }
  });

  inner.addEventListener('pointermove', event => {
    if (!drag) return;
    if (drag.kind === 'scrub') { sequencer.seek(secondsAt(event.clientX)); return; }
    const delta = (event.clientX - drag.startX) / pxPerSecond;
    if (drag.kind === 'track') {
      if (Math.abs(event.clientX - drag.startX) > 3) drag.moved = true;
      if (!drag.moved) return;
      const { start, skip, length, file, edge } = drag;
      if (edge === 'l') {  // the track's start moves with the file under it: what it plays there stays in place
        const shift = Math.max(-skip, -start, Math.min(snapTo(start + delta).value - start, length - 0.1));
        drag.to = { start_s: +(start + shift).toFixed(2), in_s: +(skip + shift).toFixed(3) };
        drag.element.style.left = `${x(start + shift)}px`;
        drag.element.style.width = `${Math.max(18, (length - shift) * pxPerSecond - 2)}px`;
      } else if (edge === 'r') {  // no further than its file
        const played = Math.max(0.1, Math.min(snapTo(start + length + delta).value - start, file - skip));
        drag.to = { out_s: +(skip + played).toFixed(3) };
        drag.element.style.width = `${Math.max(18, played * pxPerSecond - 2)}px`;
      } else {
        drag.to = { start_s: +snapTo(Math.max(0, start + delta)).value.toFixed(2) };
        drag.element.style.left = `${x(drag.to.start_s)}px`;
      }
      return;
    }
    if (drag.kind === 'text') {
      if (Math.abs(event.clientX - drag.startX) > 3) drag.moved = true;
      if (!drag.moved) return;
      const { item, edge } = drag;
      let start = item.start_s, end = item.end_s;
      if (edge === 'l') start = Math.min(end - 0.5, snapTo(Math.max(0, item.start_s + delta)).value);
      else if (edge === 'r') end = Math.max(start + 0.5, snapTo(item.end_s + delta).value);
      else { const length = end - start; start = snapTo(Math.max(0, item.start_s + delta)).value; end = start + length; }
      drag.times = [+start.toFixed(2), +end.toFixed(2)];
      drag.element.style.left = `${x(start)}px`;
      drag.element.style.width = `${Math.max(18, (end - start) * pxPerSecond - 2)}px`;
      return;
    }
    if (drag.kind === 'move') {
      if (Math.abs(event.clientX - drag.startX) > 4) drag.moved = true;
      if (!drag.moved) return;
      drag.clip.classList.add('dragging');
      drag.clip.style.left = `${drag.left + event.clientX - drag.startX}px`;
      return;
    }
    const { seg, original, edge } = drag;
    const take = seg.take;
    const length = take?.duration_s || null;
    if (edge === 'r') {
      let end = original.start + original.duration + delta;
      const snapped = snapTo(end, seg.shot.id);
      end = snapped.value;
      showGuide(snapped.guide);
      let duration = Math.max(0.3, end - original.start);
      if (take && length) duration = Math.min(duration, (length - original.srcIn) / seg.speed);
      drag.duration = duration;
      drag.clip.style.width = `${Math.max(14, duration * pxPerSecond - 2)}px`;
      drag.label = `${duration.toFixed(2)} s`;
    } else {
      let start = original.start + delta;
      const snapped = snapTo(start, seg.shot.id);
      start = snapped.value;
      showGuide(snapped.guide);
      let shift = start - original.start;
      if (take) shift = Math.max(-original.srcIn / seg.speed, Math.min(shift, original.duration - 0.3));
      else shift = Math.min(shift, original.duration - 0.3);
      drag.shift = shift;
      drag.clip.style.left = `${x(original.start + shift)}px`;
      drag.clip.style.width = `${Math.max(14, (original.duration - shift) * pxPerSecond - 2)}px`;
    }
  });

  inner.addEventListener('pointerup', async event => {
    const current = drag;
    drag = null;
    showGuide(null);
    container.querySelectorAll('.handle.on').forEach(element => element.classList.remove('on'));
    if (!current) return;
    if (current.kind === 'track') {
      if (!current.moved) { selectTrack(current.index); sequencer.seek(current.start + 0.01); return; }  // a click opens the track beside the monitor
      const tracks = state.doc.edit.audio_tracks.map((track, index) => index === current.index ? { ...track, ...current.to } : track);
      await actions.ops([{ op: 'edit', fields: { audio_tracks: tracks } }]);
      return;
    }
    if (current.kind === 'text') {
      if (!current.moved) {
        if (doubleClick(`text:${current.item.id}`) && current.item.style) { editText(current.item); return; }
        selectedText = current.item.id;
        if (selectedTrack != null) selectTrack(null);
        sequencer.seek(current.item.start_s + 0.01);
        renderText();
        return;
      }
      if (current.times) await actions.ops([{ op: 'update_overlay', overlay: current.item.id, fields: { start_s: current.times[0], end_s: current.times[1] } }]);
      return;
    }
    if (current.kind === 'move') {
      current.clip.classList.remove('dragging');
      if (!current.moved) {
        const seg = segs.find(item => item.shot.id === current.id);
        if (doubleClick(`shot:${current.id}`)) { actions.openShot(current.id); return; }  // double-click: open the shot
        actions.select(current.id, { additive: event.ctrlKey || event.metaKey, range: event.shiftKey, open: false });
        if (seg && !sequencer.playing) sequencer.seek(seg.start + 0.01);  // a click shows the shot in the monitor
        renderClips();
        return;
      }
      const seconds = secondsAt(event.clientX);
      const others = segs.filter(seg => seg.shot.id !== current.id);
      const before = others.find(seg => seconds < seg.start + seg.duration / 2)?.shot.id ?? null;
      await actions.ops([{ op: 'move_shots', shots: [current.id], before }]);
      return;
    }
    if (current.kind === 'trim') {
      const { seg } = current;
      const take = seg.take;
      if (current.edge === 'r' && current.duration != null) {
        await actions.ops([take ? { op: 'update_shot', shot: seg.shot.id, fields: { trim: [seg.srcIn, +(seg.srcIn + current.duration * seg.speed).toFixed(3)] } } : { op: 'update_shot', shot: seg.shot.id, fields: { duration_s: +(current.duration * seg.speed).toFixed(2) } }]);
      } else if (current.edge === 'l' && current.shift != null) {
        await actions.ops([take ? { op: 'update_shot', shot: seg.shot.id, fields: { trim: [+(seg.srcIn + current.shift * seg.speed).toFixed(3), seg.shot.trim?.[1] ?? null] } } : { op: 'update_shot', shot: seg.shot.id, fields: { duration_s: +((seg.duration - current.shift) * seg.speed).toFixed(2) } }]);
      } else render();
    }
  });

  inner.addEventListener('click', event => {
    const pin = event.target.closest('.transition-pin');
    if (!pin) return;
    const shot = actions.shot(pin.dataset.pin);
    const current = shot.transition?.type || 'cut';
    const set = (type, duration) => actions.ops([{ op: 'update_shot', shot: shot.id, fields: { transition: { type, duration_s: duration ?? (shot.transition?.duration_s || 0.5) } } }]);
    menu(pin, [...TRANSITION_ORDER.map(type => ({ label: `${TRANSITION_LABEL[type]}${type === current ? ' ✓' : ''}`, icon: type === 'cut' ? 'scissors' : 'transition', action: () => set(type) })),
      ...(current !== 'cut' ? ['-', ...[0.25, 0.5, 1, 2].map(duration => ({ label: `${duration} s${Math.abs((shot.transition?.duration_s || 0.5) - duration) < 0.01 ? ' ✓' : ''}`, icon: 'clock', action: () => set(current, duration) }))] : [])]);
  });

  inner.addEventListener('contextmenu', event => {
    const textClip = event.target.closest('.text-clip');
    if (textClip) {
      event.preventDefault();
      const item = overlays().find(entry => entry.id === textClip.dataset.text);
      menu({ x: event.clientX, y: event.clientY }, [
        ...(item.style ? [{ label: 'Edit the text…', icon: 'note', action: () => editText(item) }] : []),
        { label: 'Delete', icon: 'trash', danger: true, action: () => removeText(item) },
      ]);
      return;
    }
    const clip = event.target.closest('.clip');
    if (!clip) return;
    event.preventDefault();
    actions.select(clip.dataset.clip, { open: false });
    shotMenu({ x: event.clientX, y: event.clientY }, [clip.dataset.clip]);
  });


  tracks.addEventListener('wheel', event => {
    if (!(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    zoom(event.deltaY < 0 ? 1.15 : 1 / 1.15, secondsAt(event.clientX));
  }, { passive: false });

  let width = tracks.clientWidth;
  const resize = () => {  // a hidden tab has no width, and the tab shown again has its old one: nothing to redraw then
    if (drag || !tracks.clientWidth || tracks.clientWidth === width) return;
    width = tracks.clientWidth;
    fit(); render();
  };
  window.addEventListener('resize', resize);

  render();
  if (state.doc.brief.song && !analysis()) api.analysis(state.pid).catch(() => {});
  return {
    update() { if (!drag) render(); },
    selection() { renderClips(); },
    progress() {},
    theme() { render(); },
    suspend() { if (sequencer?.playing) togglePlay(); },
    key(event) {
      const text = selectedText && overlays().find(item => item.id === selectedText);
      if (text && (event.key === 'Delete' || event.key === 'Backspace')) { event.preventDefault(); removeText(text); return; }
      if (text && event.key === 'Enter' && text.style) { event.preventDefault(); editText(text); return; }
      if (selectedTrack != null && (event.key === 'Delete' || event.key === 'Backspace')) { event.preventDefault(); removeTrack(selectedTrack); return; }
      if (event.key === 'Escape') { selectedText = null; renderText(); if (selectedTrack != null) selectTrack(null); }
      if (event.key === ' ') { event.preventDefault(); togglePlay(); }
      else if (event.key === 'Home') sequencer.seek(0);
      else if (event.key === 'End') sequencer.seek(sequencer.total);
      else if (event.key === 'ArrowRight') sequencer.seek(sequencer.time + (event.shiftKey ? 1 : 1 / 24));
      else if (event.key === 'ArrowLeft') sequencer.seek(sequencer.time - (event.shiftKey ? 1 : 1 / 24));
      else if (event.key === '+' || event.key === '=') zoom(1.4);
      else if (event.key === '-') zoom(1 / 1.4);
    },
    destroy() { sequencer?.destroy(); window.removeEventListener('resize', resize); },
  };
}
