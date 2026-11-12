// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Showrunner page: state, routing, the shell (top bar, view rail, side panel), live events and global shortcuts.
import { api, listen } from './api.js';
import { icon, esc, h, toast, debounce, throttle, aspectValue, kindOfFile, closeMenu, confirmDialog, formatDuration, count, $, adviceBar, bindAdvice } from './ui.js';
import { renderHome } from './home.js';
import { openDefaultModels } from './models.js';
import { openSettings } from './settings.js';
import { mountBrief } from './brief.js';
import { mountStoryboard, toShoot, visible as onBoard } from './storyboard.js';
import { mountTimeline } from './timeline.js';
import { mountCast } from './cast.js';
import { mountDeliver } from './deliver.js';
import { mountInspector } from './inspector.js';
import { mountDirector } from './director.js';
import { mountConversation } from './conversation.js';
import { mountBin } from './bin.js';

const params = new URLSearchParams(location.search);
export const state = {
  embedded: params.has('embedded'), catalog: [], directorInfo: { available: false }, kinds: [], aspects: [],
  pid: null, doc: null, view: 'storyboard', side: 'inspector', sideOpen: dockOpen(), deepyTab: false,
  selection: new Set(), focus: null, live: { text: '', tools: [], status: null, progress: null, running: false, task: null },
  renders: {}, filter: 'all', shootTier: localStorage.getItem('sr.tier') === 'final' ? 'final' : 'draft', cardWidth: +(localStorage.getItem('sr.card') || 250), binOpen: localStorage.getItem('sr.bin') !== '0',
};
const VIEWS = { brief: ['Brief', 'brief', mountBrief], storyboard: ['Board', 'grid', mountStoryboard], timeline: ['Timeline', 'timeline', mountTimeline], cast: ['Cast', 'users', mountCast], deliver: ['Deliver', 'download', mountDeliver] };
const UNDONE = { add_take: 'a new take', update_shot: 'a shot edit', verdict: 'a verdict', add_shots: 'new shots', replace_plan: 'the plan', remove_shot: 'a deletion',
  move_shots: 'a move', reorder: 'a move', remove_take: 'a removed take', brief: 'a brief edit', add_cast: 'a new cast member', update_cast: 'a cast edit',
  remove_cast: 'a cast removal', edit: 'a sound edit', add_overlay: 'a text', update_overlay: 'a text edit', remove_overlay: 'a text deletion', style: 'a style edit',
  outline: 'the outline', deliverable: 'an export', bin_add: 'a bin import', bin_remove: 'a bin removal', bin_to_take: 'a take from the bin', bin_to_track: 'a sound from the bin',stage: 'a stage change' };
const GUIDE_PAGES = { home: 'Quick Start', brief: 'Starting a Production', storyboard: 'Reviewing and Changing the Plan', timeline: 'The Timeline', cast: 'The Cast', deliver: 'Delivering' };
const STAGES = [['brief', 'Brief'], ['planning', 'Plan'], ['review', 'Review'], ['shooting', 'Shoot'], ['editing', 'Edit'], ['delivered', 'Deliver']];
// What each stage's button opens: its view, on the Board the shots of that step, and its tooltip
const STAGE_VIEWS = { brief: ['brief', null, 'opens the Brief'], planning: ['storyboard', 'all', 'shows all the shots on the Board'], review: ['storyboard', 'review', 'shows the takes to review on the Board'],
  shooting: ['storyboard', 'todo', 'shows the shots still to shoot on the Board'], editing: ['timeline', null, 'opens the Timeline'], delivered: ['deliver', null, 'opens Deliver'] };
const openStage = () => STAGES.find(([key]) => STAGE_VIEWS[key][0] === state.view && (!STAGE_VIEWS[key][1] || STAGE_VIEWS[key][1] === state.filter))?.[0];
const app = document.getElementById('app');
let stopEvents = null, mounted = null, inspector = null, director = null, conversation = null, bin = null, catalogStop = null;

// Theme: follow the WanGP page when embedded, else the system; the toggle overrides it.
function applyTheme() {
  let theme = localStorage.getItem('sr.theme');
  if (!theme && state.embedded) {
    try { theme = window.parent.document.body.classList.contains('dark') || window.parent.document.documentElement.classList.contains('dark') ? 'dark' : 'light'; } catch { theme = null; }
  }
  theme = theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.dataset.theme = theme;
}
applyTheme();
window.addEventListener('focus', () => { applyTheme(); mounted?.refreshSingerVoice?.(); });

export const actions = {
  async ops(ops, { quiet = false } = {}) {
    if (!state.pid) return null;
    try {
      const result = await api.ops(state.pid, ops);
      setDoc(result.doc);
      return result.results;
    } catch (error) {
      if (!quiet) toast(error.message, 'error', 6000);
      if (error.status === 409) refreshDoc();
      throw error;
    }
  },
  async makeFormat(aspect) {  // the shots are made again in the new format; the versions so far stay as they are
    const count = state.doc.shots.filter(shot => shot.included).length;
    if (!(await confirmDialog({ title: `Make a ${aspect} version?`, text: `The showrunner makes the ${count} shots again in ${aspect}, with the same words, cast and music, and the cards in the same design. This takes about as long as the first shoot. The versions you have stay as they are.`, ok: 'Make it' }))) return;
    await actions.ops([{ op: 'brief', fields: { aspect } }]);
    actions.direct('message', `Make a ${aspect} version of the current cut: the same shots, words, cast, narrator and music, every shot made again in ${aspect}, and the cards in the same colours, typefaces and layout at the new format. The current version stays as it is.`);
  },
  refreshTopbar() { renderTopbar(); },
  refreshDirector() { director?.update(state.doc); renderTopbar(); },  // what it says about the user's turn changed without a new document
  select(shotId, { additive = false, range = false, open = true, keep = false } = {}) {
    const ids = mounted?.visibleIds?.() || state.doc?.shots.map(shot => shot.id) || [];  // a range covers the shots on screen only
    if (range && state.focus) {
      const [a, b] = [ids.indexOf(state.focus), ids.indexOf(shotId)].sort((x, y) => x - y);
      ids.slice(a, b + 1).forEach(id => state.selection.add(id));
    } else if (keep) {
      // the selection was already filtered: keep it as it is
    } else if (additive) {
      state.selection.has(shotId) ? state.selection.delete(shotId) : state.selection.add(shotId);
    } else {
      state.selection = new Set(shotId ? [shotId] : []);
    }
    state.focus = shotId;
    if (open && shotId) state.side = 'inspector';  // a folded panel stays folded: a double-click on the shot or its Shot tab opens it
    renderSide();
    mounted?.selection?.();
  },
  async direct(task, message = '', shots = [], { show = false, newVersion = true } = {}) {  // show: the user just asked for it and wants to see it taken up, even with a shot open
    try {
      const status = await api.direct(state.pid, task, message, shots, newVersion);
      state.directorBaseline = structuredClone(state.doc);
      state.live = { text: '', tools: [], status: null, progress: null, running: true, task, summary: null };
      if (show || !(state.side === 'inspector' && state.selection.size)) state.side = 'director';  // a shot being looked at stays in view
      if (show) state.sideOpen = true;
      renderSide();
      renderTopbar();
      return status;
    } catch (error) { toast(error.message, 'error', 7000); }
  },
  async steer(text) {
    try { await api.steer(state.pid, text); } catch (error) { toast(error.message, 'error', 6000); }
  },
  async stop() { try { await api.stop(state.pid); toast('Stopping the showrunner…'); } catch (error) { toast(error.message, 'error'); } },
  async pause(now = false) {
    try { await api.pause(state.pid, now); toast(now ? 'Pausing the showrunner now…' : 'The showrunner pauses after its current step…'); refreshDoc(); } catch (error) { toast(error.message, 'error'); }
  },
  async resume(message = '') {
    try {
      await api.resume(state.pid, message);
      state.directorBaseline = structuredClone(state.doc);
      state.live = { text: '', tools: [], status: null, progress: null, running: true, task: state.doc.director.task, summary: null };
      renderSide();
      renderTopbar();
    } catch (error) { toast(error.message, 'error', 7000); }
  },
  busy: doc => !['idle', 'paused'].includes(doc.director.state),  // a paused director waits for the user: nothing runs
  // Why the buttons that give the showrunner a task are greyed out now ('': they are not).
  directorBlocked() {
    return actions.renderBlocked() || state.directorInfo.available === false && state.directorInfo.reason || '';
  },
  // Why Render now cannot run now ('': it can): it needs no showrunner, but does not run beside one.
  renderBlocked: () => state.live.running || actions.busy(state.doc) ? 'The showrunner is working: this comes back when it finishes, or after Pause or Stop.' : '',
  // Why the edit cannot be assembled now ('': it can).
  assembleBlocked: doc => doc.shots.some(shot => shot.included && shot.takes.length) ? '' : 'Nothing to assemble yet: shoot at least one shot, the edit is made of their takes.',
  planBlocked: doc => ['text', 'script', 'song'].some(key => String(doc.brief[key] || '').trim()) ? '' : 'Write your idea in the brief first, or drop a song or a script: the showrunner plans from it.',
  async upload(files, role = 'bin', target = null) {
    const list = [...files].filter(file => kindOfFile(file.name));
    if (!list.length) { toast('Drop images, videos or audio files.', 'error'); return []; }
    const results = [];
    for (const file of list) {
      toast(`Importing ${file.name}…`, 'info', 1800);
      try { results.push(await api.upload(state.pid, file, role, target)); } catch (error) { toast(`${file.name}: ${error.message}`, 'error', 6000); }
    }
    refreshDoc();
    return results;
  },
  async detachSound(shotId) {
    try {
      await api.detachSound(state.pid, shotId);
      refreshDoc();
      toast('Its sound is now a track on the Sound lane of the Timeline.', 'ok');
    } catch (error) { toast(error.message, 'error', 6000); }
  },
  reattachSound(shotId) {
    return actions.ops([{ op: 'edit', fields: { audio_tracks: state.doc.edit.audio_tracks.filter(track => track.shot !== shotId) } }, { op: 'update_shot', shot: shotId, fields: { sound: true } }]);
  },
  pausePlayback() {  // the timeline's preview, and every player on the page
    mounted?.suspend?.();
    document.querySelectorAll('video, audio').forEach(media => media.pause());
  },
  file: (path, download = false) => api.file(state.pid, path, download),
  sourcePart: shot => api.sourcePart(state.pid, shot.id),  // what a remake's shot delivers, as a video of its own
  takePart: (shot, take) => api.takePart(state.pid, shot.id, take.id),
  sourceFrame: (shot, frame) => api.sourceFrame(state.pid, shot.id, frame),  // one frame of a remake's source, to paint elsewhere
  templateRoles(refresh = false) {  // the template picker's roles, once per production unless a pick changed them
    if (refresh || !state.templateRoles || state.templateRoles.pid !== state.pid) {
      state.templateRoles = { pid: state.pid, promise: api.templates(state.pid).then(result => result.roles) };
      state.templateRoles.promise.catch(() => { state.templateRoles = null; });
    }
    return state.templateRoles.promise;
  },
  go(view) { location.hash = `#/p/${state.pid}/${view}`; },
  shot: id => state.doc?.shots.find(shot => shot.id === id),
  take(shot) { return shot?.takes.find(take => take.id === shot.chosen_take) || shot?.takes.at(-1) || null; },
  async undo() {
    if (!state.pid) return;
    const result = await api.undo(state.pid);
    if (result.revision == null) { toast('Nothing to undo.'); return; }
    setDoc(result.doc);
    const what = (result.undone?.ops || []).map(op => UNDONE[op] || 'a change');
    toast(`Undid ${[...new Set(what)].join(', ') || 'the last change'}${result.undone?.by === 'deepy' ? ' (by the showrunner)' : ''}. Redo: Ctrl+Shift+Z`);
  },
  async redo() { if (!state.pid) return; const result = await api.redo(state.pid); if (result.revision == null) toast('Nothing to redo.'); else setDoc(result.doc); },
  rerender: () => route(),
  tierChanged() { mounted?.update?.(state.doc); inspector?.update?.(state.doc); },
  // Assemble the edit from the page: warns about shots left out, reports once, and says when the video is ready.
  async assemble() {
    const doc = state.doc;
    const missing = doc.shots.filter(shot => shot.included && !shot.takes.length);
    if (!doc.shots.some(shot => shot.included && shot.takes.length)) { toast('Shoot at least one shot first: the edit is made of the shots\' takes.', 'error'); return null; }
    if (missing.length && !(await confirmDialog({ title: `${missing.length} ${missing.length === 1 ? 'shot has' : 'shots have'} no take yet`,
      text: `${missing.map(shot => `${doc.shots.indexOf(shot) + 1}. ${shot.title || 'Untitled'}`).slice(0, 6).join(', ')}${missing.length > 6 ? '…' : ''} will be left out of this version.`, ok: 'Assemble anyway' }))) return null;
    state.userAssembling = true;
    try {
      const result = await api.assemble(state.pid);
      toast(`Your video is ready: ${formatDuration(result.duration_s)}.`, 'ok', 5000);
      return result;
    } catch (error) { toast(error.message, 'error', 8000); return null; } finally { state.userAssembling = false; }
  },
  dock(open) {
    state.sideOpen = open;
    try { localStorage.setItem('sr.dock', open ? '1' : '0'); } catch { /* a convenience */ }
    renderSide();
  },
  toggleSide(panel) {
    if (state.sideOpen && state.side === panel) { actions.dock(false); return; }
    state.side = panel;
    actions.dock(true);
  },
  openShot(id) {
    actions.select(id);
    actions.dock(true);
  },
};

function setDoc(doc) {
  if (!doc || doc.id !== state.pid) return;
  if (state.doc && doc.revision < state.doc.revision) return;
  if (state.doc && doc.revision === state.doc.revision && JSON.stringify(doc) === JSON.stringify(state.doc)) return;  // a catch-up that found nothing new redraws nothing (no flash when the tab comes back)
  const first = !state.doc;
  state.doc = doc;
  document.documentElement.style.setProperty('--aspect', aspectValue(doc.brief.aspect));
  const [aspectWidth, aspectHeight] = aspectValue(doc.brief.aspect).split('/').map(Number);
  document.documentElement.style.setProperty('--aspect-num', String(aspectWidth / aspectHeight));  // for sizes computed in CSS
  state.selection = new Set([...state.selection].filter(id => doc.shots.some(shot => shot.id === id)));
  if (first) return;
  renderTopbar();
  if (document.getElementById('rail')) renderRail();  // the board's review count follows every change
  mounted?.update?.(doc);
  inspector?.update?.(doc);
  director?.update?.(doc);
  bin?.update?.(doc);
}

const refreshDoc = debounce(async () => {
  if (!state.pid) return;
  try { setDoc(await api.get(state.pid)); } catch (error) { if (error.status === 404) location.hash = '#/'; }
}, 120);

// The showrunner streams its words and steps many times a second, and a render its progress: the page shows them at most ten times a second.
const drawLive = throttle(() => { director?.live?.(); renderTopbar(); });
const drawProgress = throttle(() => { mounted?.progress?.(); renderTopbar(); });  // the pill says how far the user's renders are

function eventsFor(pid) {
  stopEvents?.();
  stopEvents = listen(pid, {
    reconnected: catchUp,
    doc: event => { if (!state.doc || event.revision > state.doc.revision) refreshDoc(); },
    director: event => {
      if (event.event === 'chat') { conversation?.live(event.data); return; }
      const live = state.live;
      if (event.event === 'started' || event.event === 'queued') {
        live.running = true; live.queued = event.event === 'queued';
        state.directorBaseline ??= structuredClone(state.doc);  // what the showrunner's task started from: its changes are measured against it
      }
      if (event.event === 'text') live.text = event.data.text;
      if (event.event === 'status') live.status = event.data;
      if (event.event === 'stats') live.stats = event.data;
      if (event.event === 'tool') {
        const index = live.tools.findIndex(tool => tool.id === event.data.id);
        index < 0 ? live.tools.push(event.data) : live.tools.splice(index, 1, event.data);
        if (!live.tools.some(tool => tool.status === 'running')) live.progress = null;  // a finished step's progress line goes with it
      }
      if (event.event === 'progress') live.progress = event.data;
      if (['queued', 'stats', 'status'].includes(event.event) && deepyWait(live)) conversation?.live({ type: 'status', status: { visible: true, kind: 'loading', text: deepyWait(live), progress: live.status?.progress } });
      if (['done', 'cancelled', 'error', 'paused'].includes(event.event)) {
        live.running = false; live.progress = null;
        if (event.event === 'error') toast(`Showrunner: ${event.data?.error || 'error'}`, 'error', 8000);
        if (event.event === 'paused') toast('The showrunner paused and the GPU is free. Resume continues its task where it stopped.', 'ok', 8000);
        reportChanges(event.event);
        conversation?.finished();
      }
      drawLive();
    },
    render: event => {
      if (event.done) delete state.renders[event.shot];
      else state.renders[event.shot] = event.progress ?? 0;
      drawProgress();
    },
    song_voice: event => {  // the song sung in another voice, changed from the Brief
      state.songVoice = event.done ? null : { progress: event.progress, status: event.status };
      if (event.error) toast(`The singer's voice could not be changed: ${event.error}`, 'error', 8000);
      else if (event.done) toast('The song is sung in its new voice. The former version is in the bin.', 'ok');
      mounted?.songVoice?.();
    },
    assemble: event => {
      if (!state.userAssembling) {  // an assembly the user started reports on its own button
        if (event.state === 'done') toast('The edit is assembled: watch it in Deliver.', 'ok');
        if (event.state === 'error') toast(`Assembly failed: ${event.error}`, 'error', 8000);
      }
      mounted?.assembling?.(event.state);
    },
  });
}

// After events were missed (the page was hidden or the connection dropped): the production, its renders and the showrunner's state now.
async function catchUp() {
  state.renders = {};  // a render still running reports again with its next progress
  mounted?.refreshSingerVoice?.();  // the SeedVC choice may have changed while Configuration was open
  refreshDoc();
  let changed = true;
  try {
    const status = await api.director(state.pid);
    changed = status.running !== state.live.running;
    if (status.live) {  // steps that ended unseen
      changed ||= Object.keys(status.live).some(key => JSON.stringify(status.live[key]) !== JSON.stringify(state.live[key]));
      Object.assign(state.live, status.live);
    }
    if (state.live.running && !status.running) { state.live.running = false; state.live.progress = null; reportChanges('caught_up'); }
    else state.live.running = status.running;
  } catch { /* the page works without it */ }
  mounted?.progress?.();
  if (!changed) return;  // the showrunner's state is unchanged: its panel and the top bar stay as they are (no flash when the tab comes back)
  director?.live?.();
  conversation?.catchUp();
  renderTopbar();
}

// What the director changed during its task, in plain words, with the shots it touched highlighted on the board for a moment.
async function reportChanges(outcome) {
  const before = state.directorBaseline;
  state.directorBaseline = null;
  let doc;
  try { doc = await api.get(state.pid); setDoc(doc); } catch { return; }
  if (!before) { if (outcome === 'done') toast('The showrunner finished.', 'ok'); return; }
  const number = id => doc.shots.findIndex(shot => shot.id === id) + 1;
  const old = new Map(before.shots.map(shot => [shot.id, shot]));
  const fields = ['title', 'section', 'intent', 'camera', 'cast', 'dialogue', 'duration_s', 'generation', 'transition', 'trim', 'speed', 'sound', 'included'];
  const added = doc.shots.filter(shot => !old.has(shot.id));
  const removed = before.shots.filter(shot => !doc.shots.some(item => item.id === shot.id));
  const edited = doc.shots.filter(shot => old.has(shot.id) && fields.some(key => JSON.stringify(shot[key]) !== JSON.stringify(old.get(shot.id)[key])));
  const takes = doc.shots.flatMap(shot => shot.takes.filter(take => !(old.get(shot.id)?.takes || []).some(item => item.id === take.id)).map(take => [shot, take]));
  const judged = doc.shots.flatMap(shot => shot.takes.filter(take => take.verdict?.by === 'deepy' && JSON.stringify(take.verdict) !== JSON.stringify((old.get(shot.id)?.takes || []).find(item => item.id === take.id)?.verdict)));
  const list = shots => shots.slice(0, 6).map(shot => number(shot.id)).join(', ') + (shots.length > 6 ? '…' : '');
  const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;
  const parts = [
    added.length && `added ${plural(added.length, 'shot')} (${list(added)})`,
    removed.length && `removed ${plural(removed.length, 'shot')}`,
    edited.length && `changed ${plural(edited.length, 'shot')} (${list(edited)})`,
    takes.length && `shot ${plural(takes.length, 'take')}`,
    judged.length && `judged ${judged.length}: ${judged.filter(take => take.verdict.ok).length} good, ${judged.filter(take => !take.verdict.ok).length} to redo`,
    doc.cast.length > before.cast.length && `added ${plural(doc.cast.length - before.cast.length, 'cast member')}`,
    JSON.stringify(doc.cast.slice(0, before.cast.length)) !== JSON.stringify(before.cast) && 'updated the cast',
    doc.edit.overlays.length !== before.edit.overlays.length && `${doc.edit.overlays.length > before.edit.overlays.length ? 'added' : 'removed'} text`,
    JSON.stringify(doc.edit.audio_tracks) !== JSON.stringify(before.edit.audio_tracks) && 'set the sound tracks',
    JSON.stringify(doc.edit.subtitles) !== JSON.stringify(before.edit.subtitles) && 'set the subtitles',
    doc.deliverables.length > before.deliverables.length && 'assembled a new version',
  ].filter(Boolean);
  state.live.summary = parts.length ? `The showrunner ${parts.join(', ')}.` : doc.revision === before.revision ? 'The showrunner changed nothing in the production.' : '';
  director?.live?.();
  renderTopbar();
  if ((outcome === 'done' || outcome === 'caught_up') && state.live.summary) toast(state.live.summary, 'ok', 9000);
  state.flash = new Set([...added, ...edited, ...takes.map(([shot]) => shot)].map(shot => shot.id));
  mounted?.update?.(doc);
  setTimeout(() => { state.flash = null; mounted?.update?.(state.doc); }, 6000);
}

// Shell ---------------------------------------------------------------------------------------------------------------------------
function renderShell() {
  inspector = director = conversation = bin = null;  // their DOM is rebuilt below
  app.innerHTML = `
    <header class="topbar" id="topbar"></header>
    <div class="workspace">
      <nav class="rail" id="rail"></nav>
      <section class="main"><div class="view" id="view"></div><div id="advice"></div><div id="next-step"></div><div id="bin"></div></section>
      <aside class="side" id="side"></aside>
    </div>
    <div class="drop-veil" id="veil"><div>${icon('upload')} Drop media to add it to the production</div></div>`;
}

// Live events ask for the bars several times a second. They redraw only when they change: a press on a button replaced before its release makes no click.
const drawnBars = new WeakMap();
function draw(element, html) {
  if (drawnBars.get(element) === html) return false;
  drawnBars.set(element, html);
  element.innerHTML = html;
  return true;
}

// The pill's dot calls for a new turn for a few seconds: a later redraw of the bar (another page, a saved edit) must not play it again.
let calledTurn = '', callingUntil = 0;
function renderTopbar() {
  const bar = document.getElementById('topbar');
  if (!bar) return;
  if (!state.pid || !state.doc) {
    if (draw(bar, `${brand()}<div class="grow"></div>${guideButton()}${themeButton()}`)) wangpButton(bar);
    return;
  }
  const doc = state.doc;
  const stageIndex = STAGES.findIndex(([key]) => key === doc.stage);
  const working = actions.busy(doc) || state.live.running, next = nextStep(doc), turn = next ? `${doc.id}|${next.label}|${next.text}` : '', renders = userRenders(doc);
  if (turn !== calledTurn) { calledTurn = turn; callingUntil = turn ? performance.now() + 4800 : 0; }
  const html = `${brand()}
    <span class="sep"></span>
    <input class="input bare crumb-title" id="title" value="${esc(doc.title)}" aria-label="Production title" spellcheck="false">
    <div class="stages">${STAGES.map(([key, label], index) => `${index ? icon('chevron', 'stage-arrow') : ''}<button class="stage ${index < stageIndex ? 'done' : index === stageIndex ? 'current' : ''} ${key === openStage() ? 'open' : ''}" data-stage="${key}" title="${label}: ${STAGE_VIEWS[key][2]}"><span class="dot">${index < stageIndex ? icon('check') : index + 1}</span><span>${label}</span></button>`).join('')}</div>
    <button class="btn ghost icon" id="undo" title="Undo (Ctrl+Z)">${icon('undo')}</button>
    <button class="btn ghost icon" id="redo" title="Redo (Ctrl+Shift+Z)">${icon('redo')}</button>
    <button class="director-pill" id="pill"></button>
    ${guideButton()}${themeButton()}`;
  const tab = document.querySelector('#side [data-panel="director"]');
  tab?.classList.toggle('working', !!working);
  tab?.classList.toggle('turn', !!next);
  renderNextStep();
  // the pill changes in place: a render's progress redraws it every few seconds, where the title being typed and the buttons being pressed would be lost
  const pill = { className: `director-pill ${working || renders ? 'working' : ''} ${next ? 'turn' : ''} ${performance.now() < callingUntil ? 'call' : ''}`, title: next ? `Your turn: ${next.text}` : renders?.title || 'Showrunner',
                 innerHTML: `<span class="avatar">${icon('director')}</span>${next ? `<span class="state">Your turn: ${esc(next.label)}</span>` : `<span>Showrunner</span><span class="state">${esc(directorLabel(doc))}</span>`}` };
  const drawn = draw(bar, html), shown = bar.querySelector('#pill'), key = Object.values(pill).join('|');
  if (shown.dataset.shown !== key) Object.assign(shown, pill).dataset.shown = key;
  if (!drawn) return;
  wangpButton(bar);
  const title = document.getElementById('title');
  title.addEventListener('change', () => title.value.trim() && actions.ops([{ op: 'brief', fields: { title: title.value.trim() } }]));
  title.addEventListener('keydown', event => { if (event.key === 'Enter') title.blur(); });
  bar.querySelector('#undo').onclick = () => actions.undo();
  bar.querySelector('#redo').onclick = () => actions.redo();
  bar.querySelector('#pill').onclick = () => actions.toggleSide('director');
  bar.querySelectorAll('[data-stage]').forEach(button => button.onclick = () => {
    const [view, filter] = STAGE_VIEWS[button.dataset.stage];
    if (filter && state.view === view) return mounted.showFilter(filter);
    if (filter) {
      state.filter = filter;
      state.selection = new Set([...state.selection].filter(id => onBoard(actions.shot(id))));  // a selection never hides behind the filter
    }
    actions.go(view);
  });
}

// The next step in full, above the media bin: its button does it and closes it; when there is nothing to press, an x closes it.
const closedSteps = {};
const STEP_ACTIONS = {
  Continue: () => actions.direct('continue'),
  Shoot: () => actions.direct('shoot'),
  Assemble: async () => { if (await actions.assemble()) actions.go('deliver'); },
  Compare: () => { state.compareLatest = true; actions.go('deliver'); },
};
function renderAdvice() {
  const element = document.getElementById('advice');
  if (!element) return;
  const bar = adviceBar(state.directorInfo);
  if (element.dataset.shown === bar) return;
  element.dataset.shown = bar;
  element.innerHTML = bar;
  bindAdvice(element, state.directorInfo);
}

function renderNextStep() {
  const element = document.getElementById('next-step');
  if (!element) return;
  const next = nextStep(state.doc), key = next ? `${next.label}|${next.text}` : '';
  const action = next?.look && next.view !== state.view ? [`Open ${VIEWS[next.view][0]}`, () => actions.go(next.view)] : next?.action ? [next.action, STEP_ACTIONS[next.action]] : null;
  try { closedSteps[state.pid] ??= localStorage.getItem(`sr.step.${state.pid}`); } catch { /* a convenience */ }
  const shown = next && key !== closedSteps[state.pid] ? `${key}|${action?.[0] || ''}` : '';
  if (element.dataset.shown === shown) return;
  element.dataset.shown = shown;
  element.innerHTML = shown ? `<div class="next-step">${icon('target')}<span class="grow"><b>Your turn.</b> ${esc(next.text)}</span>${action ? `<button class="btn primary small" id="step-action">${esc(action[0])}</button>` : ''}<button class="btn ghost icon" id="step-close" title="Close" aria-label="Close">${icon('x')}</button></div>` : '';
  const close = () => {
    closedSteps[state.pid] = key;
    try { localStorage.setItem(`sr.step.${state.pid}`, key); } catch { /* a convenience */ }
    renderNextStep();
  };
  element.querySelector('#step-close')?.addEventListener('click', close);
  element.querySelector('#step-action')?.addEventListener('click', () => { close(); action[1](); });
}

// What the user does next, once the showrunner's task is over, read from the production: it stays in the top bar until it is done and above
// the media bin until the user closes it, on every page (the toast at the end of a task passes unseen, and not at all while the page is out of sight).
// look: a page to see first; action: what its button does on that page, or at once without one.
export function nextStep(doc) {
  if (!doc || actions.busy(doc) || state.live.running || userRenders(doc)) return null;
  const last = doc.director.last?.status, latest = doc.deliverables.at(-1);
  const takes = doc.shots.filter(shot => shot.included && shot.takes.length).length, todo = toShoot(doc).length;
  const toReview = doc.shots.filter(shot => shot.included && shot.status === 'rendered').length;
  if (last === 'error' || last === 'interrupted') return { label: 'Stopped', text: 'The showrunner stopped before the end of its task: its panel says why, and Continue picks it up.', action: 'Continue' };
  if (!doc.shots.length) return null;
  if (!takes) return { label: 'Review the plan', text: `The plan is ready: check its ${count(doc.shots.length, 'shot')} on the Board, then press Shoot.`, view: 'storyboard', look: true, action: 'Shoot' };
  if (todo) return { label: `${count(todo, 'shot')} to shoot`, text: `${count(todo, 'shot')} still ${todo === 1 ? 'needs' : 'need'} a take: press Shoot.`, view: 'storyboard', action: 'Shoot' };
  if (doc.brief.review_before_edit && toReview) return { label: `Review ${count(toReview, 'take')}`, text: `${count(toReview, 'take')} ${toReview === 1 ? 'waits' : 'wait'} for your review on the Board: approve each one or mark it for a redo.`, view: 'storyboard', look: true };
  if (!latest) return { label: 'Assemble the edit', text: 'Every shot has a take: press Assemble to make the video.', view: 'timeline', action: 'Assemble' };
  let watched = null;
  try { watched = localStorage.getItem(`sr.watched.${doc.id}`); } catch { /* a convenience */ }
  if (latest.edit_key !== doc.edit_key) return { label: 'Assemble', text: `The cut changed since version ${doc.deliverables.length}: press Assemble for a new version.`, view: 'timeline', action: 'Assemble' };
  if (watched !== latest.id && doc.deliverables.length > 1) return { label: 'Compare versions', text: `Version ${doc.deliverables.length} is ready: see what changed since version ${doc.deliverables.length - 1} and compare them side by side.`, view: 'deliver', action: 'Compare' };
  if (watched !== latest.id) return { label: 'Watch the video', text: `Version ${doc.deliverables.length} is ready: watch it on Deliver.`, view: 'deliver', look: true };
  return null;
}

// Before Deepy can answer: waiting behind another production, then reading this production's conversation again (about 0.6 s per 1,000 tokens, measured): nothing else shows for that long.
export function deepyWait(live) {
  if (live.queued) return 'Waiting its turn: Deepy is working on another production';
  if (live.running && /compact/i.test(live.status?.text || '')) return 'Deepy is condensing a long conversation to make room';  // it prefills as part of that too
  if (live.running && live.status?.kind === 'loading' && live.status.progress) return "Reloading this production's conversation into Deepy";  // a tool's status has a progress too (Render: 0/4)
  return live.running && live.status?.kind === 'loading' ? live.status.text : '';
}

// What the showrunner does at this moment, from its current or last step: the whole task's name ("Shooting") stays true for an hour of review, music and editing.
const PHASES = [[/Deep Review/, 'Measuring the version'], [/version \d/, 'Checking the new version'], [/Render/, 'Shooting'], [/Speak|Audio Using TTS|Speech/, 'Recording the lines'], [/Verdict|Inspect/, 'Checking the takes'], [/Transcribe|Check Song|Remove Background/, 'Listening'],
  [/Assemble/, 'Assembling the edit'], [/Song|Score|Music/, 'Composing the music'], [/Showrunner: Edit/, 'Setting the sound and text'], [/Cast|Image/, 'Making the cast pictures'], [/Plan|Add Shots|Update Shot|Move Shots|Template|Validate/, 'Preparing the shots']];
export function activity(live) {
  const tool = live.tools.findLast(item => item.status === 'running') || live.tools.at(-1), label = tool?.label || '';
  if (tool && tool.status !== 'running' && /Assemble/.test(label)) return 'Checking the new version';
  const phase = PHASES.find(([pattern]) => pattern.test(label))?.[1] || '';
  if (/Deep Review/.test(label) && tool.status === 'running' && !/^Running/.test(tool.status_text || 'Running')) {  // its steps take minutes: say which
    const step = /^(?:(\d+)\/(\d+) · )?(.+)$/.exec(tool.status_text);
    return step[1] ? `${step[3]} · ${+step[1] + 1} of ${step[2]}` : step[3];
  }
  const shot = phase === 'Shooting' && tool.status === 'running' ? /Rendering (s\d+) \((\d+)\/(\d+)\)/.exec(tool.status_text || '') : null;
  return shot ? `Shooting ${shot[1]} · ${shot[2]} of ${shot[3]}` : phase;
}

// Renders the user started without the showrunner (Render now, Reshoot): their shots show rendering from the click, the one WanGP works on
// reports its progress, with the neighbours a remake renders together with it.
function userRenders(doc) {
  const shots = doc.shots.filter(shot => shot.status === 'rendering');
  if (!shots.length || actions.busy(doc) || state.live.running) return null;
  const current = shots.find(shot => state.renders[shot.id] != null), now = current ? [current, ...shots.filter(shot => shot.rendering_with === current.id)] : [];
  const later = shots.filter(shot => !now.includes(shot)), progress = current ? Math.round(state.renders[current.id]) : 0, ids = list => list.map(shot => shot.id).join(' + ');
  return now.length ? { label: `Rendering ${ids(now)}${progress ? ` · ${progress}%` : ''}${later.length ? ` · ${later.length} more` : ''}`,
                        title: `Rendering what you asked for, without the showrunner: ${ids(now)} now${progress ? ` (${progress}%)` : ''}${later.length ? `, then ${later.map(shot => shot.id).join(', ')}` : ''}. Each card shows its progress.` }
    : { label: `Rendering ${count(shots.length, 'shot')}…`, title: `${count(shots.length, 'shot')} to render, without the showrunner: ${shots.map(shot => shot.id).join(', ')}. Its progress shows once WanGP works on it.` };
}

function directorLabel(doc) {
  const next = nextStep(doc);
  if (next) return next.label;
  const renders = userRenders(doc);
  if (renders) return renders.label;
  if (state.live.running && state.live.queued) return 'Waiting its turn';
  if (state.live.running || doc.director.state === 'running') return activity(state.live).replace(/ s\d+ ·/, '') || { plan: 'Planning…', shoot: 'Shooting…', reshoot: 'Fixing…', assemble: 'Assembling…' }[doc.director.task] || 'Working…';
  if (doc.director.state === 'queued') return 'Queued';
  if (doc.director.state === 'pausing') return 'Pausing…';
  if (doc.director.state === 'paused') return 'Paused';
  return state.directorInfo.available ? 'Ready' : 'Unavailable';
}

function brand() {
  return `<div class="brand" id="brand" title="All productions"><span class="brand-mark">${icon('logo')}</span><span>Showrunner</span></div>`;
}
function guideButton() {  // with the default models: both are about every production
  return `<button class="btn ghost" id="default-models" title="Default models: the models new productions start with">${icon('bolt')}<span>Default models</span></button>
    <button class="btn ghost" id="guide" title="How Showrunner works (?)">${icon('help')}<span>Guide</span></button>`;
}
// The guide's code and pages load when it first opens, at the part about the page on screen.
const openGuide = () => import('./guide.js').then(module => module.openGuide(GUIDE_PAGES[state.pid ? state.view : 'home']));
function themeButton() {
  return `<button class="btn ghost icon" id="settings" title="Settings">${icon('settings')}</button>
    <button class="btn ghost icon" id="theme" title="Light / dark">${icon(document.documentElement.dataset.theme === 'light' ? 'moon' : 'sun')}</button>`;
}
document.addEventListener('click', event => {
  if (event.target.closest('#brand')) location.hash = '#/';
  if (event.target.closest('#guide')) openGuide();
  if (event.target.closest('#default-models')) openDefaultModels();
  if (event.target.closest('#settings')) openSettings(() => { director?.loadSettings(); loadDeepyTab(); if (state.view === 'cast' && state.pid) route(); });
  if (event.target.closest('#theme')) {
    localStorage.setItem('sr.theme', document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
    applyTheme();
    renderTopbar();
    mounted?.theme?.();
    conversation?.theme();
  }
});

function renderRail() {
  const rail = document.getElementById('rail');
  const review = state.doc?.brief.review_before_edit ? state.doc.shots.filter(shot => shot.status === 'rendered').length : 0;  // takes wait for the user only when the brief asks for it
  if (!draw(rail, Object.entries(VIEWS).map(([key, [label, iconName]]) => `<button class="${state.view === key ? 'on' : ''}" data-view="${key}" title="${label}">${icon(iconName)}<span>${label}</span>${key === 'storyboard' && review ? `<span class="badge">${review}</span>` : ''}</button>`).join(''))) return;
  rail.querySelectorAll('[data-view]').forEach(button => button.onclick = () => actions.go(button.dataset.view));
}

function renderSide() {
  const side = document.getElementById('side');
  if (!side || !state.doc) return;
  side.classList.toggle('collapsed', !state.sideOpen);
  if (!inspector || !director) {
    side.innerHTML = `<div class="side-tabs"><button data-panel="inspector" title="The selected shot">${icon('target')}<span>Shot</span></button><button data-panel="director" title="Talk to the showrunner">${icon('director')}<span>Showrunner</span></button><button data-panel="deepy" title="How Deepy works on this production: its thoughts and every step. Read only.">${icon('chat')}<span>Deepy</span></button><button class="dock-handle" id="dock">${icon('chevron')}</button></div>
      <div class="side-body" id="side-inspector"></div><div class="side-body" id="side-director" style="display:flex;flex-direction:column"></div><div class="side-body" id="side-deepy"></div>`;
    side.querySelectorAll('[data-panel]').forEach(button => button.onclick = () => {
      state.side = button.dataset.panel;
      actions.dock(true);
    });
    side.querySelector('#dock').onclick = () => actions.dock(!state.sideOpen);
    inspector = mountInspector(side.querySelector('#side-inspector'));
    director = mountDirector(side.querySelector('#side-director'));
    conversation = mountConversation(side.querySelector('#side-deepy'));
    loadDeepyTab();
  }
  side.querySelector('#dock').title = state.sideOpen ? 'Fold this panel (D)' : 'Open this panel (D)';
  if (state.side === 'deepy' && !state.deepyTab) state.side = 'director';
  side.querySelector('[data-panel="deepy"]').classList.toggle('hidden', !state.deepyTab);
  side.querySelectorAll('[data-panel]').forEach(button => button.classList.toggle('on', button.dataset.panel === state.side));
  side.querySelector('#side-inspector').classList.toggle('hidden', state.side !== 'inspector');
  side.querySelector('#side-director').classList.toggle('hidden', state.side !== 'director');
  side.querySelector('#side-deepy').classList.toggle('hidden', state.side !== 'deepy');
  inspector.update(state.doc);
  director.update(state.doc);
  conversation.visible(state.sideOpen && state.side === 'deepy');
  renderRail();
}
function loadDeepyTab() {
  api.settings().then(settings => { state.deepyTab = settings.show_deepy_tab; renderSide(); }).catch(() => {});
}
// The panel starts open from 1280 pixels, then stays as the user left it.
function dockOpen() {
  try {
    const saved = localStorage.getItem('sr.dock');
    if (saved) return saved === '1';
  } catch { /* a convenience */ }
  return window.innerWidth >= 1280;
}
// WanGP's button in this header (Full Screen in its tab, back to WanGP in full screen), in place of a bar above the page.
function wangpButton(bar) {
  try { window.parent.WanGPAddonNavigation?.headerButton(window.frameElement, bar, 'btn ghost icon'); } catch { /* opened on its own */ }
}

// Routing -------------------------------------------------------------------------------------------------------------------------
async function route() {
  const [, kind, pid, view] = location.hash.match(/^#\/(p)\/([^/]+)\/?(\w+)?/) || [];
  if (kind !== 'p') return openHome();
  if (pid !== state.pid || !state.doc) {
    state.pid = pid;
    state.doc = null;
    state.selection = new Set();
    state.filter = 'all';  // a production opens on all its shots, not on the filter left in another one
    state.live = { text: '', tools: [], status: null, progress: null, running: false };
    try { setDoc(await api.get(pid)); } catch (error) { toast(error.message, 'error'); location.hash = '#/'; return; }
    catalogStop?.(); catalogStop = null;
    if (active) eventsFor(pid);
    renderShell();
    try {
      const status = await api.director(pid);
      state.live.running = status.running;
      if (status.running && status.live) Object.assign(state.live, status.live);  // the work under way, not only what comes next
      state.directorInfo = status.info || { available: status.available, reason: status.reason };
      renderAdvice();
    } catch { /* the page works without it */ }
  }
  const doc = state.doc;
  const fallback = doc.shots.length ? 'storyboard' : 'brief';
  state.view = VIEWS[view] ? view : fallback;
  showPlace();
  renderTopbar();
  renderRail();
  mounted?.destroy?.();
  const container = document.getElementById('view');
  container.innerHTML = '';
  container.className = 'view';
  void container.offsetWidth;
  mounted = VIEWS[state.view][2](container, doc);
  if (!bin) bin = mountBin(document.getElementById('bin'));
  bin.update(doc);
  document.getElementById('bin').classList.toggle('hidden', !['storyboard', 'timeline', 'cast'].includes(state.view));
  renderSide();
}

async function openHome() {
  stopEvents?.(); stopEvents = null;
  mounted?.destroy?.(); mounted = null;
  state.pid = null; state.doc = null; inspector = director = conversation = bin = null;
  showPlace();
  app.innerHTML = `<header class="topbar" id="topbar"></header><div class="view" id="view"></div>`;
  renderTopbar();
  const container = document.getElementById('view');
  homeRefresh = async () => {
    try {
      const data = await api.state();
      const same = container.querySelector('.home') && JSON.stringify([data.productions, data.director]) === JSON.stringify([state.catalog, state.directorInfo]);
      Object.assign(state, { catalog: data.productions, directorInfo: data.director, kinds: data.kinds, aspects: data.aspects });
      if (!state.pid && !same) renderHome(container, state, { open: id => location.hash = `#/p/${id}` });  // unchanged: the cards and their posters stay
    } catch (error) { toast(error.message, 'error'); }
  };
  await homeRefresh();
  if (active) watchCatalog();
}

function watchCatalog() {
  catalogStop?.();
  catalogStop = listen(null, { productions: homeRefresh, doc: debounce(homeRefresh, 600), reconnected: homeRefresh });
}

// Nothing runs while the page is out of sight (its WanGP tab is not selected, or the browser tab is in the background): its live events
// stop and playback pauses. Shown again, it reconnects and catches up.
let tabShown = true, active = document.visibilityState === 'visible', homeRefresh = null;
function updateActivity() {
  const now = tabShown && document.visibilityState === 'visible';
  if (now === active) return;
  active = now;
  if (!active) {
    stopEvents?.(); stopEvents = null;
    catalogStop?.(); catalogStop = null;
    actions.pausePlayback();
  } else if (state.pid && state.doc) { eventsFor(state.pid); catchUp(); }
  else if (homeRefresh && !state.pid) { homeRefresh(); watchCatalog(); }
}
document.addEventListener('visibilitychange', updateActivity);
// WanGP hides an unselected tab with display:none, and showing the frame again replays every CSS animation of its page: an entrance
// plays once (the page, its cards and messages do not slide in again each time the tab is selected; a new page still does).
document.addEventListener('animationend', event => { if (event.animationName === 'rise') event.target.classList.add('risen'); });
window.addEventListener('message', event => {  // from WanGP's navigation, when this app is embedded in a tab
  if (event.source !== window.parent || event.origin !== location.origin || !event.data?.wangpWebView) return;
  tabShown = event.data.wangpWebView === 'shown';
  updateActivity();
  showPlace();
});

// Embedded in WanGP (its tab or full screen), the address bar is WanGP's: while the tab is shown, it carries the open production and page
// (and, in the tab, the app to open), so that a reload or a copied link comes back to them, as on Showrunner's own page.
const host = (() => { try { return state.embedded && parent !== window && parent.location.origin === location.origin ? parent : null; } catch { return null; } })();
let shownPlace = host && /^#\/p\//.test(host.location.hash) ? host.location.hash : '';
if (shownPlace && !location.hash) history.replaceState(history.state, '', shownPlace);  // a reload or a link: open the place it names
function showPlace() {
  if (!host) return;
  const url = new URL(host.location.href), place = tabShown && state.pid ? `#/p/${state.pid}/${state.view}` : '';
  if (place || url.hash === shownPlace) url.hash = place;  // leaves alone a place that another app wrote
  shownPlace = place;
  const tab = url.pathname.replace(/\/$/, '') !== location.pathname.replace(/\/$/, '');  // full screen is at the app's own folder URL
  if (place && tab) url.searchParams.set('app', 'showrunner'); else if (url.searchParams.get('app') === 'showrunner') url.searchParams.delete('app');
  if (url.href !== host.location.href) host.history.replaceState(host.history.state, '', url);
}
window.addEventListener('resize', showPlace);  // the frame resizes when WanGP enters or leaves full screen, which rewrites its address

window.addEventListener('hashchange', route);

// A picture of the page held and moved is no file brought in, though browsers hand it to a drop as one: it would land in the bin or on a card.
window.addEventListener('dragstart', event => { if (event.target.tagName === 'IMG' && !event.target.closest('[draggable="true"]')) event.preventDefault(); });

// Files dropped anywhere on a production go to its bin, unless a drop target took them.
let dragDepth = 0;
window.addEventListener('dragenter', event => { if (state.pid && event.dataTransfer?.types?.includes('Files')) { dragDepth++; document.getElementById('veil')?.classList.add('on'); } });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.getElementById('veil')?.classList.remove('on'); } });
window.addEventListener('dragover', event => { if (event.dataTransfer?.types?.includes('Files')) event.preventDefault(); });
window.addEventListener('drop', event => {
  dragDepth = 0;
  document.getElementById('veil')?.classList.remove('on');
  if (!event.dataTransfer?.files?.length) return;
  event.preventDefault();
  if (state.pid) actions.upload(event.dataTransfer.files, 'bin');  // drop targets that use the files stop the event before it gets here
});

window.addEventListener('keydown', event => {
  if (document.querySelector('.modal-back')) return;  // a dialog owns the keyboard: no undo or shortcut behind it
  if (document.querySelector('.menu')) { if (event.key === 'Escape') { event.preventDefault(); closeMenu(); } return; }
  const typing = event.target.closest('input, textarea, select, [contenteditable]');
  const ctrl = event.ctrlKey || event.metaKey;
  if (ctrl && event.key.toLowerCase() === 'z' && !typing) { event.preventDefault(); event.shiftKey ? actions.redo() : actions.undo(); return; }
  if (ctrl && event.key.toLowerCase() === 'y' && !typing) { event.preventDefault(); actions.redo(); return; }
  if (typing) return;
  if (event.key === '?') { openGuide(); return; }
  if (!state.doc) return;
  const key = event.key.toLowerCase();
  if (['1', '2', '3', '4', '5'].includes(key)) { actions.go(Object.keys(VIEWS)[+key - 1]); return; }
  if (key === 'd') { actions.toggleSide('director'); return; }
  mounted?.key?.(event);
});

route();
