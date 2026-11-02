// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Showrunner panel: the conversation in time order, what the showrunner is doing now, an input for tasks or notes while it works, and what can be done at this moment.
import { state, actions, activity, deepyWait } from './app.js';
import { api } from './api.js';
import { icon, esc, shotRefs, count, availability } from './ui.js';
import { toShoot, shootTime } from './storyboard.js';
import { askForFix } from './fix.js';

const MAX_STEPS = 60;  // shown at once in a long task
const TASK_LABEL = { plan: 'Planning the production', shoot: 'Shooting', reshoot: 'Fixing', assemble: 'Assembling the edit', review: 'Reviewing in depth', continue: 'Continuing', message: 'Working on your note' };

export function mountDirector(root) {
  root.innerHTML = `<div class="director"><div id="d-banner"></div><div class="dir-feed" id="d-feed"></div>
    <div class="composer"><div class="d-status" id="d-status"></div><div id="d-context"></div>
      <div class="box"><textarea id="d-input" rows="1"></textarea><button class="btn primary small icon" id="d-send" title="Send">${icon('send')}</button></div>
      <div class="quick" id="d-actions"></div></div></div>`;
  const feed = root.querySelector('#d-feed'), input = root.querySelector('#d-input'), written = new WeakMap();
  let historyEnd = -1, stick = true, ignored = '';  // ignored: the selection the user took off the next message
  let showAllErrors = false;
  const loadSettings = () => api.settings().then(settings => { showAllErrors = settings.show_all_errors; if (state.doc) renderFeed(); }).catch(() => {});
  loadSettings();
  feed.addEventListener('scroll', () => { stick = feed.scrollTop + feed.clientHeight > feed.scrollHeight - 40; });

  // Live events come many times a second: what did not change stays as it is, with the button under the pointer and its tooltip.
  function write(element, html) {
    if (written.get(element) === html) return;
    written.set(element, html);
    element.innerHTML = html;
  }
  function working() { return state.live.running || (state.doc && actions.busy(state.doc)); }
  function paused() { return !state.live.running && state.doc?.director.state === 'paused'; }
  // The shots a message is about: those selected on the board or the timeline, shown above the input, unless the user took them off.
  function about() {
    const ids = ['storyboard', 'timeline'].includes(state.view) ? [...state.selection] : [];
    return ids.join(',') === ignored ? [] : ids;
  }

  function renderBanner() {
    const info = state.directorInfo;
    write(root.querySelector('#d-banner'), !info.available && info.reason ? `<div class="banner">${icon('alert')}<span>${esc(info.reason)}</span></div>` : info.simulated ? `<div class="banner">${icon('info')}<span>Simulated director (development mode, no GPU).</span></div>` : '');
  }

  function renderStatus() {
    const doc = state.doc, info = state.directorInfo, live = state.live, busy = working();
    const task = TASK_LABEL[doc.director.task] || 'Working…';
    const step = [activity(live) || live.tools.findLast(tool => tool.status === 'running')?.label, live.progress?.status || live.progress?.phase].filter(Boolean).join(' · ');
    const line = doc.director.state === 'pausing' ? 'Pausing after its current step…' : !busy ? (paused() ? `Paused · ${task.toLowerCase()}` : info.available ? 'Ready · Deepy Prime' : 'Unavailable')
      : deepyWait(live) || step || live.status?.text || task;
    const percent = live.progress ? `<span class="mono">${Math.round(live.progress.progress || 0)}%</span>` : '';
    const reload = busy && live.status?.progress ? `<div class="progress" title="Reloading this production's conversation into Deepy"><i style="width:${Math.round(100 * live.status.progress.done / live.status.progress.total)}%"></i></div>` : '';
    write(root.querySelector('#d-status'), `<div class="d-line">${busy ? '<span class="typing"><i></i><i></i><i></i></span>' : ''}<span class="d-doing" ${busy && live.stats?.text ? `title="${esc(live.stats.text)}"` : ''}>${shotNames(line)}</span>${percent}</div>${live.progress ? `<div class="progress"><i style="width:${live.progress.progress || 0}%"></i></div>` : ''}${reload}`);
    input.placeholder = busy ? 'Your note reaches it at its next step…' : paused() ? 'Your note resumes its task…' : 'Ask for a change: "make shot 3 at night"…';
  }

  function renderActions() {
    const doc = state.doc, blocked = actions.directorBlocked(), selected = [...state.selection];
    let items;  // action, icon, label, its tooltip, why it cannot be used now
    if (paused()) items = [['resume', 'play', 'Resume', 'Continue the paused task where it stopped'], ['stop', 'stop', 'Stop', 'Give up the paused task']];
    else if (working()) {
      items = [doc.director.state === 'pausing' ? ['pause-now', 'pause', 'Pause now', 'Stop what it is generating too: it is generated again on resume']
        : ['pause', 'pause', 'Pause', 'Pause after the current step and free the GPU; Resume continues the task in the same conversation, even after a restart'], ['stop', 'stop', 'Stop', 'Stop the task']];
    } else if (!doc.shots.length) items = [['plan', 'wand', 'Plan', '', blocked || actions.planBlocked(doc)]];
    else {
      const shots = toShoot(doc), estimate = shots.length ? shootTime(doc, shots, renderActions) : '';
      items = [['shoot', 'clapper', 'Shoot', estimate && `${count(shots.length, 'shot')} to shoot: ${estimate}`, blocked || (shots.length ? '' : 'Nothing to shoot: every shot has a take. To shoot one again, select it and use Fix, or mark its take for a redo.')]];
      if (selected.length) items.push(['reshoot', 'refresh', `Fix ${count(selected.length, 'shot')}…`, '', blocked]);
      items.push(['assemble', 'film', 'Assemble', 'The showrunner assembles the edit, watches it and reports', blocked || actions.assembleBlocked(doc)], ['plan', 'wand', 'Re-plan', '', blocked || actions.planBlocked(doc)]);
      if (['cancelled', 'error', 'interrupted'].includes(doc.director.last?.status)) items.push(['continue', 'play', 'Continue', '', blocked]);
    }
    write(root.querySelector('#d-actions'), items.map(([task, iconName, label, title, reason]) => `<button class="chip clickable ${task === 'stop' ? 'danger' : ''}" data-task="${task}" ${availability(reason, title)}>${icon(iconName)}${label}</button>`).join(''));
    const shots = about(), numbers = shots.map(id => doc.shots.findIndex(shot => shot.id === id) + 1).filter(Boolean);
    write(root.querySelector('#d-context'), numbers.length ? `<span class="chip context" title="Your message is about the selected shots">${icon('target')}About shot${numbers.length > 1 ? 's' : ''} ${numbers.join(', ')}<button class="context-x" id="d-context-x" title="Send without the selected shots">${icon('x')}</button></span>` : '');
  }
  root.querySelector('#d-context').addEventListener('click', event => {
    if (event.target.closest('#d-context-x')) { ignored = about().join(','); renderActions(); }
  });
  root.querySelector('#d-actions').addEventListener('click', event => {
    const task = event.target.closest('[data-task]')?.dataset.task;
    if (task === 'stop') actions.stop();
    else if (task === 'pause' || task === 'pause-now') actions.pause(task === 'pause-now');
    else if (task === 'resume') actions.resume();
    else if (task === 'reshoot' && !working()) askForFix([...state.selection]);
    else if (task && !working()) actions.direct(task);
  });

  function message(entry) {
    const who = entry.by === 'user' ? 'user' : entry.by === 'deepy' ? 'deepy' : 'system';
    return `<div class="msg ${who}">${shotNames(entry.text.replace(/^Told the showrunner: /, ''))}${who !== 'system' ? `<span class="when">${new Date(entry.time * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>` : ''}</div>`;
  }

  // What the showrunner says and does during its task, in the order it comes: each step knows how many entries of the log came before it.
  function noteSteps() {
    const live = state.live, anchor = state.doc.log.length, steps = live.steps ??= [], known = live.knownSteps ??= new Map();
    for (const tool of live.tools) {
      if (known.has(tool.id)) known.get(tool.id).tool = tool;
      else { const step = { tool, anchor }; known.set(tool.id, step); steps.push(step); }
    }
    const said = steps.findLast(step => step.said != null);
    if (live.text && live.text !== said?.said) {
      if (said && live.text.startsWith(said.said)) said.said = live.text;
      else steps.push({ said: live.text, anchor });
    }
  }

  function stepLine(step, logged) {
    if (step.said != null) return logged.has(step.said) ? '' : `<div class="msg deepy">${shotNames(step.said.slice(-1800))}</div>`;  // its last words join the log when its task ends
    const tool = step.tool, label = tool.label || tool.name || '', failed = tool.status === 'error' || tool.error;
    if (!showAllErrors && (/Schema$/.test(label) || (failed && (REFUSED.test(tool.error || '') || /^(Unknown Tool|Retrying)/.test(label))))) return '';  // the contracts it reads, the calls it corrects by itself
    return `<div class="step ${failed ? 'error' : tool.status}">${icon(tool.status === 'running' ? 'loader' : failed ? 'alert' : 'check')}<span>${shotNames(label.replace(/^Showrunner: /, ''))}${tool.error ? `<span class="why">${esc(showAllErrors ? tool.error : tool.error.slice(0, 200))}</span>` : ''}</span></div>`;
  }

  function renderFeed(force = false) {
    const doc = state.doc;
    noteSteps();
    const end = Math.min(state.live.steps[0]?.anchor ?? doc.log.length, doc.log.length);  // what comes before the task's first step does not change
    if (force || end !== historyEnd) {
      historyEnd = end;
      feed.innerHTML = doc.log.slice(0, end).map(message).join('') + '<div id="d-live"></div>';
    }
    renderLive();
  }

  // The task's steps among the log's entries, then what it changed.
  function renderLive() {
    const doc = state.doc, live = state.live, element = feed.querySelector('#d-live');
    if (!element) return;
    const logged = new Set(doc.log.slice(-8).map(entry => entry.text)), steps = live.steps.slice(-MAX_STEPS);
    let html = live.steps.length > MAX_STEPS ? `<div class="step more">${count(live.steps.length - MAX_STEPS, 'earlier step')}</div>` : '', next = 0;
    for (let index = historyEnd; index <= doc.log.length; index++) {
      while (next < steps.length && steps[next].anchor <= index) html += stepLine(steps[next++], logged);
      if (index < doc.log.length) html += message(doc.log[index]);
    }
    const summary = !working() && live.summary ? `<div class="msg system changes">${icon('check')} ${shotNames(live.summary)}</div>` : '';
    write(element, doc.log.length || html || summary ? html + summary : `<div class="msg system">The showrunner's decisions and your notes appear here.</div>`);
    if (stick) feed.scrollTop = feed.scrollHeight;
  }

  function send() {
    const text = input.value.trim();
    if (!text) return;
    const selected = about();
    const full = selected.length ? `About ${selected.join(", ")}: ${text}` : text;  // shown as "About shot 2, shot 3: …"
    input.value = '';
    ignored = selected.join(',');  // a selection is about one message: selecting again brings the chip back
    autosize();
    if (working()) actions.steer(full);
    else if (paused()) actions.resume(full);
    else actions.direct('message', full, selected);
  }
  function autosize() { input.style.height = 'auto'; if (input.value) input.style.height = `${Math.min(140, input.scrollHeight)}px`; }  // empty: one line, whatever the width of its placeholder
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } });
  root.querySelector('#d-send').onclick = send;

  return {
    loadSettings,
    update() { renderBanner(); renderStatus(); renderActions(); renderFeed(); },
    live() { renderBanner(); renderStatus(); renderActions(); renderFeed(); },
  };
}

// Deepy writes Markdown: the panel shows its words (a link by its text, code as plain words, bold as bold).
// A call refused before anything ran: Deepy reads why and corrects it, and nothing was lost. Anything else that failed is the user's to see.
const REFUSED = /^(Deepy's tool call|No action was named|Name the plugin|MCP tool result exceeded|Markdown section not found|arguments is not valid JSON|Unknown or unavailable action|Unknown Deepy template)|media_id must be returned by|inspect_audio measures audio|Path does not exist|^Error executing tool \w+: (arguments:|Unknown |No [\w ]+ has the id|Every sequence|'Unknown WanGP media_id)|Nothing was (submitted|executed)|is a required property|is not of type|is not one of|is not valid under any|Additional properties are not allowed|Visual inspection currently supports/;
function plainWords(text) { return String(text ?? '').replace(/\[([^\]]*)\]\([^)\s]*\)/g, '$1').replace(/`/g, ''); }
function shotNames(text) { return shotRefsOf(plainWords(text)).replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>'); }
function shotRefsOf(text) { return shotRefs(text, state.doc?.shots || [], state.doc?.cast || [], state.doc?.deliverables || []); }
