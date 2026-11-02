// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Director panel: what Deepy is doing now, the production's log, and a composer to give it tasks or notes while it works.
import { state, actions, nextStep } from './app.js';
import { icon, esc, ago, shotRefs, count, availability } from './ui.js';
import { toShoot, shootTime } from './storyboard.js';

const TASK_LABEL = { plan: 'Planning the production', shoot: 'Shooting', reshoot: 'Re-shooting', assemble: 'Assembling the edit', continue: 'Continuing', message: 'Working on your note' };

export function mountDirector(root) {
  root.innerHTML = `<div class="director"><div class="dir-head" id="d-head"></div><div id="d-banner"></div><div class="dir-feed" id="d-feed"></div>
    <div class="composer"><div class="quick" id="d-quick"></div><div id="d-context"></div>
      <div class="box"><textarea id="d-input" rows="1" placeholder="Tell the showrunner… (Enter to send)"></textarea><button class="btn primary small icon" id="d-send" title="Send">${icon('send')}</button></div>
      <div class="muted" style="font-size:11.5px" id="d-hint"></div></div></div>`;
  const feed = root.querySelector('#d-feed'), input = root.querySelector('#d-input');
  let lastLog = -1, stick = true, ignored = '';  // ignored: the selection the user took off the next message
  let thoughtsFolded = false;
  try { thoughtsFolded = localStorage.getItem('sr.thoughts') === 'folded'; } catch { /* a convenience */ }
  feed.addEventListener('scroll', () => { stick = feed.scrollTop + feed.clientHeight > feed.scrollHeight - 40; });
  feed.addEventListener('click', event => {
    if (!event.target.closest('.thought-head')) return;
    thoughtsFolded = !thoughtsFolded;
    try { localStorage.setItem('sr.thoughts', thoughtsFolded ? 'folded' : 'open'); } catch { /* a convenience */ }
    renderLive();
  });

  function working() { return state.live.running || (state.doc && actions.busy(state.doc)); }
  function paused() { return !state.live.running && state.doc?.director.state === 'paused'; }
  // The shots a message is about: those selected on the board or the timeline, shown above the input, unless the user took them off.
  function about() {
    const ids = ['storyboard', 'timeline'].includes(state.view) ? [...state.selection] : [];
    return ids.join(',') === ignored ? [] : ids;
  }

  function renderHead() {
    const doc = state.doc, info = state.directorInfo;
    const busy = working(), pausing = doc.director.state === 'pausing';
    const task = TASK_LABEL[doc.director.task] || 'Working…';
    const line = pausing ? 'Pausing after its current step…' : busy ? (state.live.queued ? 'Waiting for Deepy to be free…' : task) : paused() ? `Paused · ${task.toLowerCase()}` : info.available ? 'Ready · Deepy Prime' : 'Unavailable';
    const buttons = busy ? `${pausing ? `<button class="btn small" id="d-pause-now" title="Stop what it is generating too: it is generated again on resume">${icon('pause')} Pause now</button>`
      : `<button class="btn small" id="d-pause" title="Pause after the current step and free the GPU; Resume continues the task in the same conversation, even after a restart">${icon('pause')} Pause</button>`}
      <button class="btn small danger" id="d-stop">${icon('stop')} Stop</button>`
      : paused() ? `<button class="btn small primary" id="d-resume" title="Continue the paused task where it stopped">${icon('play')} Resume</button>
      <button class="btn small ghost icon" id="d-stop" title="Give up the paused task">${icon('x')}</button>` : '';
    root.querySelector('#d-head').innerHTML = `<div class="dir-avatar ${busy ? 'working' : ''}">${icon(paused() ? 'pause' : 'director')}</div>
      <div class="grow"><b>Showrunner</b><div class="muted" style="font-size:12.5px">${esc(line)}</div></div>${buttons}`;
    root.querySelector('#d-stop')?.addEventListener('click', () => actions.stop());
    root.querySelector('#d-pause')?.addEventListener('click', () => actions.pause());
    root.querySelector('#d-pause-now')?.addEventListener('click', () => actions.pause(true));
    root.querySelector('#d-resume')?.addEventListener('click', () => actions.resume());
    const next = nextStep(doc);
    root.querySelector('#d-banner').innerHTML = (!info.available && info.reason ? `<div class="banner">${icon('alert')}<span>${esc(info.reason)}</span></div>` : info.simulated ? `<div class="banner">${icon('info')}<span>Simulated director (development mode, no GPU).</span></div>` : '')
      + (next ? `<div class="banner turn">${icon('target')}<span class="grow"><b>Your turn.</b> ${esc(next.text)}</span>${next.view && next.view !== state.view ? `<button class="btn small" id="d-go">Open</button>` : ''}</div>` : '');
    root.querySelector('#d-go')?.addEventListener('click', () => actions.go(next.view));
  }

  function renderQuick() {
    const doc = state.doc, busy = working();
    const selected = [...state.selection];
    const items = [];  // task, icon, label, its tooltip, why it cannot be used now
    if (!doc.shots.length) items.push(['plan', 'wand', 'Plan', '', actions.planBlocked(doc)]);
    else {
      const shots = toShoot(doc), estimate = shots.length ? shootTime(doc, shots, renderQuick) : '';
      items.push(['shoot', 'clapper', 'Shoot', estimate && `${count(shots.length, 'shot')} to shoot: ${estimate}`, shots.length ? '' : 'Nothing to shoot: every shot has a take. To shoot one again, select it and use Re-shoot, or mark its take for a redo.']);
      if (selected.length) items.push(['reshoot', 'refresh', `Re-shoot ${count(selected.length, 'shot')}`]);
      items.push(['assemble', 'film', 'Assemble', 'The showrunner assembles the edit, watches it and reports', actions.assembleBlocked(doc)], ['plan', 'wand', 'Re-plan', '', actions.planBlocked(doc)]);
    }
    if (paused()) items.unshift(['resume', 'play', 'Resume']);
    else if (['cancelled', 'error', 'interrupted'].includes(doc.director.last?.status)) items.push(['continue', 'play', 'Continue']);
    const blocked = actions.directorBlocked();
    root.querySelector('#d-quick').innerHTML = items.map(([task, iconName, label, title, reason]) => `<button class="chip clickable" data-task="${task}" ${availability(blocked || reason, title)}>${icon(iconName)}${label}</button>`).join('');
    root.querySelectorAll('[data-task]').forEach(button => button.onclick = () => !working() && (button.dataset.task === 'resume' ? actions.resume() : actions.direct(button.dataset.task, '', button.dataset.task === 'reshoot' ? selected : [])));
    root.querySelector('#d-hint').textContent = busy ? 'Your message reaches the showrunner at its next step.' : paused() ? 'The showrunner is paused: your message resumes its task with your note.' : 'Ask for changes in plain language: "faster cuts in the chorus", "make shot 3 at night"…';
    const shots = about(), numbers = shots.map(id => doc.shots.findIndex(shot => shot.id === id) + 1).filter(Boolean);
    root.querySelector('#d-context').innerHTML = numbers.length ? `<span class="chip context" title="Your message is about the selected shots">${icon('target')}About shot${numbers.length > 1 ? 's' : ''} ${numbers.join(', ')}<button class="context-x" id="d-context-x" title="Send without the selected shots">${icon('x')}</button></span>` : '';
    root.querySelector('#d-context-x')?.addEventListener('click', () => { ignored = shots.join(','); renderQuick(); });
  }

  function message(entry) {
    const who = entry.by === 'user' ? 'user' : entry.by === 'deepy' ? 'deepy' : 'system';
    return `<div class="msg ${who}">${shotNames(entry.text)}${who !== 'system' ? `<span class="when">${ago(entry.time)}</span>` : ''}</div>`;
  }

  function renderFeed(force = false) {
    const doc = state.doc;
    if (force || doc.log.length !== lastLog) {
      lastLog = doc.log.length;
      feed.innerHTML = (doc.log.length ? doc.log.map(message).join('') : `<div class="msg system">The showrunner's decisions and your notes appear here.</div>`) + '<div id="d-live"></div>';
    }
    renderLive();
  }

  function renderLive() {
    const live = state.live, element = feed.querySelector('#d-live');
    if (!element) return;
    if (!working() && !live.text && !live.summary) { element.innerHTML = ''; return; }
    const tools = live.tools.filter(tool => !/^(Unknown Tool|Retrying)/.test(tool.label || "") && !/^(Deepy's tool call|Error executing tool \w+: arguments:)/.test(tool.error || "")).slice(-6).map(  // the agent's own retries are no news to the user: a call refused before anything ran
      tool => `<div class="tool-line ${tool.status}">${icon(tool.status === 'running' ? 'loader' : tool.status === 'done' ? 'check' : 'alert')}<span>${shotNames(tool.label || tool.name)}${tool.error ? `<span class="why">${esc(tool.error.slice(0, 160))}</span>` : ''}</span></div>`).join('');
    const progress = live.progress ? `<div class="col" style="gap:4px"><div class="row" style="font-size:12px"><span class="grow muted">${esc(live.progress.status || live.progress.phase || '')}</span><span class="mono">${Math.round(live.progress.progress || 0)}%</span></div><div class="progress"><i style="width:${live.progress.progress || 0}%"></i></div></div>` : '';
    const summary = !working() && live.summary ? `<div class="msg system changes">${icon('check')} ${shotNames(live.summary)}</div>` : '';
    const prefilling = working() && live.stats?.prefilling_tokens;
    const doing = prefilling ? `Prefilling ${prefilling.toLocaleString('en-US')} Tokens...` : live.status?.text || 'Directing';
    const thought = working() && live.thought ? `<div class="thought"><button class="thought-head" aria-expanded="${!thoughtsFolded}">${icon(thoughtsFolded ? 'chevron' : 'chevronDown')}${live.thought.streaming ? 'Thinking' : 'Last Thought'}</button>${thoughtsFolded ? '' : `<div class="thought-text">${shotNames(live.thought.text)}</div>`}</div>` : '';
    const stats = working() && live.stats ? `<div class="live-stats" title="Deepy's reading (prefill) and writing (gen) speed, and how full its context is">${esc(live.stats.text)}</div>` : '';
    const reading = feed.querySelector('.thought-text'), follow = !reading || reading.scrollTop + reading.clientHeight > reading.scrollHeight - 12, position = reading?.scrollTop;
    element.innerHTML = `${summary}<div class="live"><div class="live-head">${working() ? '<span class="typing"><i></i><i></i><i></i></span>' : icon('check')}${working() ? esc(doing) : 'Last reply'}</div>${shotNames((live.text || '').slice(-1800)) || '<span class="muted">Reading the production…</span>'}</div>${thought}${tools ? `<div class="col" style="gap:6px">${tools}</div>` : ''}${progress}${stats}`;
    const written = feed.querySelector('.thought-text');
    if (written) written.scrollTop = follow ? written.scrollHeight : position;  // the latest words, unless the user went back up to read
    if (stick) feed.scrollTop = feed.scrollHeight;
  }

  function send() {
    const text = input.value.trim();
    if (!text) return;
    const selected = about();
    const full = selected.length ? `About ${selected.join(", ")}: ${text}` : text;  // shown as "About shot 2, shot 3: …"
    input.value = '';
    ignored = '';
    autosize();
    if (working()) actions.steer(full);
    else if (paused()) actions.resume(full);
    else actions.direct('message', full, selected);
  }
  function autosize() { input.style.height = 'auto'; input.style.height = `${Math.min(140, input.scrollHeight)}px`; }
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } });
  root.querySelector('#d-send').onclick = send;

  return {
    update() { renderHead(); renderQuick(); renderFeed(); },
    live() { renderHead(); renderQuick(); renderLive(); },
  };
}

function shotNames(text) { return shotRefs(text, state.doc?.shots || [], state.doc?.cast || [], state.doc?.deliverables || []); }
