// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Shot inspector: watch and compare takes, judge them, and edit everything the director will use.
import { COMPARE_SEEK, bindCompareSeek } from './ui.js';
import { askForFix } from './fix.js';
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, art, fieldSaver, formatDuration, toast, promptDialog, menu, confirmDialog, shotRefs, shotText, availability, pickFiles } from './ui.js';
import { shotMenu, notFinal, finalTier, tierToggle, bindTierToggle } from './storyboard.js';
import { showItem } from './bin.js';
import { choiceLabel, rulesButton, showRules, templateLabel } from './models.js';
import { segments } from './sequencer.js';

const TRANSITIONS = [['cut', 'Cut'], ['crossfade', 'Dissolve'], ['fade_black', 'Fade black'], ['fade_white', 'Fade white']];

export async function rerun(shotId) {
  if (state.doc.brief.kind === 'remake') return reshoot([shotId]);  // its neighbours may render with it, at the size Draft / Final says
  try { await api.rerun(state.pid, shotId); toast('Rendering a new take…'); } catch (error) { toast(error.message, 'error', 7000); }
}

// Another take of each, with a new seed, at the size the Draft / Final choice says
export const reshoot = shotIds => renderNow(shotIds, finalTier(state.doc), true);
export const reshootSize = doc => finalTier(doc) || !doc.brief.draft_resolution ? doc.brief.resolution : doc.brief.draft_resolution;

export async function renderNow(shotIds, final = false, newSeed = null) {
  try { await api.renderNow(state.pid, shotIds, final, newSeed); toast(shotIds.length > 1 ? `Rendering ${shotIds.length} shots…` : 'Rendering…'); } catch (error) { toast(error.message, 'error', 7000); }
}

const keptSeconds = (shot, take) => shot.source_range && shot.trim?.[1] != null ? shot.trim[1] - (shot.trim[0] || 0) : +take.duration_s;  // a remake's take runs on past its shot's cut: its length is what the edit keeps

function loopOn() { try { return localStorage.getItem('sr.loop') === '1'; } catch { return false; } }  // off unless this viewer turned it on

export function mountInspector(root) {
  let shownId = null, compareId = null, sourceOn = false, signatureShown = '', promptOpen = false;  // the prompt is the showrunner's: folded unless opened
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
    if (shot.id !== shownId) { compareId = null; sourceOn = false; saveShot.flush(); root.scrollTop = 0; }
    shownId = shot.id;
    const index = doc.shots.indexOf(shot);
    const take = actions.take(shot);
    const compare = compareId === 'source' ? { id: 'source' } : compareId && shot.takes.find(item => item.id === compareId && item.id !== take?.id);
    const verdict = take?.verdict;
    const rendering = shot.status === 'rendering' || state.renders[shot.id] != null;
    const sections = [...new Set(doc.shots.map(item => item.section).filter(Boolean))];
    const settings = shot.generation?.settings || {};
    const remake = doc.brief.kind === 'remake', notForRemake = remake ? ' style="display:none"' : '';  // its motion, length, words and joins come from the source
    const drafts = doc.brief.draft_resolution, busy = rendering ? 'This shot is rendering: wait for its take.' : '';
    const final = finalTier(doc);
    const retake = remake && !final && shot.takes.length;  // as the Board's toolbar: a remake shot with a take gets Reshoot, not the showrunner's Shoot
    const shootButtons = !drafts || take?.kept_source ? '' : `${tierToggle(doc)}${retake ? '' : `<button class="btn ${final && take?.tier === 'draft' ? 'primary' : ''}" id="i-shoot" ${availability(busy || (final ? actions.renderBlocked() || (notFinal(doc, [shot]).length ? '' : 'This shot already has a final take: Reshoot makes another.') : actions.directorBlocked()), final ? 'Renders it at the target size now, with the prompt, seed and pictures of its take: no language model, and nobody looks at the take' : 'The showrunner renders it at the draft size and looks at the take')}>${icon(final ? 'bolt' : 'clapper')} Shoot this shot</button>`}`;
    root.innerHTML = `<div class="inspector">
      <div class="insp-head"><span class="shot-num">${index + 1}</span><input class="input bare grow" id="i-title" value="${esc(shot.title)}" placeholder="Shot ${index + 1}" style="font-weight:650;font-size:15px">
        <button class="btn small icon ${shot.locked ? 'active' : ''}" id="i-lock" title="${shot.locked ? 'Locked: the showrunner keeps it. Click to unlock' : 'Lock: the showrunner will not change it'}">${icon(shot.locked ? 'lock' : 'unlock')}</button>
        ${take ? `<button class="btn small icon approve ${shot.status === 'approved' ? 'active' : ''}" id="i-approve" title="${shot.status === 'approved' ? 'Approved: click to take your approval back' : doc.brief.review_before_edit ? 'Approve this take: the showrunner waits for your approval before the edit' : 'Mark this take as approved. A note for yourself: it changes nothing unless Review the takes is on in the Brief'}">${icon('check')}</button>` : ''}
        ${take && doc.brief.review_before_edit ? `<button class="btn small icon reject ${shot.status === 'rejected' && verdict?.by === 'user' ? 'active' : ''}" id="i-reject" title="Mark the take for a redo, with a note for the showrunner, and carry on reviewing">${icon('x')}</button>` : ''}
        <button class="btn small icon" id="i-more" title="More">${icon('more')}</button></div>
      <div class="player" id="i-player">${playerHtml(shot, take, compare)}</div>
      ${remakeFrames(shot)}${startPictures(shot)}
      ${shot.takes.length ? `<div><div class="label">${icon('layers')} Takes <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">click to use · right-click to compare or remove</span></div>
        <div class="takes">${shot.source_range ? `<div class="take ${sourceOn || compare?.id === 'source' ? 'cmp' : ''}" data-source title="Your video: click to watch it here, right-click to compare it with the used take"><img src="${shot.source_poster ? actions.file(shot.source_poster) : ''}" alt=""><span class="take-label">Source</span></div>` : ''}${shot.takes.map((item, number) => `<div class="take ${item.id === take?.id ? 'on' : ''} ${item.id === compare?.id ? 'cmp' : ''}" data-take="${item.id}" title="${esc(shotText(item.note || "", doc.shots, doc.cast))} · ${item.by === 'deepy' ? 'the showrunner' : item.by === 'user' ? 'you' : item.by}">
          ${item.poster ? `<img src="${actions.file(item.poster)}" alt="">` : `<div style="aspect-ratio:16/9;${art(item.id)}"></div>`}<span class="take-label">T${number + 1}${item.kept_source ? ' · kept' : drafts && item.height ? ` · ${Math.min(item.width, item.height)}p` : item.duration_s ? ` · ${keptSeconds(shot, item).toFixed(1)}s` : ''}</span>
          ${item.verdict ? `<span class="verdict-dot" style="background:${item.verdict.ok ? 'var(--green)' : 'var(--red)'}"></span>` : ''}</div>`).join('')}</div></div>` : ''}
      ${shot.status === 'failed' ? failure(shot, doc) : ''}
      ${take ? `<div class="verdict-row">
          ${shootButtons}
          <button class="btn ${drafts && take?.tier === 'draft' ? '' : 'primary'}" id="i-fix" ${availability(actions.directorBlocked(), 'Say what is wrong or what should change: the showrunner fixes this shot and keeps the rest')}>${icon('director')} Fix…</button>
          ${doc.brief.review_before_edit ? `<button class="btn reject ${shot.status === 'rejected' && verdict?.by === 'user' ? 'active' : ''}" id="i-reject" title="Mark the take for a redo, with a note for the showrunner, and carry on reviewing">${icon('x')} Redo…</button>` : ''}
          ${take.settings && Object.keys(take.settings).length ? `<button class="btn" id="i-rerun" ${availability(rendering ? 'This shot is rendering: wait for its take.' : '', 'Another take now, with the same prompt and pictures and a new seed, without the showrunner: nobody is told what was wrong. To say it, use Fix.')}>${icon('refresh')} Reshoot</button>` : ''}</div>
        ${[verdict, verdict?.by === 'user' && take.showrunner_verdict].filter(Boolean).map(item => `<div class="verdict-note ${item.ok ? 'ok' : 'bad'}">${icon(item.ok ? 'check' : 'alert')}<span><b>${item.by === 'user' ? 'You' : 'Showrunner'}:</b> ${shotRefs(item.note || (item.by === 'user' ? (item.ok ? 'Approved.' : 'Marked for a redo.') : item.ok ? 'Good take.' : 'Needs a retry.'), doc.shots, doc.cast)}</span></div>`).join('')}
        ${heardHtml(take, shot)}${checksHtml(take, shot)}` : `<div class="verdict-row">${shootButtons}${drafts ? '' : `<button class="btn primary" id="i-shoot" ${availability(rendering ? 'This shot is rendering: wait for its take.' : actions.directorBlocked())}>${icon('clapper')} ${rendering ? 'Rendering…' : 'Shoot this shot'}</button>`}
          ${rendering ? '' : `<button class="btn" id="i-render" ${availability(actions.renderBlocked(), 'Renders it now from its prompt, without the showrunner: no language model, and nobody checks the take. Without a prompt, one is made from what you wrote on the shot.')}>${icon('bolt')} Render now</button>`}
          ${rendering || remake ? '' : `<button class="btn" id="i-own-take" title="A video or a picture of yours as this shot's take, instead of rendering one">${icon('upload')} Use my video…</button>`}</div>`}
      ${remake ? '<div class="together-note" id="i-together" hidden></div>' : ''}
      <div class="field"><div class="label">${icon('sparkles')} What happens</div><textarea class="textarea" id="i-intent" rows="3" placeholder="${{ music_video: 'The singer turns to the camera as the chorus starts…', ad: 'The product in use, close up, the moment it pays off…', story: 'The fox finds the lantern at the edge of the wood…', remake: 'What changes in this shot: who replaces whom, what is removed or restyled…' }[doc.brief.kind] || 'She hears the radio crackle and turns to the window…'}">${esc(shot.intent)}</textarea></div>
      <div class="field"${notForRemake}><div class="label">${icon('chat')} Dialogue <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">${doc.brief.voices_from_model && !['music_video', 'story'].includes(doc.brief.kind) ? 'the video model says these words: they join the prompt when the shot renders' : 'words spoken or sung in this shot'}</span></div>
        <div class="dialogue" id="i-dialogue">${(shot.dialogue || []).map(item => lineRow(item, doc)).join('')}</div>
        <button class="btn small ghost" id="i-add-line" style="align-self:flex-start">${icon('plus')} Line</button>
        ${doc.brief.voices_from_model && !['music_video', 'story'].includes(doc.brief.kind) && (shot.dialogue || []).length ? `<label class="row muted" style="font-size:12.5px;cursor:pointer;gap:10px" title="After each take, the shot starts just before its first line and ends just before any word the model adds after its last line. Turn it off for a shot that should run on."><span class="toggle ${shot.cut_extra_words !== false ? 'on' : ''}" id="i-cut-words"></span>Cut words that are not in these lines</label>` : ''}</div>
      <details class="details" id="i-prompt-box" ${promptOpen ? 'open' : ''}><summary>${icon('chevron')} Prompt <span class="muted" id="i-count" style="font-weight:500">${remake ? 'what the model reads' : 'made from what happens, the camera and the cast'} · ${(settings.prompt || '').length} chars</span></summary>
        <textarea class="textarea" id="i-prompt" rows="6" style="margin-top:8px" placeholder="The showrunner writes the full generation prompt here. Render now makes one from what you wrote above.">${esc(settings.prompt || '')}</textarea></details>
      <div class="insp-grid">
        <div class="field"${notForRemake}><div class="label">${icon('clock')} Duration ${take?.duration_s ? `<span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500" title="What the edit plays of this shot's video">plays ${(segments(doc, actions.take).find(item => item.shot.id === shot.id)?.duration ?? 0).toFixed(1)} s of ${take.duration_s.toFixed(1)} s</span>` : ''}</div><input class="input" id="i-duration" type="number" min="0.5" max="60" step="0.1" value="${shot.duration_s ?? ''}"></div>
        <div class="field"><div class="label">${icon('layers')} Section</div><input class="input" id="i-section" list="i-sections" value="${esc(shot.section)}"><datalist id="i-sections">${sections.map(name => `<option value="${esc(name)}">`).join('')}</datalist></div>
        <div class="field" style="grid-column:1/-1${remake ? ';display:none' : ''}"><div class="label">${icon('video')} Camera</div><input class="input" id="i-camera" value="${esc(shot.camera)}" placeholder="Slow dolly-in, low angle…"></div>
        <div class="field" style="grid-column:1/-1"><div class="label">${icon('bolt')} Model</div><div class="model-pick"><select class="select" id="i-model" title="The Deepy template that renders this shot">
          <option value="${esc(shot.generation?.template || 'gen_video')}">${esc(templateLabel(shot.generation?.template))}</option></select>${rulesButton()}</div></div>
      </div>
      ${doc.cast.length ? `<div class="field"><div class="label">${icon('users')} In this shot</div><div class="cast-toggle">${doc.cast.map(member => {
        const image = member.master || member.images[0];
        return `<button class="chip ${shot.cast.includes(member.id) ? 'on' : ''}" data-cast="${member.id}"><span class="avatar-xs" style="${image ? `background-image:url('${actions.file(image)}')` : ''}">${image ? '' : esc((member.name || '?')[0])}</span>${esc(member.name || member.id)}</button>`;
      }).join('')}</div></div>` : ''}
      <div class="field"${notForRemake}><div class="label">${icon('transition')} Transition into this shot</div>
        <div class="row"><div class="seg" id="i-transition">${TRANSITIONS.map(([key, label]) => `<button class="${(shot.transition?.type || 'cut') === key ? 'on' : ''}" data-tr="${key}">${label}</button>`).join('')}</div>
        ${(shot.transition?.type || 'cut') !== 'cut' ? `<input class="input" id="i-tr-d" type="number" min="0.1" max="3" step="0.1" value="${shot.transition?.duration_s || 0.5}" style="width:74px" title="Seconds">` : ''}</div></div>
      ${take?.duration_s ? `<div class="field"><div class="label">${icon('scissors')} Trim <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">seconds of the take</span></div>
        <div class="row"><input class="input" id="i-in" type="number" min="0" step="0.05" value="${shot.trim?.[0] ?? 0}" style="width:90px"><span class="muted">to</span><input class="input" id="i-out" type="number" min="0" step="0.05" value="${shot.trim?.[1] ?? ''}" placeholder="${(+take.duration_s).toFixed(2)}" style="width:90px">
        <button class="btn small ghost" id="i-trim-reset">Reset</button><span class="grow"></span><label class="row muted" style="font-size:12.5px;cursor:pointer"><span class="toggle ${shot.included ? 'on' : ''}" id="i-included"></span>In the edit</label></div></div>` : ''}
      ${take?.kind === 'video' ? speedField(shot, doc) : ''}
      ${take?.kind === 'video' && take.has_audio ? soundField(shot, take, doc) : ''}
      <div class="field"><div class="label">${icon('note')} Notes for the showrunner</div><textarea class="textarea" id="i-notes" rows="2" placeholder="Anything to keep in mind for this shot…">${esc(shot.notes)}</textarea></div>
      ${take ? '' : `<div class="row"><button class="btn grow" id="i-ask">${icon('director')} Ask the showrunner</button><button class="btn grow" id="i-reshoot" ${availability(actions.directorBlocked())}>${icon('refresh')} Fix…</button></div>`}
      <details class="details"><summary>${icon('chevron')} Generation settings · ${esc(templateLabel(shot.generation?.template))}</summary>
        <div class="col" style="margin-top:10px"><input class="input mono" id="i-template" value="${esc(shot.generation?.template || 'gen_video')}">
        <textarea class="textarea json" id="i-settings" spellcheck="false">${esc(JSON.stringify(settings, null, 2))}</textarea><div class="muted" style="font-size:12px">JSON merged into the shot's settings when you leave the field.</div></div></details>
    </div>`;
    signatureShown = JSON.stringify(shot) + state.shootTier;
    bindShot(shot, take);
  }

  const thumb = (path, label) => `<button type="button" class="painted-thumb" data-view="${esc(path)}" data-name="${esc(label)}" title="Look at it closely"><img src="${actions.file(path)}" alt="" loading="lazy" draggable="false"><span>${label}</span></button>`;

  function startPictures(shot) {  // the pictures the shot's video starts and ends on, as its settings name them
    const settings = shot.generation?.settings || {}, first = value => (Array.isArray(value) ? value[0] : value) || null;
    const pictures = [[first(settings.image_start), 'Start'], [first(settings.image_end), 'End']].filter(([path]) => path);
    if (shot.source_range || !pictures.length) return '';
    return `<div class="field"><div class="label">${icon('image')} ${pictures.length > 1 ? 'Start and end pictures' : `${pictures[0][1]} picture`} <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">${pictures.length > 1 ? 'its first and last frames' : `its ${pictures[0][1] === 'Start' ? 'first' : 'last'} frame`}</span></div>
        <div class="painted-pairs">${pictures.map(([path, label]) => thumb(path, label)).join('')}</div></div>`;
  }

  function remakeFrames(shot) {  // a remake's shot: what the plan saw in it, the frames painted for it beside what they were painted from, and the user's own
    const painted = shot.painted_frames || [];
    if (!shot.source_range) return '';
    const [start, stop] = shot.source_range, frames = state.doc.remake?.frames ?? stop;
    const first = start ? start + 3 : start, last = stop < frames ? stop - 4 : stop - 1;  // as the server: a painted frame keeps 3 frames from a cut
    const own = first > last ? '<div class="muted" style="font-size:12px">Too short for a painted frame of its own: it takes the look of the shots it renders with.</div>'
      : `<div class="row"><input class="input" id="i-own-frame" type="number" min="${first}" max="${last}" step="1" value="${painted[0]?.frame ?? Math.round((first + last) / 2)}" style="width:96px" aria-label="Frame of your video">
          <a class="btn small ghost" id="i-own-source" download title="This frame of your video, to paint in any image editor">${icon('download')} Source frame</a>
          <button class="btn small" id="i-own-add" title="Your own painted picture of this frame">${icon('upload')} Use my picture…</button></div>
        <div class="muted" style="font-size:12px">Frames ${first} to ${last} of your video. Paint one in any image editor and bring it back: it takes the place of a painted frame less than 72 frames from it.</div>`;
    return `<div class="field"><div class="label">${icon('image')} Painted frames <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">what this shot should look like</span></div>
        ${painted.length ? `<div class="painted-pairs">${painted.map(item => `${item.source ? thumb(item.source, `Source, frame ${item.frame}`) : ''}${thumb(item.path, `${item.by === 'user' ? 'Yours' : 'Painted'}, frame ${item.frame}`)}`).join('')}</div>` : ''}
        ${own}</div>
      ${shot.seen ? `<details class="details"><summary>${icon('chevron')} What the plan saw in this shot</summary><div class="muted" style="margin-top:8px;font-size:12.5px;white-space:pre-wrap">${esc(shot.seen)}</div></details>` : ''}`;
  }

  function playerHtml(shot, take, compare) {
    if ((!take || sourceOn) && shot.source_range && shot.generation?.settings?.video_guide) return `<video src="${actions.sourcePart(shot)}" controls playsinline preload="metadata" poster="${shot.source_poster ? actions.file(shot.source_poster) : ''}"></video><span class="player-source">${take ? 'Source clip' : 'Source clip: no take yet'}</span>`;
    if (!take) return `<div class="ph-art" style="position:absolute;inset:0;${art(shot.id + shot.section, shot.intent)}"></div><div class="empty" style="color:rgba(255,255,255,.85)">${esc(shot.intent || 'Not shot yet')}</div>`;
    if (take.kind === 'image') return `<img src="${actions.file(take.path)}" alt="">`;
    if (take.kind === 'audio') return `<div class="empty"><audio controls src="${actions.file(take.path)}" style="width:90%"></audio></div>`;
    const loop = loopOn() ? 'loop' : '';
    const played = item => item.id === 'source' ? actions.sourcePart(shot) : shot.source_range ? actions.takePart(shot, item) : actions.file(item.path);  // a remake's take runs on past its shot's cut: the player shows what the edit keeps
    if (!compare) return `<video src="${played(take)}" controls ${loop} playsinline preload="metadata" poster="${take.poster ? actions.file(take.poster) : ''}"></video>
      <button class="btn small player-loop ${loop ? 'active' : ''}" id="i-loop" role="switch" aria-checked="${!!loop}" title="Play this video again and again">${icon('refresh')}<span>Loop</span></button>`;
    return `<video id="cmp-a" src="${played(take)}" ${loop} playsinline muted></video>
      <div class="compare" id="cmp-clip" style="clip-path:inset(0 0 0 50%)"><video id="cmp-b" src="${played(compare)}" ${loop} playsinline muted></video></div>
      <div class="compare-handle" id="cmp-handle" style="left:50%"></div>
      <span class="tag compare-label" style="left:8px">A · used</span><span class="tag compare-label" style="right:8px;left:auto;background:rgba(56,225,208,.85);color:#002b28">${compare.id === 'source' ? 'Source' : 'B'}</span>
      ${COMPARE_SEEK}
      <div class="row" style="position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:4;flex-wrap:nowrap"><button class="btn small" id="cmp-play">${icon('play')} Play both</button>
        ${compare.id === 'source' ? '' : `<button class="btn small" id="cmp-use" title="Use take B in the edit">${icon('check')} Use B</button>`}<button class="btn small" id="cmp-done">${icon('x')} Done</button></div>`;
  }

  // What was heard in a take whose model says the shot's lines
  function heardHtml(take, shot) {
    const heard = take.heard, said = heard?.text ? `Heard: “${esc(heard.text)}”` : 'No words were heard.';
    if (!heard) return '';
    const extra = [heard.before, heard.after].filter(Boolean).map(words => `“${esc(words)}”`).join(' and ');
    const notes = [
      ...heard.missing.map(line => [false, `<b>Not said as written:</b> “${esc(line)}”<br>${said}<br><b>Reshoot</b> tries another take.`]),
      ...heard.between.map(words => [false, `<b>Words added between its lines:</b> “${esc(words)}”<br>They cannot be cut without a jump: <b>Reshoot</b> tries another take.`]),
      ...heard.inexact.map(line => [true, `A word may differ in “${esc(line)}”<br>${said}`]),
      ...(extra && shot.line_cut ? [[true, `<b>Cut, as not in its lines:</b> ${extra}`]] : extra && !heard.missing.length ? [[true, `<b>Not in its lines, left in the shot:</b> ${extra}`]] : [])];
    return notes.map(([ok, text]) => `<div class="verdict-note ${ok ? 'ok' : 'bad'}">${icon(ok ? 'scissors' : 'alert')}<span>${text}</span></div>`).join('');
  }

  function checksHtml(take, shot) {
    const checks = take.checks || {};
    const chips = [];
    if (take.duration_s) chips.push(`${icon('clock')}${keptSeconds(shot, take).toFixed(2)} s`);
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
      root.querySelector('#i-count').textContent = `${state.doc.brief.kind === 'remake' ? 'what the model reads' : 'made from what happens, the camera and the cast'} · ${event.target.value.length} chars`;
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
    const setSpeed = value => actions.ops([{ op: 'update_shot', shot: id, fields: { speed: Math.min(4, Math.max(0.25, +value || 1)) } }]);
    root.querySelectorAll('[data-speed]').forEach(button => button.onclick = () => setSpeed(button.dataset.speed));
    root.querySelector('#i-speed')?.addEventListener('change', event => setSpeed(event.target.value));
    root.querySelector('#i-detach')?.addEventListener('click', () => actions.detachSound(id));
    root.querySelector('#i-reattach')?.addEventListener('click', () => actions.reattachSound(id));
    root.querySelector('#i-cut-words')?.parentElement.addEventListener('click', () => actions.ops([{ op: 'update_shot', shot: id, fields: { cut_extra_words: shot.cut_extra_words === false } }]));
    root.querySelector('#i-included')?.parentElement.addEventListener('click', () => actions.ops([{ op: 'update_shot', shot: id, fields: { included: !shot.included } }]));
    const sourceTile = root.querySelector('[data-source]');
    if (sourceTile) {
      sourceTile.onclick = () => { sourceOn = !sourceOn; compareId = null; render(); };
      sourceTile.oncontextmenu = event => {
        event.preventDefault();
        menu({ x: event.clientX, y: event.clientY }, [{ label: 'Compare with the used take', icon: 'compare', action: () => { sourceOn = false; compareId = 'source'; render(); } }]);
      };
    }
    root.querySelectorAll('[data-take]').forEach(element => {
      element.onclick = () => {
        if (sourceOn) { sourceOn = false; render(); }
        actions.ops([{ op: 'update_shot', shot: id, fields: { chosen_take: element.dataset.take } }]);
      };
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
    root.querySelector('#i-approve')?.addEventListener('click', () => actions.ops([{ op: 'verdict', shot: id, take: take.id, ok: shot.status === 'approved' ? null : true, note: '' }]));  // a second click takes the approval back
    root.querySelector('#i-reject')?.addEventListener('click', async () => {
      const note = await promptDialog({ title: 'What should change?', text: 'The showrunner reads your note before re-shooting.', placeholder: 'The hands look wrong; make the move slower…', multiline: true, ok: 'Mark for a redo' });
      if (note !== null) actions.ops([{ op: 'verdict', shot: id, take: take.id, ok: false, note }]);
    });
    root.querySelector('#i-rerun')?.addEventListener('click', () => rerun(id));
    root.querySelector('#i-own-take')?.addEventListener('click', async () => {
      const files = await pickFiles('video/*,image/*', false);
      if (files.length && (await actions.upload(files, 'take', id)).length) toast('Your file is the take of this shot.', 'ok');
    });
    if (state.doc.brief.kind === 'remake') together([id]).then(groups => {  // a click on this shot renders these with it: said before the click
      const note = root.querySelector('#i-together'), mates = (groups[0] || []).filter(other => other !== id);
      if (note && mates.length) Object.assign(note, { hidden: false, innerHTML: `${icon('layers')}<span>Renders together with ${mates.join(', ')}, in one render: short shots follow speech better that way. Each keeps its own take; one that has a take gets the new one beside it.</span>` });
    });
    bindTierToggle(root);
    root.querySelector('#i-shoot')?.addEventListener('click', () => finalTier(state.doc) ? renderNow([id], true) : actions.direct('shoot', '', [id]));
    root.querySelectorAll('[data-view]').forEach(button => button.onclick = () => showItem({ path: button.dataset.view, kind: 'image', name: button.dataset.name }));
    const ownFrame = root.querySelector('#i-own-frame');  // a remake's shot: the user's own painted frame
    if (ownFrame) {
      const source = root.querySelector('#i-own-source'), sync = () => { source.href = actions.sourceFrame(shot, ownFrame.value); };
      ownFrame.addEventListener('input', sync);
      sync();
      root.querySelector('#i-own-add').onclick = async () => {
        const files = await pickFiles('image/*', false);
        if (files.length) actions.upload(files, 'painted', `${id}:${ownFrame.value}`);
      };
    }
    root.querySelector('#i-render')?.addEventListener('click', () => renderNow([id]));
    root.querySelector('#i-fix')?.addEventListener('click', () => askForFix([id]));
    if (root.querySelector('#i-reshoot')) root.querySelector('#i-reshoot').onclick = () => askForFix([id]);
    if (root.querySelector('#i-ask')) root.querySelector('#i-ask').onclick = async () => {
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
    root.querySelector('#i-loop')?.addEventListener('click', event => {
      const on = !loopOn(), button = event.currentTarget;
      try { localStorage.setItem('sr.loop', on ? '1' : '0'); } catch { /* a convenience */ }
      root.querySelector('#i-player video').loop = on;
      button.classList.toggle('active', on);
      button.setAttribute('aria-checked', on);
    });
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
    const used = [...new Set(state.doc.shots.map(item => item.generation?.template).filter(value => value && value !== 'draw_svg' && value !== current))];  // the first choice to look for
    select.innerHTML = (known ? '' : `<option value="${esc(current)}" selected>${esc(templateLabel(current))}</option>`)
      + (used.length ? `<optgroup label="Used by the other shots">${used.map(value => `<option value="${esc(value)}">${esc(templateLabel(value))}</option>`).join('')}</optgroup>` : '')
      + groups.map(role => `<optgroup label="${esc(role.label)}">${role.choices.map(choice => `<option value="${esc(choice.value)}" ${choice.value === current ? 'selected' : ''}>${esc(choice.label)}${choice.value === (role.value || role.auto) ? ' (the brief\'s choice)' : ''}</option>`).join('')}</optgroup>`).join('')
      + `<optgroup label="Other"><option value="draw_svg" ${current === 'draw_svg' ? 'selected' : ''}>${templateLabel('draw_svg')}</option></optgroup>`;
    const rules = select.parentElement.querySelector('[data-rules]'), all = roles.flatMap(role => role.choices);  // the rules of the model picked here
    const update = () => showRules(rules, all.find(choice => choice.value === select.value));
    select.addEventListener('change', update);
    update();
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
    bindCompareSeek(root);
    const use = root.querySelector('#cmp-use');
    if (use) use.onclick = () => { const id = compareId; compareId = null; actions.ops([{ op: 'update_shot', shot: shownId, fields: { chosen_take: id } }]); };
    root.querySelector('#cmp-play').onclick = event => {
      const playing = !a.paused;
      if (playing) { a.pause(); b.pause(); } else { b.currentTime = a.currentTime = 0; a.play(); b.play(); }
      event.currentTarget.innerHTML = `${icon(playing ? 'play' : 'pause')} ${playing ? 'Play both' : 'Pause'}`;
    };
    a.ontimeupdate = () => { if (Math.abs(b.currentTime - a.currentTime) > .08) b.currentTime = a.currentTime; };
  }

  function renderMany(shots) {
    const retake = state.doc.brief.kind === 'remake' && !finalTier(state.doc) && shots.every(shot => shot.takes.length);  // as the Board's toolbar: Reshoot, not the showrunner's Shoot
    const total = shots.reduce((sum, shot) => sum + (actions.take(shot)?.duration_s || shot.duration_s || 0), 0);
    root.innerHTML = `<div class="inspector"><div class="insp-head"><b style="font-size:15px">${shots.length} shots selected</b><span class="muted">· ${formatDuration(total)}</span></div>
      <div class="refs" style="grid-template-columns:repeat(3,1fr)">${shots.slice(0, 9).map(shot => { const take = actions.take(shot); return `<div class="ref" style="aspect-ratio:16/9">${take?.poster ? `<img src="${actions.file(take.poster)}" alt="">` : `<div style="position:absolute;inset:0;${art(shot.id + shot.section, shot.intent)}"></div>`}</div>`; }).join('')}</div>
      <div class="col">
        <button class="btn approve" data-bulk="approve" ${availability(shots.some(shot => actions.take(shot)) ? '' : 'None of them has a take yet.')}>${icon('check')} Approve their takes</button>
        ${state.doc.brief.draft_resolution ? `<div class="row" style="flex-wrap:nowrap">${tierToggle(state.doc)}${retake ? `<button class="btn grow" data-bulk="reshoot-now" ${availability(actions.renderBlocked(), `Another take of each now at ${reshootSize(state.doc)}, with the same prompt and pictures and a new seed, without the showrunner`)}>${icon('refresh')} Reshoot ${shots.length} shots</button>` : `<button class="btn grow" data-bulk="shoot" ${availability(finalTier(state.doc) ? actions.renderBlocked() || (notFinal(state.doc, shots).length ? '' : 'They already have a final take.') : actions.directorBlocked(), finalTier(state.doc) ? 'Renders them at the target size now, with the prompt, seed and pictures of their takes: no language model, and nobody looks at the takes' : 'The showrunner renders them at the draft size and looks at the takes')}>${icon(finalTier(state.doc) ? 'bolt' : 'clapper')} Shoot ${shots.length} shots</button>`}</div>` : ''}
        <button class="btn" data-bulk="lock">${icon('lock')} Lock / unlock</button>
        <button class="btn" data-bulk="include">${icon('eye')} Include / exclude</button>
        <button class="btn" data-bulk="reshoot" ${availability(actions.directorBlocked())}>${icon('refresh')} Ask the showrunner to fix them…</button>
        <button class="btn" data-bulk="render" ${availability(actions.renderBlocked() || (shots.some(shot => !shot.takes.length) ? '' : 'All of them have a take.'), 'Renders the ones without a take now, without the showrunner: no language model, and nobody checks the takes')}>${icon('bolt')} Render now</button>
        <div class="field"><div class="label">${icon('layers')} Move to section</div><input class="input" id="bulk-section" placeholder="The part of the video they belong to"></div>
        <div class="field"${state.doc.brief.kind === 'remake' ? ' style="display:none"' : ''}><div class="label">${icon('transition')} Transition</div><div class="seg" id="bulk-tr">${TRANSITIONS.map(([key, label]) => `<button data-tr="${key}">${label}</button>`).join('')}</div></div>
        ${retake && state.doc.brief.draft_resolution ? '' : `<button class="btn" data-bulk="reshoot-now" ${availability(actions.renderBlocked() || (shots.some(shot => shot.takes.length) ? '' : 'None of them has a take yet: Render now makes their first.'), `Another take of each one that has a take, now at ${reshootSize(state.doc)}, with the same prompt and pictures and a new seed, without the showrunner`)}>${icon('refresh')} Reshoot them</button>`}
        <div class="together-note" id="bulk-together" hidden></div>
        <button class="btn danger" data-bulk="delete">${icon('trash')} Delete</button></div></div>`;
    bindTierToggle(root);
    if (state.doc.brief.kind === 'remake') together(shots.map(shot => shot.id)).then(groups => {  // what one click renders, as renders
      const note = root.querySelector('#bulk-together'), joined = groups.filter(group => group.length > 1);
      if (note && joined.length) Object.assign(note, { hidden: false, innerHTML: `${icon('layers')}<span>${groups.length === 1 ? `They render together, in one render: ${groups[0].join(' + ')}.` : `They make ${groups.length} renders: ${groups.map(group => group.join(' + ')).join(', ')}.`}</span>` });
    });
    root.querySelectorAll('[data-bulk]').forEach(button => button.onclick = () => {
      const kind = button.dataset.bulk;
      if (kind === 'approve') actions.ops(shots.filter(shot => actions.take(shot)).map(shot => ({ op: 'verdict', shot: shot.id, ok: true, note: '' })));
      if (kind === 'shoot') finalTier(state.doc) ? renderNow(notFinal(state.doc, shots).map(shot => shot.id), true) : actions.direct('shoot', '', shots.map(shot => shot.id));
      if (kind === 'lock') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { locked: !shots.every(item => item.locked) } })));
      if (kind === 'include') actions.ops(shots.map(shot => ({ op: 'update_shot', shot: shot.id, fields: { included: !shots.every(item => item.included) } })));
      if (kind === 'reshoot') askForFix(shots.map(shot => shot.id));
      if (kind === 'render') renderNow(shots.filter(shot => !shot.takes.length).map(shot => shot.id));
      if (kind === 'reshoot-now') reshoot(shots.filter(shot => shot.takes.length).map(shot => shot.id));
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
          ${doc.brief.review_before_edit ? check(doc.shots.length && count('approved') === doc.shots.length, `${count('approved')} approved by you`) : ''}${check(doc.deliverables.length, doc.deliverables.length ? 'Edit assembled' : 'Edit not assembled yet')}
        </div></div>
      <div class="three" style="gap:8px">${stat(count('rendered'), doc.brief.review_before_edit ? 'to review' : 'shot', 'var(--amber)')}${stat(count('rejected') + count('failed'), 'to redo', 'var(--red)')}${stat(minutes.toFixed(1), 'GPU min', 'var(--cyan)')}</div>
      <details class="panel details" style="padding:12px 16px"><summary class="label" style="margin:0;cursor:pointer">${icon('keyboard')} Keyboard shortcuts</summary>
        <div class="col" style="gap:6px;font-size:12.5px;margin-top:10px">${[['←→', 'Select shot'], ['A', 'Approve / unapprove'],['X', 'Redo, with a note'], ['L', 'Lock'], ['E', 'Include / exclude'], ['N', 'New shot'], ['Del', 'Delete'], ['Space', 'Play (timeline)'], ['1-5', 'Views'], ['D', 'Showrunner'], ['Ctrl+Z', 'Undo'], ['?', 'Guide']].map(([key, text]) => `<div class="row"><span class="kbd">${key}</span><span class="muted">${text}</span></div>`).join('')}</div></details>
      <p class="muted" style="font-size:12.5px;margin:0">Select a shot to watch its takes, judge them and edit what the showrunner will use. Drop media from your computer or the bin onto a shot to make it a take.</p>
    </div>`;
  }

  return {
    update() {
      const active = document.activeElement;
      if (active && root.contains(active) && active.matches('textarea, input, select')) return;  // no redraw under the user's hands
      const id = state.selection.size === 1 ? [...state.selection][0] : null;
      const shot = id && actions.shot(id);
      if (shot && shot.id === shownId && JSON.stringify(shot) + state.shootTier === signatureShown && !root.querySelector('#cmp-handle')) return;
      render();
    },
  };
}

// The renders a click on these shots makes, as the server plans them (remake_groups): fetched once a revision and selection.
let groupsAsked = { key: '', value: Promise.resolve([]) };
function together(ids) {
  const key = `${state.pid}:${state.doc.revision}:${ids.join(',')}`;
  if (groupsAsked.key !== key) groupsAsked = { key, value: api.renderGroups(state.pid, ids).then(result => result.groups).catch(() => []) };
  return groupsAsked.value;
}

// Why the shot's last render failed: the newest log line about it (New take and render write the error there).
function failure(shot, doc) {
  const entry = [...doc.log].reverse().find(item => new RegExp(`\\b${shot.id}\\b`).test(item.text || '') && /fail/i.test(item.text || ''));
  return `<div class="verdict-note bad">${icon('alert')}<span><b>Failed:</b> ${entry ? shotRefs(entry.text, doc.shots, doc.cast) : 'the render returned nothing; ask the showrunner or try again.'}</span></div>`;
}

// How fast the take plays in the edit; its sound follows, at the same pitch. A shot on the song leaves its beat and its lips the song's words.
function speedField(shot, doc) {
  const speed = +(shot.speed || 1), seg = segments(doc, actions.take).find(item => item.shot.id === shot.id);
  const onSong = doc.brief.song && doc.edit.audio_mode !== 'clips' && shot.audio_slice;
  return `<div class="field"><div class="label">${icon('gauge')} Speed ${seg ? `<span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">plays for ${formatDuration(seg.duration)} in the edit</span>` : ''}</div>
    <div class="row"><div class="seg">${[0.5, 0.75, 1, 1.5, 2].map(value => `<button class="${value === speed ? 'on' : ''}" data-speed="${value}">${value}×</button>`).join('')}</div>
      <input class="input" id="i-speed" type="number" min="0.25" max="4" step="0.05" value="${speed}" style="width:74px" title="From 0.25× to 4×"></div>
    ${speed < 0.75 ? '<div class="muted" style="font-size:12px">Slow motion repeats frames: the movement can look jerky.</div>' : ''}
    ${speed !== 1 && onSong ? `<div class="verdict-note bad">${icon('alert')}<span>This shot was made on its part of the song: at another speed, its lips and its cuts leave the song.</span></div>` : ''}</div>`;
}

// The take's sound in the edit: its own, or a track of its own on the Timeline's Sound lane.
function soundField(shot, take, doc) {
  const tracks = doc.edit.audio_tracks.filter(track => track.shot === shot.id);
  const other = tracks.length && tracks.every(track => track.take !== take.id);
  return `<div class="field"><div class="label">${icon('audio')} Sound</div>
    ${shot.sound !== false ? `<div class="row"><button class="btn small" id="i-detach" title="Makes its sound a track on the Timeline's Sound lane, to shorten, extend, move or fade apart from the picture">${icon('scissors')} Detach Sound</button></div>`
      : `<div class="row"><span class="muted grow" style="font-size:12.5px">${tracks.length ? `A track on the Sound lane${other ? ', from another take' : ''}` : 'This shot plays silent'}</span><button class="btn small" id="i-reattach" title="${tracks.length ? 'Removes its track from the Sound lane; the shot plays the sound of its take again' : 'The shot plays the sound of its take again'}">${icon('undo')} Reattach</button></div>`}</div>`;
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
