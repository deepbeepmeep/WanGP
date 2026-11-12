// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Brief: what to make, from which material, in which format and style. Every change saves itself.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, fieldSaver, drawWave, formatTime, formatDuration, sectionColor, pickFiles, toast, confirmDialog, promptDialog, availability, menu, AUDIO_ACCEPT, personFields } from './ui.js';
import { kindTiles, KINDS } from './home.js';
import { bindRules, modelRows, openDefaultModels } from './models.js';

const STYLES = ['Cinematic', 'Anime', 'Documentary', 'Noir', 'Neon', 'Pastel', 'Claymation', 'Watercolor', 'Vintage film', 'VHS', 'Epic', 'Minimal', 'Dreamy', 'Gritty', 'Surreal', 'Cyberpunk'];
const ASPECT_FRAMES = { '16:9': [44, 25], '9:16': [20, 34], '1:1': [30, 30], '4:3': [38, 28], '2.39:1': [50, 21] };
const PACING = ['Very calm', 'Calm', 'Balanced', 'Energetic', 'Frenetic'];
const RESOLUTION_CHOICES = ['480p', '720p', '1080p'];
const LANGUAGES = [['en', 'English'], ['fr', 'French'], ['es', 'Spanish'], ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'], ['ru', 'Russian'], ['tr', 'Turkish'], ['ar', 'Arabic'], ['hi', 'Hindi'], ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean']];

// A format chosen after shooting changes nothing by itself: the note says so, with the button that does it.
function formatNote(doc) {
  const latest = doc.deliverables.at(-1), [width, height] = doc.brief.aspect.split(':').map(Number);
  if (!latest?.width || Math.abs(latest.width / latest.height - width / height) < 0.08) return '';
  return `<div class="stale-note" style="margin-top:8px">${icon('alert')}<span class="grow">Your last version is ${latest.width}×${latest.height}: this choice changes nothing until the shots are made again in ${esc(doc.brief.aspect)}.</span><button class="btn small" id="make-format" ${availability(actions.directorBlocked())}>Make a ${esc(doc.brief.aspect)} version</button></div>`;
}

export function mountBrief(container, doc) {
  container.innerHTML = '<div class="brief" id="brief"></div>';
  const root = container.querySelector('#brief');
  const save = fieldSaver((_, fields) => actions.ops([{ op: 'brief', fields }], { quiet: true }).catch(() => {}), 450);
  const queue = fields => save('brief', fields);
  let audio = null, audioSong = null, analysisRequested = false;

  function render() {
    const doc = state.doc, brief = doc.brief;
    if (audio && audioSong !== brief.song) { audio.pause(); audio = null; }  // a new song (a new voice, another file) plays from its own file
    const kind = KINDS[brief.kind] || KINDS.free;
    const narrative = ['short_film', 'trailer', 'story', 'music_video'].includes(brief.kind), lyrics = brief.kind === 'music_video';  // a music video's script is its lyrics
    const analysis = doc.audio?.analysis;
    root.innerHTML = `
      <section class="panel"><div class="panel-head"><span class="num">1</span><h3>Format</h3><span class="hint">Shapes how the showrunner plans and cuts</span></div>
        <div class="kind-row">${kindTiles(brief.kind)}</div></section>

      <section class="panel"><div class="panel-head"><span class="num">2</span><h3>Your idea</h3><span class="hint">Plain language; the showrunner develops short ideas and keeps detailed ones</span></div>
        <div class="${narrative ? 'two' : ''}">
          <div class="field"><div class="label">${icon('sparkles')} Concept</div>
            <textarea class="textarea" id="text" rows="${narrative ? 9 : 5}" placeholder="${esc(placeholder(brief.kind))}">${esc(brief.text)}</textarea></div>
          ${narrative ? `<div class="field"><div class="label">${icon(lyrics ? 'music' : 'note')} ${lyrics ? 'Lyrics' : 'Script'} <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">${lyrics ? 'optional: the song\'s words, for subtitles and lip sync' : 'optional, exact words kept'}</span></div>
            <textarea class="textarea mono" id="script" rows="9" placeholder="${lyrics ? '[Verse 1]\nThe sun climbs up over the old town square…' : 'NARRATOR: Once, at the edge of the sea…\nMIA: Did you hear that?'}">${esc(brief.script)}</textarea></div>` : ''}
        </div></section>

      ${brief.kind === 'remake' ? `<section class="panel"><div class="panel-head"><span class="num">3</span><h3>The source video</h3><span class="hint">Its motion, camera, timing and sound stay</span></div><div id="source">${brief.source ? `<div class="row" style="gap:10px"><video src="${actions.file(brief.source)}" controls preload="metadata" style="max-width:320px;border-radius:8px"></video><button class="btn small" id="remove-source">${icon('trash')} Remove</button></div>` : dropzone('source', 'video/*', 'Drop the video here', 'or click to choose a file. Change only footage you have the rights to, and never put a real person into a video without their agreement.')}</div>
        <div class="field" style="margin-top:12px"><div class="label">${icon('clock')} Shots rendered together <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">short shots render with their neighbours: alone, the lips barely follow the speech</span></div>
          <div class="slider-row" style="justify-content:flex-start"><span class="muted">at least</span><input class="input" id="render-min" type="number" min="0" max="15" step="1" value="${brief.render_min_s ?? 0}" aria-label="Shortest render in seconds" title="Consecutive shots join until a render lasts this long; 0 renders every shot alone" style="width:76px;flex:none;padding:6px 8px"><span class="muted">to</span>
            <input class="input" id="render-max" type="number" min="1" max="15" step="1" value="${brief.render_max_s ?? ''}" placeholder="Auto" aria-label="Longest render in seconds" title="The longest render of shots together; Auto is the model's window" style="width:76px;flex:none;padding:6px 8px"><span class="muted">s</span></div></div></section>` : `<section class="panel"><div class="panel-head"><span class="num">3</span><h3>${brief.kind === 'music_video' ? 'The song' : 'Soundtrack'}</h3><span class="hint">${brief.kind === 'music_video' ? 'Drives timing, cuts and lip sync' : 'Optional: music the edit follows'}</span></div>
        <div id="song">${brief.song ? songCard(doc, analysis) : dropzone('song', AUDIO_ACCEPT + ',video/*', 'Drop the song here', `or click to choose a file · WAV, MP3, FLAC, M4A, or a video to take its audio${brief.kind === 'music_video' ? '. No song yet? Leave this empty: the showrunner composes one from your idea and lyrics.' : ''}`)}</div>
        ${!brief.song && brief.kind === 'music_video' ? sungBy(doc) : ''}</section>`}

      <section class="panel"><div class="panel-head"><span class="num">4</span><h3>References</h3><span class="hint">Looks, places, moods, motion: images or videos</span></div>
        <div class="refs">${brief.references.map((path, index) => reference(path, index)).join('')}
          <div class="dropzone" data-drop="reference" style="aspect-ratio:1;flex-direction:column;justify-content:center;text-align:center;padding:10px" title="Add references">${icon('plus')}<span style="font-size:12px">Add</span></div></div></section>

      <section class="panel"><div class="panel-head"><span class="num">5</span><h3>Cast</h3><span class="hint">Drop a photo per person to keep them consistent; the showrunner makes the rest</span>
        <button class="btn small" id="to-cast">${icon('users')} Open cast</button></div>
        <div class="refs">${doc.cast.map(member => castTile(member)).join('')}
          <div class="dropzone" data-drop="cast" style="aspect-ratio:1;flex-direction:column;justify-content:center;text-align:center;padding:10px">${icon('plus')}<span style="font-size:12px">Add a person</span></div></div></section>

      <section class="panel"><div class="panel-head"><span class="num">6</span><h3>Look and format</h3></div>
        <div class="col" style="gap:18px">
          <div class="field"><div class="label">${icon('ratio')} Aspect ratio</div><div class="aspects">${Object.entries(ASPECT_FRAMES).map(([aspect, [width, height]]) =>
            `<button class="aspect ${brief.aspect === aspect ? 'on' : ''}" data-aspect="${aspect}"><i style="width:${width}px;height:${height}px"></i>${aspect}<span class="muted" style="font-size:10.5px">${{ '16:9': 'Widescreen', '9:16': 'Vertical', '1:1': 'Square', '4:3': 'Classic', '2.39:1': 'Scope' }[aspect]}</span></button>`).join('')}</div>${formatNote(state.doc)}</div>
          <div class="three">
            <div class="field"><div class="label">${icon('image')} Target resolution</div><div class="seg" id="res" title="What the final shots and every picture are made at">${RESOLUTION_CHOICES.map(value => `<button class="${brief.resolution === value ? 'on' : ''}" data-res="${value}">${value}</button>`).join('')}</div></div>
            <div class="field"><div class="label">${icon('clock')} Draft resolution</div><div class="seg" id="draft-res" title="Try shots at this size first, then Shoot set to Final renders the ones you like at the target resolution. The pictures stay the same">${['', ...RESOLUTION_CHOICES.slice(0, RESOLUTION_CHOICES.indexOf(brief.resolution))].map(value => `<button class="${(brief.draft_resolution || '') === value ? 'on' : ''}" data-draft="${value}">${value || 'None'}</button>`).join('')}</div></div>
            <div class="field"><div class="label">${icon('clock')} Duration ${brief.song && analysis ? `<button class="chip clickable ${!brief.duration_s ? 'on' : ''}" id="match-song" style="height:20px;margin-left:4px">${icon('music')}Match the song</button>` : ''}</div>
              <div class="slider-row"><input type="range" class="range" id="duration" min="5" max="600" step="5" value="${brief.duration_s || analysis?.duration_s || 30}" aria-label="Duration">
                <input class="input" id="duration-n" type="number" min="5" max="600" step="1" value="${Math.round(brief.duration_s || analysis?.duration_s || 30)}" aria-label="Duration in seconds" title="Seconds, from 5 to 600" style="width:76px;flex:none;padding:6px 8px"><span class="muted">s</span></div></div>
            <div class="field"><div class="label">${icon('gauge')} Pacing</div>
              <div class="slider-row"><input type="range" class="range" id="pacing" min="-2" max="2" step="1" value="${brief.pacing || 0}"><span class="value" id="pacing-v">${PACING[(brief.pacing || 0) + 2]}</span></div></div>
          </div>
          <div class="field"><div class="label">${icon('clock')} Shot length <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">empty lets the showrunner choose; long shots take longer to render</span></div>
            <div class="slider-row" style="justify-content:flex-start"><input class="input" id="shot-min" type="number" min="1" max="60" step="1" value="${brief.shot_min_s ?? ''}" placeholder="Auto" aria-label="Shortest video shot in seconds" title="The shortest video shot, in seconds" style="width:76px;flex:none;padding:6px 8px"><span class="muted">to</span>
              <input class="input" id="shot-max" type="number" min="1" max="60" step="1" value="${brief.shot_max_s ?? ''}" placeholder="Auto" aria-label="Longest video shot in seconds" title="The longest video shot, in seconds" style="width:76px;flex:none;padding:6px 8px"><span class="muted">s</span></div></div>
          <div class="field"><div class="label">${icon('palette')} Style</div>
            <div class="styles">${STYLES.map(style => `<button class="chip clickable ${brief.style_tags.includes(style) ? 'on' : ''}" data-style="${style}">${style}</button>`).join('')}</div>
            <input class="input" id="style" style="margin-top:10px" placeholder="Anything else about the look: lighting, era, references, palette…" value="${esc(brief.style)}"></div>
        </div></section>

      <section class="panel"><div class="panel-head"><span class="num">7</span><h3>Directing</h3></div>
        <div class="two">
          <label class="row" style="gap:12px;cursor:pointer"><span class="toggle ${brief.plan_first ? 'on' : ''}" id="plan-first"></span><span><b>Review the plan before shooting</b><br><span class="muted">The showrunner stops after planning; you edit the storyboard, then press Shoot.</span></span></label>
          <label class="row" style="gap:12px;cursor:pointer"><span class="toggle ${brief.review_before_edit ? 'on' : ''}" id="review-edit"></span><span><b>Review the takes before the edit</b><br><span class="muted">The showrunner shoots and checks the shots, then waits for you: it assembles the edit only once you have approved every take. Assemble on the Board, the Timeline or the showrunner's panel works at any time.</span></span></label>
          <label class="row" style="gap:12px;cursor:pointer"><span class="toggle ${brief.read_other_productions ? 'on' : ''}" id="read-others"></span><span><b>Let the showrunner look at my other productions</b><br><span class="muted">When you ask for it ("make it like my other film", "the same characters as in Lanterns"), it reads them and copies the files it needs (pictures, voices) into this production; it never changes them. Off, it only knows this production.</span></span></label>
          <div class="field"><div class="label">${icon('bolt')} GPU budget</div><div class="slider-row"><input type="range" class="range" id="budget" min="0" max="240" step="5" value="${brief.budget_minutes || 0}"><span class="value" id="budget-v">${brief.budget_minutes ? `${brief.budget_minutes} min` : 'Auto'}</span></div></div>
          ${['music_video', 'story'].includes(brief.kind) ? '' : `<div class="field" style="grid-column:1/-1"><div class="label">${icon('chat')} Voices</div>
            <div class="row" style="gap:10px;flex-wrap:wrap"><div class="seg" id="voices"><button class="${brief.voices_from_model ? '' : 'on'}" data-voices="">Recorded first</button><button class="${brief.voices_from_model ? 'on' : ''}" data-voices="1">From the video model</button></div>
              ${brief.voices_from_model ? `<select class="select" id="spoken-language" aria-label="Language of the lines" title="The language of the lines" style="width:auto">${LANGUAGES.map(([code, name]) => `<option value="${code}" ${brief.language === code ? 'selected' : ''}>${name}</option>`).join('')}</select>` : ''}</div>
            <div class="muted" style="font-size:12.5px;margin-top:6px">${brief.voices_from_model ? 'The video model says the lines itself, with its own sound. Nothing is recorded first, but a speaker\'s voice can change from shot to shot.' : 'Each line is recorded first in its speaker\'s voice, and the video follows it: a speaker keeps one voice for the whole production.'} <span id="voices-models" style="color:var(--amber)"></span></div></div>`}
        </div></section>

      <section class="panel"><div class="panel-head clickable" id="models-head" title="Show or hide the models"><span class="num">8</span><h3>Models</h3>
        <span class="hint" id="models-hint">Auto: the showrunner picks the best model for each shot</span><span class="fold">${icon(modelsOpen() ? 'chevronDown' : 'chevron')}</span></div>
        <div class="models" id="models" ${modelsOpen() ? '' : 'hidden'}><div class="muted">Loading the models…</div></div></section>

      ${state.doc.deliverables.length ? `<div class="muted" style="font-size:12.5px;margin:4px 2px 0">This brief is where the production started: later changes live in the shots, the cast and the edit, and it is not rewritten. To keep going from where the production is, talk to the showrunner.</div>` : ''}
      <div class="cta-bar">
        <div class="grow"><b>${esc(kind.label)}</b> <span class="muted">· ${esc(brief.aspect)} · ${brief.draft_resolution ? `${brief.draft_resolution} draft, ` : ''}${brief.resolution} · <span id="brief-duration">${formatDuration(brief.duration_s || analysis?.duration_s || 30)}</span> · ${doc.cast.length} cast · ${brief.references.length} ${brief.references.length === 1 ? 'reference' : 'references'}</span>
          ${state.directorInfo.available === false && state.directorInfo.reason ? `<div class="muted" style="font-size:12px;color:var(--amber)">${esc(state.directorInfo.reason)}</div>` : ''}</div>
        ${doc.shots.length ? `<button class="btn" id="to-board">${icon('grid')} Storyboard (${doc.shots.length})</button>` : ''}
        ${!(brief.text || '').trim() && !(brief.script || '').trim() && !brief.song ? '<span class="muted" style="font-size:12.5px">Write your idea first</span>' : ''}
        <button class="btn primary lg" id="plan" ${availability(actions.directorBlocked() || actions.planBlocked(doc))}>${icon('wand')} ${actions.busy(doc) || state.live.running ? 'The showrunner is working…' : (doc.shots.length ? 'Plan again' : 'Plan with the showrunner') + (doc.brief.plan_first ? '' : ' and shoot')}</button>
      </div>`;
    bind();
    renderModels();
    if (brief.song) drawSong();
    if ((brief.song || brief.kind === 'music_video') && state.singerVoice === undefined) {
      state.singerVoice = null;
      refreshSingerVoice();
    }
    if (brief.song && !analysisRequested) {  // cheap when cached; recomputed when the song or the analysis changed
      analysisRequested = true;
      api.analysis(doc.id).catch(error => toast(error.message, 'error'));
    }
  }

  function refreshSingerVoice() {
    if (!state.doc.brief.song && state.doc.brief.kind !== 'music_video') return;
    api.singerVoice().then(data => {
      const changed = state.singerVoice !== data.unavailable;
      state.singerVoice = data.unavailable;
      if (changed && root.isConnected) render();
    }).catch(() => { state.singerVoice = undefined; });
  }

  function bind() {
    root.querySelectorAll('[data-kind]').forEach(button => button.onclick = () => actions.ops([{ op: 'brief', fields: { kind: button.dataset.kind } }]).then(render));
    const ready = () => {  // the Plan button follows the idea as it is typed
      const filled = root.querySelector('#text').value.trim() || root.querySelector('#script')?.value.trim() || state.doc.brief.song;
      const button = root.querySelector('#plan');
      const blocked = actions.directorBlocked() || (filled ? '' : actions.planBlocked({ brief: {} }));
      button.disabled = !!blocked;
      button.title = blocked;
      button.previousElementSibling?.classList.contains('muted') && (button.previousElementSibling.style.display = filled ? 'none' : '');
    };
    root.querySelector('#text').oninput = event => { queue({ text: event.target.value }); ready(); };
    root.querySelector('#script')?.addEventListener('input', event => { queue({ script: event.target.value }); ready(); });
    root.querySelector('#style').oninput = event => queue({ style: event.target.value });
    root.querySelectorAll('[data-aspect]').forEach(button => button.onclick = () => actions.ops([{ op: 'brief', fields: { aspect: button.dataset.aspect } }]).then(render));
    root.querySelector('#make-format')?.addEventListener('click', () => actions.makeFormat(state.doc.brief.aspect));
    root.querySelectorAll('[data-res]').forEach(button => button.onclick = () => {
      const lower = RESOLUTION_CHOICES.indexOf(state.doc.brief.draft_resolution) < RESOLUTION_CHOICES.indexOf(button.dataset.res);
      actions.ops([{ op: 'brief', fields: { resolution: button.dataset.res, ...(lower ? {} : { draft_resolution: null }) } }]).then(render);
    });
    root.querySelectorAll('[data-draft]').forEach(button => button.onclick = () => actions.ops([{ op: 'brief', fields: { draft_resolution: button.dataset.draft || null } }]).then(render));
    root.querySelectorAll('[data-style]').forEach(button => button.onclick = () => {
      const tags = new Set(save.pending('brief')?.style_tags ?? state.doc.brief.style_tags);  // the chips clicked a moment ago count too
      tags.has(button.dataset.style) ? tags.delete(button.dataset.style) : tags.add(button.dataset.style);
      button.classList.toggle('on');
      queue({ style_tags: [...tags] });
    });
    // The slider (mouse or keys), the typed seconds and the summary bar show one length, whichever was changed
    const duration = root.querySelector('#duration'), typed = root.querySelector('#duration-n');
    const setDuration = seconds => { root.querySelector('#brief-duration').textContent = formatDuration(seconds); root.querySelector('#match-song')?.classList.remove('on'); queue({ duration_s: seconds }); };
    duration.oninput = () => { typed.value = duration.value; setDuration(+duration.value); };
    typed.oninput = () => { if (typed.value !== '' && +typed.value >= 5 && +typed.value <= 600) { duration.value = typed.value; setDuration(+typed.value); } };  // out of range: kept until the field is left
    typed.onchange = () => { const seconds = Math.min(600, Math.max(5, Math.round(+typed.value || +duration.value))); typed.value = duration.value = seconds; setDuration(seconds); };
    root.querySelector('#match-song')?.addEventListener('click', () => actions.ops([{ op: 'brief', fields: { duration_s: null } }]).then(render));
    const pacing = root.querySelector('#pacing');
    pacing.oninput = () => { root.querySelector('#pacing-v').textContent = PACING[+pacing.value + 2]; queue({ pacing: +pacing.value }); };
    [['#render-min', 'render_min_s', 0], ['#render-max', 'render_max_s', null]].forEach(([selector, field, empty]) => {  // a remake's shots rendered together
      const input = root.querySelector(selector);
      if (input) input.onchange = () => queue({ [field]: input.value === '' ? empty : Math.min(15, Math.max(0, +input.value)) || empty });
    });
    [['#shot-min', 'shot_min_s'], ['#shot-max', 'shot_max_s']].forEach(([selector, field]) => {
      const input = root.querySelector(selector);
      input.oninput = () => { if (input.value === '' || (+input.value >= 1 && +input.value <= 60)) queue({ [field]: +input.value || null }); };  // out of range: kept until the field is left
      input.onchange = () => { if (input.value !== '') input.value = Math.min(60, Math.max(1, Math.round(+input.value))); queue({ [field]: +input.value || null }); };
    });
    const budget = root.querySelector('#budget');
    budget.oninput = () => { root.querySelector('#budget-v').textContent = +budget.value ? `${budget.value} min` : 'Auto'; queue({ budget_minutes: +budget.value || null }); };
    root.querySelectorAll('[data-voices]').forEach(button => button.onclick = () => actions.ops([{ op: 'brief', fields: { voices_from_model: !!button.dataset.voices } }]).then(render));
    root.querySelector('#spoken-language')?.addEventListener('change', event => actions.ops([{ op: 'brief', fields: { language: event.target.value } }]).then(render));
    root.querySelector('#plan-first').parentElement.onclick = () => actions.ops([{ op: 'brief', fields: { plan_first: !state.doc.brief.plan_first } }]).then(render);
    root.querySelector('#read-others').parentElement.onclick = () => actions.ops([{ op: 'brief', fields: { read_other_productions: !state.doc.brief.read_other_productions } }]).then(render);
    root.querySelector('#review-edit').parentElement.onclick = () => actions.ops([{ op: 'brief', fields: { review_before_edit: !state.doc.brief.review_before_edit } }]).then(render);
    root.querySelector('#to-cast').onclick = () => actions.go('cast');
    root.querySelectorAll('[data-cast-tile]').forEach(tile => tile.onclick = () => actions.go('cast'));
    root.querySelector('#models-head').onclick = () => { try { localStorage.setItem('sr.models', modelsOpen() ? '0' : '1'); } catch { /* a convenience */ } render(); };
    root.querySelector('#to-board')?.addEventListener('click', () => actions.go('storyboard'));
    root.querySelector('#plan').onclick = async () => {
      save.flush();
      if (state.doc.shots.length && !(await confirmDialog({ title: 'Plan the production again?', text: 'The showrunner replaces the shots you have not edited, locked or shot. Your locked and shot shots stay.', ok: 'Plan again' }))) return;
      if (await actions.direct('plan')) actions.go('storyboard');
    };
    root.querySelectorAll('[data-drop]').forEach(zone => dropTarget(zone, zone.dataset.drop));
    const removeSource = root.querySelector('#remove-source');
    if (removeSource) removeSource.onclick = () => actions.ops([{ op: 'brief', fields: { source: null } }]).then(render);
    root.querySelectorAll('[data-remove-ref]').forEach(button => button.onclick = () => {
      const references = state.doc.brief.references.filter((_, index) => index !== +button.dataset.removeRef);
      actions.ops([{ op: 'brief', fields: { references } }]).then(render);
    });
    root.querySelector('#song-remove')?.addEventListener('click', () => { audio?.pause(); actions.ops([{ op: 'brief', fields: { song: null } }, { op: 'audio', analysis: null }]).then(render); });
    root.querySelector('#song-voice')?.addEventListener('click', event => menu(event.currentTarget, [
      ...state.doc.cast.filter(member => !isPlace(member)).map(member => ({
        label: `${member.name || member.id}'s voice`, icon: 'users', action: () => changeVoice([member.id]),
        unavailable: hasVoice(member) ? '' : 'Give them a voice sample, or describe their voice, on the Cast page' })),
      { label: 'A voice sample from my computer…', icon: 'upload', action: async () => {
        const files = await pickFiles(AUDIO_ACCEPT, false);
        if (!files.length) return;
        const [sample] = await actions.upload([...files].slice(0, 1), 'bin');
        if (sample?.path) changeVoice([sample.path]);
      } },
      { label: 'Two singers (a duet)…', icon: 'users', action: () => duetDialog([...castSingers(),
        ...state.doc.bin.filter(item => item.kind === 'audio' && item.path !== state.doc.brief.song).map(item => ({ value: item.path, label: `Media bin: ${item.name || item.path.split('/').pop()}` }))], changeVoice) },
    ]));
    root.querySelector('#song-voices')?.addEventListener('change', event => {  // who sings the song the showrunner composes
      const setSingers = singers => actions.ops([{ op: 'brief', fields: { song_voices: singers } }]).then(render);
      if (event.target.value !== 'duet') return setSingers(event.target.value ? event.target.value.split(',') : []);
      render();  // the list shows the current choice again while the dialog asks for the duet
      duetDialog(castSingers(), setSingers, 'Choose');
    });
    root.querySelector('#song-play')?.addEventListener('click', togglePlay);
    root.querySelector('.song-wave')?.addEventListener('click', event => {
      const rect = event.currentTarget.getBoundingClientRect();
      ensureAudio();
      audio.currentTime = (event.clientX - rect.left) / rect.width * (audio.duration || state.doc.audio.analysis?.duration_s || 0);
      if (audio.paused) togglePlay();
    });
  }

  // Models: one picker per Deepy template category, Auto being Deepy's selection; a pick moves the shots still on the previous one.
  async function renderModels() {
    const box = root.querySelector('#models');
    let roles;
    try { roles = await actions.templateRoles(); } catch (error) { box.innerHTML = `<div class="muted">${esc(error.message)}</div>`; return; }
    if (!root.contains(box)) return;
    voicesModels(roles);
    const kind = state.doc.brief.kind, picked = roles.filter(role => role.value).length;
    const hint = root.querySelector('#models-hint');
    if (hint) hint.textContent = picked ? `${picked} picked by you · the others on Auto` : "Auto: the templates selected in Deepy's settings";
    const editing = ['people_edits', 'scene_edits', 'frame_painter'], used = role => kind === 'remake' ? ['image_edits', 'frame_painter'].includes(role.id) : !editing.includes(role.id)
      && !(kind === 'music_video' && ['voices', 'cloned_voices'].includes(role.id)) && !(role.id === 'music' && state.doc.brief.song);  // kinds this production makes
    box.innerHTML = modelRows(roles.filter(role => used(role) || role.value))  // a remake's shots are Viggle Remake's: their templates show only when picked before, to be reset
      + '<div class="muted" style="grid-column:1/-1;font-size:12px">Auto is the template selected in Deepy\'s settings for the category (the Frame Painter\'s is H3-Image, which keeps each person\'s pose and expression); add your own templates there (Ask Deepy &gt; Settings). A pick applies to the shots of its category that still use the previous template; the shot panel changes one shot. The notes button beside a list holds your rules for the prompts written with its model. <button class="link" id="models-defaults">Default models</button> sets what new productions start with.</div>';
    box.querySelector('#models-defaults').onclick = () => openDefaultModels();
    bindRules(box, roles);
    box.querySelectorAll('[data-role]').forEach(select => select.onchange = async () => {
      try {
        await api.chooseTemplate(state.doc.id, select.dataset.role, select.value || null);
        voicesModels(await actions.templateRoles(true));
        toast(select.value ? `${select.selectedOptions[0].textContent} for ${roles.find(role => role.id === select.dataset.role).label.toLowerCase()}` : 'Back to the showrunner\'s choice');
      } catch (error) { toast(error.message, 'error'); renderModels(); }
    });
  }

  // What the Voices choice does with the models of this brief
  function voicesModels(roles) {
    const note = root.querySelector('#voices-models');
    if (!note || !state.doc.brief.voices_from_model) return;
    const choices = roles.flatMap(role => role.choices), able = choices.find(choice => choice.speaks);
    const current = roles.filter(role => ['cast_shots', 'shots_without_cast'].includes(role.id)).map(role => role.value || role.auto);
    note.textContent = !able ? 'None of your video templates can do this: lines are recorded first.'
      : current.every(value => choices.find(choice => choice.value === value)?.speaks) ? '' : `The model picked for some shots cannot speak: shots with lines use ${able.label} or another model that can.`;
  }

  function modelsOpen() { try { return localStorage.getItem('sr.models') === '1'; } catch { return false; } }

  function dropTarget(zone, role) {
    const accept = role === 'song' ? AUDIO_ACCEPT + ',video/*' : role === 'source' ? 'video/*' : 'image/*,video/*';
    zone.onclick = async () => {
      if (role === 'cast') {  // as on the Cast page: a person is named first; dropping photos adds people too
        const name = await promptDialog({ title: 'Who or what is it?', text: 'Describe them on the Cast page, or drop a photo here.', placeholder: 'Mia, the lead singer', ok: 'Add' });
        if (name?.trim()) actions.ops([{ op: 'add_cast', fields: personFields(name) }]).then(render);
        return;
      }
      const files = await pickFiles(accept, role !== 'song' && role !== 'source');
      if (files.length) upload(files, role);
    };
    zone.ondragover = event => { event.preventDefault(); zone.classList.add('over'); };
    zone.ondragleave = () => zone.classList.remove('over');
    zone.ondrop = event => { event.preventDefault(); event.stopPropagation(); zone.classList.remove('over'); document.getElementById('veil')?.classList.remove('on'); upload(event.dataTransfer.files, role); };
  }

  async function upload(files, role) {
    if (role === 'song') analysisRequested = true;
    const results = await actions.upload(role === 'song' || role === 'source' ? [...files].slice(0, 1) : files, role);
    if (role === 'cast') for (const result of results) {
      const name = await promptDialog({ title: 'Who is it?', text: 'The showrunner uses this name in the plan and the dialogue.', value: state.doc.cast.find(member => member.id === result.target)?.name || '', ok: 'Save' });
      if (name?.trim()) await actions.ops([{ op: 'update_cast', cast: result.target, fields: { name: name.trim() } }]);
    }
    analysisRequested = false;
    setTimeout(render, 150);
  }

  // The song sung in other voices (SeedVC: the music, words and timing stay), in the background: its progress shows on the song card.
  // singers: cast ids or voice samples of the production, first heard first (two for a duet).
  async function changeVoice(singers) {
    try { await api.songVoice(state.doc.id, singers); }  // its events show the progress, the first one at once
    catch (error) { toast(error.message, 'error', 8000); }
  }

  const castSingers = () => state.doc.cast.filter(member => !isPlace(member) && hasVoice(member)).map(member => ({ value: member.id, label: member.name || member.id }));

  // A duet: SeedVC tells the song's two voices apart and gives each its own, in the order they are first heard. singers: the choices
  // ({value, label}); onChoose receives the two values chosen, first heard first; ok names what the button does.
  function duetDialog(singers, onChoose, ok = 'Change the voices') {
    if (singers.length < 2) { toast('A duet needs two voices: give two cast members a voice on the Cast page, or drop voice samples in the media bin.', 'error', 8000); return; }
    const order = [], known = new Set(singers.map(singer => singer.value));  // the plan's singers as they first sing, else the list's order
    [...state.doc.shots].filter(shot => shot.audio_slice).sort((a, b) => a.audio_slice[0] - b.audio_slice[0])
      .forEach(shot => (shot.dialogue || []).forEach(line => { if (known.has(line.speaker) && !order.includes(line.speaker)) order.push(line.speaker); }));
    singers.forEach(singer => { if (!order.includes(singer.value)) order.push(singer.value); });
    const select = (id, chosen) => `<select class="select" id="${id}" style="width:100%">${singers.map(singer => `<option value="${esc(singer.value)}" ${singer.value === chosen ? 'selected' : ''}>${esc(singer.label)}</option>`).join('')}</select>`;
    const back = document.createElement('div');
    back.className = 'modal-back';
    back.innerHTML = `<div class="modal" role="dialog" aria-modal="true"><h3>Two singers</h3><p>SeedVC tells the two voices of the song apart, in the order they are first heard, and gives each its new voice.</p>
      <div class="field"><div class="label">First voice heard</div>${select('duet-first', order[0])}</div>
      <div class="field"><div class="label">Second voice heard</div>${select('duet-second', order[1])}</div>
      <div class="actions"><button class="btn ghost" data-r="0">Cancel</button><button class="btn primary" data-r="1">${esc(ok)}</button></div></div>`;
    const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
    const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
    back.addEventListener('click', event => {
      if (event.target === back) return close();
      const button = event.target.closest('[data-r]');
      if (!button) return;
      const chosen = ['#duet-first', '#duet-second'].map(id => back.querySelector(id).value);
      if (button.dataset.r === '1' && chosen[0] === chosen[1]) return toast('Choose two different voices.', 'error');
      if (button.dataset.r === '1') onChoose(chosen);
      close();
    });
    document.addEventListener('keydown', keys);
    document.getElementById('overlay-root').append(back);
  }

  function ensureAudio() {
    if (!audio) {
      audioSong = state.doc.brief.song;
      audio = new Audio(actions.file(audioSong));
      audio.ontimeupdate = () => drawSong();
      audio.onended = () => updatePlayButton();
    }
  }
  function togglePlay() { ensureAudio(); audio.paused ? audio.play() : audio.pause(); updatePlayButton(); }
  function updatePlayButton() { const button = root.querySelector('#song-play'); if (button) button.innerHTML = icon(audio && !audio.paused ? 'pause' : 'play'); }

  function drawSong() {
    const canvas = root.querySelector('#song-canvas');
    const analysis = state.doc.audio?.analysis;
    if (!canvas || !analysis?.waveform) return;
    const duration = analysis.duration_s || 1;
    drawWave(canvas, analysis.waveform, { progress: audio ? audio.currentTime / duration : 0 });
    const time = root.querySelector('#song-time');
    if (time) time.textContent = `${formatTime(audio?.currentTime || 0)} / ${formatTime(duration)}`;
  }

  render();
  return {
    refreshSingerVoice,
    update(doc) {
      const active = document.activeElement;
      if (active && root.contains(active) && active.matches('textarea, input')) return;  // keep what the user is typing
      render();
    },
    destroy() { audio?.pause(); save.flush(); },
    suspend() { audio?.pause(); updatePlayButton(); },
    theme: drawSong,
    songVoice() {  // the song card shows how the new voice is coming along
      const status = root.querySelector('#song-voice-status');
      if (status && state.songVoice) status.innerHTML = songVoiceStatus();
      else this.update(state.doc);
    },
  };
}

function placeholder(kind) {
  return {
    music_video: 'A synth-pop song about leaving a neon city at dawn. A singer walks through empty streets while the city wakes up behind her…',
    short_film: 'A lighthouse keeper finds a message in a bottle written in her own handwriting, dated tomorrow…',
    ad: 'A 30-second spot for a reusable bottle that keeps drinks cold for 24 hours: hikers, a desert, a cold sip at sunset…',
    trailer: 'A one-minute trailer for a heist film set in a floating city…',
    story: 'A shy fox who collects lost sounds and returns them to their owners…',
    remake: 'What should change: who replaces whom, or what is removed, added or restyled. Everything else stays as in the video…',
    free: 'Describe what you want to make…',
  }[kind] || '';
}

function dropzone(role, accept, title, text) {
  return `<div class="dropzone" data-drop="${role}"><span class="dz-icon">${icon(role === 'song' ? 'music' : 'upload')}</span><span><b>${esc(title)}</b>${esc(text)}</span></div>`;
}

const isPlace = member => ['place', 'location', 'set'].includes((member.role || '').trim().toLowerCase());
const hasVoice = member => !!(member.voice || (member.voice_description || '').trim());

// Who sings the song the showrunner composes: song models take no voice, so it composes the song, then has it sung in these voices.
function sungBy(doc) {
  const voiced = doc.cast.filter(member => !isPlace(member) && hasVoice(member)), chosen = doc.brief.song_voices || [];
  const name = id => doc.cast.find(member => member.id === id)?.name || id;
  const options = [`<option value="" ${chosen.length ? '' : 'selected'}>The song model's own voice</option>`,
    ...voiced.map(member => `<option value="${esc(member.id)}" ${chosen.length === 1 && chosen[0] === member.id ? 'selected' : ''}>${esc(member.name || member.id)}</option>`),
    ...(chosen.length === 2 ? [`<option value="${esc(chosen.join(','))}" selected>${esc(chosen.map(name).join(', then '))} (a duet)</option>`] : []),
    ...(voiced.length >= 2 ? ['<option value="duet">Two singers (a duet)…</option>'] : [])];
  return `<div class="row" style="gap:10px;margin-top:10px;flex-wrap:wrap;align-items:center"><span class="label" style="margin:0">${icon('audio')} Sung by</span>
    <select class="select" id="song-voices" style="width:auto" ${availability(state.singerVoice, "The showrunner composes the song, then has it sung in this voice: song models cannot sing in a given voice. The music, the words and their timing stay.")}>${options.join('')}</select>
    ${voiced.length ? '' : '<span class="muted" style="font-size:12.5px">Give a cast member a voice sample on the Cast page to have the song sung in it.</span>'}</div>`;
}

function songVoiceStatus() {
  const { progress, status } = state.songVoice;
  return `<span class="spinner"></span>${esc(status || 'Changing the voice')}${progress != null && progress > 0 ? ` ${Math.round(progress)}%` : '…'}`;
}

function songCard(doc, analysis) {
  const sections = analysis?.sections || [];
  const duration = analysis?.duration_s || 1;
  const name = doc.brief.song.split('/').pop().replace(/-[0-9a-f]{6}(\.\w+)$/, '$1');
  return `<div class="song-card">
    <div class="song-head"><button class="song-play" id="song-play" title="Play">${icon('play')}</button>
      <div class="grow"><b>${esc(name)}</b><div class="muted mono" id="song-time">0:00 / ${formatTime(duration)}</div></div>
      ${analysis ? `<div class="song-stats"><span class="chip">${icon('music')}${analysis.bpm ? `${Math.round(analysis.bpm)} BPM` : 'Free tempo'}</span><span class="chip">${icon('layers')}${sections.length} ${sections.length === 1 ? 'section' : 'sections'}</span><span class="chip">${icon('clock')}${formatDuration(duration)}</span></div>`
        : `<span class="chip"><span class="spinner"></span>Listening to the song…</span>`}
      ${state.songVoice ? `<span class="chip" id="song-voice-status">${songVoiceStatus()}</span>`
        : `<button class="btn ghost small" id="song-voice" ${availability(state.singerVoice, "Have the song sung in another voice: a cast member's or a voice sample's. The music, the words and their timing stay, and the former version goes to the bin.")}>${icon('audio')} Change the voice</button>`}
      <button class="btn ghost small icon" id="song-remove" title="Remove the song">${icon('x')}</button></div>
    <div class="song-wave"><canvas id="song-canvas"></canvas>
      ${sections.map(section => `<div style="position:absolute;top:0;height:5px;left:${100 * section.start_s / duration}%;width:${100 * (section.end_s - section.start_s) / duration}%;background:${sectionColor(section.name)}"></div>`).join('')}</div>
    ${sections.length ? `<div class="row" style="flex-wrap:wrap;gap:6px">${sections.map(section => `<span class="chip"><i style="width:8px;height:8px;border-radius:2px;background:${sectionColor(section.name)}"></i>${esc(section.name)} <span class="muted">${formatTime(section.start_s)}</span></span>`).join('')}</div>` : ''}
  </div>`;
}

function reference(path, index) {
  const url = actions.file(path);
  const video = /\.(mp4|mov|webm|mkv|m4v|avi)$/i.test(path);
  return `<div class="ref">${video ? `<video src="${url}" muted loop playsinline onmouseenter="this.play()" onmouseleave="this.pause()"></video>` : `<img src="${url}" alt="" loading="lazy">`}
    <button class="btn small icon x" data-remove-ref="${index}" title="Remove">${icon('x')}</button></div>`;
}

function castTile(member) {
  const image = member.master || member.images[0];
  return `<div class="ref clickable" data-cast-tile title="${esc(member.name)}: open the cast">${image ? `<img src="${actions.file(image)}" alt="">` : `<div style="position:absolute;inset:0;display:grid;place-items:center;font-size:28px;font-weight:800;color:var(--muted)">${esc((member.name || '?')[0])}</div>`}
    <span class="tag" style="position:absolute;left:6px;bottom:6px">${esc(member.name || 'Unnamed')}</span></div>`;
}
