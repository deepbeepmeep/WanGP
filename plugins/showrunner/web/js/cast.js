// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Cast: the people and recurring things of the production, with their reference pictures, master image and voice.
import { state, actions } from './app.js';
import { api } from './api.js';
import { icon, esc, art, h, fieldSaver, pickFiles, confirmDialog, promptDialog, toast, availability, AUDIO_ACCEPT, personFields } from './ui.js';
import { BIN_TYPE } from './storyboard.js';

// The shots a cast member appears in that already have a take.
function shot_ids(member) { return state.doc.shots.filter(shot => shot.cast.includes(member.id) && shot.takes.length).map(shot => shot.id); }

export function mountCast(container) {
  container.innerHTML = `<div class="toolbar"><h2>Cast</h2><span class="stat" id="cast-stat"></span><div class="grow"></div>
    <button class="btn small" id="cast-ask">${icon('director')} Ask the showrunner for master images</button><button class="btn small" id="cast-add-place" title="A location that stays the same from shot to shot">${icon('plus')} Add a place</button><button class="btn primary small" id="cast-add">${icon('plus')} Add</button></div><div class="cast-grid" id="cast"></div>`;
  const grid = container.querySelector('#cast');
  const save = fieldSaver((id, fields) => actions.ops([{ op: 'update_cast', cast: id, fields }], { quiet: true }).catch(error => toast(error.message, 'error')), 450);
  let loras = null;  // the LoRA choices, once Settings has LoRAs on cast cards on
  api.settings().then(settings => settings.cast_loras ? api.castLoras(state.pid) : null).then(found => { if (found) { loras = found; render(); } }).catch(error => toast(error.message, 'error', 6000));

  // A member's LoRA, among those of the model families this production uses; the shots of other families are made without it.
  function loraRow(member) {
    if (!loras) return '';
    const lora = member.lora, known = lora && loras.families.some(family => family.folder === lora.folder && family.loras.includes(lora.file));
    const name = file => file.split(/[\\/]/).pop().replace(/\.safetensors$/i, '');
    const shots = state.doc.shots.filter(shot => shot.cast.includes(member.id) && shot.included);
    const fitting = lora ? shots.filter(shot => loras.shots[shot.id] === lora.folder).length : 0;
    const value = item => esc(JSON.stringify([item.folder, item.file]));
    return `<div class="cast-lora"><div class="row"><span class="muted" style="font-size:12.5px">LoRA</span><select class="select grow" data-lora title="LoRA of this cast member"><option value="">No LoRA</option>
        ${lora && !known ? `<option value="${value(lora)}" selected>${esc(name(lora.file))} (not found for the models in use)</option>` : ''}
        ${loras.families.map(family => `<optgroup label="${esc(`For ${family.label}`)}">${family.loras.map(file => `<option value="${value({ folder: family.folder, file })}" ${lora?.folder === family.folder && lora.file === file ? 'selected' : ''}>${esc(name(file))}</option>`).join('')}</optgroup>`).join('')}</select></div>
      ${lora ? `<div class="row"><input class="input" data-lora-field="trigger" value="${esc(lora.trigger || '')}" placeholder="Trigger word" title="Added to the prompts of the shots that use it" style="flex:1;min-width:0"><input class="input" data-lora-field="multiplier" type="number" min="0" max="2" step="0.05" value="${lora.multiplier}" title="Strength" style="width:74px"></div>
        <span class="muted" style="font-size:12px">Used on ${fitting} of the ${shots.length} ${shots.length === 1 ? 'shot' : 'shots'} showing them${fitting < shots.length ? ': the others use another family of models' : ''}.</span>` : ''}</div>`;
  }

  container.querySelector('#cast-add').onclick = async () => {
    const name = await promptDialog({ title: 'Who or what is it?', placeholder: 'Mia, the lead singer', ok: 'Add' });
    if (name?.trim()) actions.ops([{ op: 'add_cast', fields: personFields(name) }]);
  };
  container.querySelector('#cast-add-place').onclick = async () => {
    const name = await promptDialog({ title: 'Which place?', text: 'Shots that list this place get its picture as a reference, so the set stays the same. Drop your own photo on its card, or ask the showrunner for a picture.', placeholder: 'The Kitchen Kings Live studio', ok: 'Add' });
    if (name?.trim()) actions.ops([{ op: 'add_cast', fields: { name: name.trim(), role: 'place', description: '' } }]);
  };
  container.querySelector('#cast-ask').onclick = () => actions.direct('message', 'Make a master image for every cast member that has none (front-facing, neutral background, consistent with the brief and style; for a place, a wide view of it as the viewer should see it, without the cast or words, with the crowd that belongs to it), and record each one with the cast action.');

  function render() {
    const doc = state.doc;
    container.querySelector('#cast-stat').textContent = `${doc.cast.length} ${doc.cast.length === 1 ? 'member' : 'members'} · used in ${doc.shots.filter(shot => shot.cast.length).length} shots`;
    const blocked = actions.directorBlocked(), ask = container.querySelector('#cast-ask');
    const reason = blocked || (!doc.cast.length ? 'Add a cast member first.' : doc.cast.every(member => member.master || member.images.length) ? 'Every cast member has a picture.' : '');
    ask.disabled = !!reason;
    ask.title = reason;
    grid.innerHTML = doc.cast.map(member => {
      const master = member.master || member.images[0], place = ['place', 'location', 'set'].includes((member.role || '').trim().toLowerCase());
      const uses = doc.shots.filter(shot => shot.cast.includes(member.id) || (shot.dialogue || []).some(line => line.speaker === member.id)).length;  // a narrator is in the shots it speaks in
      return `<article class="cast-card" data-cast="${member.id}">
        <div class="cast-hero">${master ? `<img src="${actions.file(master)}" alt="">` : `<div class="ph" style="${art(member.id, member.name)}"><div>${icon('image')}<div style="color:#fff;margin-top:6px">Drop a photo here</div></div></div>`}
          <button class="btn small icon lock ${member.locked ? 'active' : ''}" data-lock title="${member.locked ? 'Locked' : 'Lock: the showrunner keeps this look'}">${icon(member.locked ? 'lock' : 'unlock')}</button></div>
        <div class="cast-views">${member.images.map(path => `<span class="view"><img src="${actions.file(path)}" data-view="${esc(path)}" class="${path === member.master ? 'on' : ''}" title="${path === member.master ? 'The reference picture of every shot' : 'Click: make it the reference picture'}" alt=""><button class="x" data-drop-image="${esc(path)}" title="Remove this picture">${icon('x')}</button></span>`).join('')}
          <button class="btn small icon" data-add-image title="Add pictures" style="width:54px;height:54px;flex:none">${icon('plus')}</button></div>
        <div class="cast-body">
          <input class="input cast-name" data-field="name" value="${esc(member.name)}" placeholder="Name" title="Name: click to edit">
          <input class="input" data-field="role" value="${esc(member.role)}" placeholder="Role (lead singer, villain, the red car…)">
          <textarea class="textarea" data-field="description" rows="3" placeholder="${place ? 'What the place looks like, what must stay the same in every shot…' : 'Look, age, clothes, what must stay the same in every shot…'}">${esc(member.description)}</textarea>
          ${member.looks?.length ? `<div class="looks">${member.looks.map(look => `<span class="look" title="${esc(look.description || look.name)}">${look.master ? `<img src="${actions.file(look.master)}" alt="">` : ''}<b>${esc(look.name || 'Look')}</b><span class="muted">${esc([...(look.sections || []), ...(look.shots || []).map(id => `shot ${state.doc.shots.findIndex(shot => shot.id === id) + 1}`)].join(', ') || 'no sequence')}</span></span>`).join('')}</div>` : ''}
          ${place ? '' : `<textarea class="textarea" data-field="voice_description" rows="2" placeholder="Voice: age, timbre, accent, pace (the showrunner makes the voice sample from it; every line is spoken in it)">${esc(member.voice_description || '')}</textarea>
          <div class="voice">${icon('audio')}${member.voice ? `<audio controls src="${actions.file(member.voice)}" style="height:30px;flex:1"></audio>` : `<span class="muted grow">No voice sample · drop audio here</span>`}<button class="btn small" data-add-voice>${member.voice ? 'Replace' : 'Add'}</button>${member.voice ? `<button class="btn small ghost icon" data-drop-voice title="Remove the voice sample (the showrunner makes a new one from the description)">${icon('x')}</button>` : ''}</div>
          ${loraRow(member)}`}
          <div class="row"><span class="muted" style="font-size:12.5px">In ${uses} ${uses === 1 ? 'shot' : 'shots'}</span><span class="grow"></span>
            <button class="btn small" data-new-picture ${availability(blocked, 'Asks the showrunner for a new reference picture from the description, after a recast or when the picture no longer matches it (a few minutes)')}>${icon('wand')} New picture</button>
            ${shot_ids(member).length ? `<button class="btn small" data-reshoot-cast ${availability(blocked, `Asks the showrunner to shoot again the ${shot_ids(member).length} shots showing them, with their look and voice as they are now`)}>${icon('play')} Re-shoot ${shot_ids(member).length}</button>` : ''}
            <button class="btn small ghost danger" data-remove>${icon('trash')}</button></div>
        </div></article>`;
    }).join('') + `<div class="add-shot" id="cast-drop" style="min-height:320px"><div class="col" style="align-items:center;text-align:center;padding:20px">${icon('users')}<b>Add a person or a thing</b><span class="muted">Drop photos here: each new photo becomes a cast member</span></div></div>`;
    bind();
  }

  function bind() {
    grid.querySelectorAll('[data-cast]').forEach(card => {
      const id = card.dataset.cast;
      const member = state.doc.cast.find(item => item.id === id);
      card.querySelectorAll('[data-field]').forEach(input => input.addEventListener('input', () => save(id, { [input.dataset.field]: input.value })));
      card.querySelector('[data-lock]').onclick = () => actions.ops([{ op: 'update_cast', cast: id, fields: { locked: !member.locked } }]);
      card.querySelector('[data-lora]')?.addEventListener('change', event => {
        const [folder, file] = event.target.value ? JSON.parse(event.target.value) : [];
        actions.ops([{ op: 'update_cast', cast: id, fields: { lora: file ? { file, folder, multiplier: member.lora?.multiplier ?? 1, trigger: member.lora?.trigger || '' } : null } }]);
      });
      card.querySelectorAll('[data-lora-field]').forEach(input => input.addEventListener('input', () => {
        const field = name => card.querySelector(`[data-lora-field="${name}"]`).value;
        save(id, { lora: { ...member.lora, trigger: field('trigger'), multiplier: +field('multiplier') || 0 } });
      }));
      card.querySelector('[data-reshoot-cast]')?.addEventListener('click', () => { save.flush(); actions.direct('reshoot', `${member.name || id} changed: shoot again the shots showing them, with their look and voice as they are now.`, shot_ids(member)); });
      card.querySelector('[data-new-picture]').onclick = () => { save.flush(); actions.direct('message', `Make a new reference picture of ${member.name || id} (${id}) from their description as it is now, make it their master, and tell me which shots should be shot again with it.`); };
      card.querySelector('.cast-hero img')?.addEventListener('click', event => {  // the whole picture, uncropped
        const back = h(`<div class="modal-back"><img src="${event.target.src}" alt="" style="max-width:calc(100% - 48px);max-height:calc(100% - 48px);border-radius:12px;box-shadow:var(--shadow-lg);cursor:zoom-out"></div>`);
        const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
        const keys = key => { if (key.key === 'Escape') { key.stopPropagation(); close(); } };
        back.onclick = close;
        document.addEventListener('keydown', keys);
        document.getElementById('overlay-root').append(back);
      });
      card.querySelectorAll('[data-view]').forEach(image => image.onclick = () => actions.ops([{ op: 'update_cast', cast: id, fields: { master: image.dataset.view } }]));
      card.querySelector('[data-add-image]').onclick = async () => { const files = await pickFiles('image/*'); if (files.length) actions.upload(files, 'cast', id); };
      card.querySelectorAll('[data-drop-image]').forEach(button => button.onclick = event => {
        event.stopPropagation();
        const images = member.images.filter(path => path !== button.dataset.dropImage);
        actions.ops([{ op: 'update_cast', cast: id, fields: { images, master: member.master === button.dataset.dropImage ? images[0] || null : member.master } }]);
      });
      card.querySelector('[data-drop-voice]')?.addEventListener('click', () => actions.ops([{ op: 'update_cast', cast: id, fields: { voice: null } }]));
      const addVoice = card.querySelector('[data-add-voice]');  // a place has none
      if (addVoice) addVoice.onclick = async () => { const files = await pickFiles(AUDIO_ACCEPT, false); if (files.length) actions.upload(files, 'cast', id); };
      card.querySelector('[data-remove]').onclick = async () => {
        if (await confirmDialog({ title: `Remove ${member.name || 'this cast member'}?`, text: 'Shots stop referencing them; their pictures stay in the production folder.', ok: 'Remove', danger: true })) actions.ops([{ op: 'remove_cast', cast: id }]);
      };
      card.ondragover = event => { if (event.dataTransfer.types.includes('Files') || event.dataTransfer.types.includes(BIN_TYPE)) { event.preventDefault(); card.classList.add('over'); } };
      card.ondragleave = () => card.classList.remove('over');
      card.ondrop = async event => {
        event.preventDefault();
        event.stopPropagation();
        card.classList.remove('over');
        document.getElementById('veil')?.classList.remove('on');
        const binId = event.dataTransfer.getData(BIN_TYPE);
        if (binId) {
          const item = state.doc.bin.find(entry => entry.id === binId);
          if (item?.kind === 'audio') actions.ops([{ op: 'update_cast', cast: id, fields: { voice: item.path } }]);
          else if (item) actions.ops([{ op: 'update_cast', cast: id, fields: { images: [...member.images, item.path], master: member.master || item.path } }]);
        } else actions.upload(event.dataTransfer.files, 'cast', id);
      };
    });
    const drop = grid.querySelector('#cast-drop');
    drop.onclick = async () => { const files = await pickFiles('image/*'); for (const file of files) await actions.upload([file], 'cast'); };
    drop.ondragover = event => { event.preventDefault(); drop.classList.add('over'); };
    drop.ondragleave = () => drop.classList.remove('over');
    drop.ondrop = async event => {
      event.preventDefault(); event.stopPropagation(); drop.classList.remove('over'); document.getElementById('veil')?.classList.remove('on');
      for (const file of event.dataTransfer.files) await actions.upload([file], 'cast');
    };
  }

  render();
  return {
    update() {
      const active = document.activeElement;
      if (active && grid.contains(active) && active.matches('textarea, input')) return;
      render();
    },
    destroy() { save.flush(); },
  };
}
