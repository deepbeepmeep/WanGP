// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Cast: the people and recurring things of the production, with their reference pictures, master image and voice.
import { state, actions } from './app.js';
import { icon, esc, art, h, fieldSaver, pickFiles, confirmDialog, promptDialog, toast, AUDIO_ACCEPT } from './ui.js';
import { BIN_TYPE } from './storyboard.js';

export function mountCast(container) {
  container.innerHTML = `<div class="toolbar"><h2>Cast</h2><span class="stat" id="cast-stat"></span><div class="grow"></div>
    <button class="btn small" id="cast-ask">${icon('director')} Ask the showrunner for master images</button><button class="btn primary small" id="cast-add">${icon('plus')} Add</button></div><div class="cast-grid" id="cast"></div>`;
  const grid = container.querySelector('#cast');
  const save = fieldSaver((id, fields) => actions.ops([{ op: 'update_cast', cast: id, fields }], { quiet: true }).catch(error => toast(error.message, 'error')), 450);
  container.querySelector('#cast-add').onclick = async () => {
    const name = await promptDialog({ title: 'Who or what is it?', placeholder: 'Mia, the lead singer', ok: 'Add' });
    if (name?.trim()) actions.ops([{ op: 'add_cast', fields: { name: name.trim() } }]);
  };
  container.querySelector('#cast-ask').onclick = () => actions.direct('message', 'Make a master image for every cast member that has none (front-facing, neutral background, consistent with the brief and style), and record each one with the cast action.');

  function render() {
    const doc = state.doc;
    container.querySelector('#cast-stat').textContent = `${doc.cast.length} ${doc.cast.length === 1 ? 'member' : 'members'} · used in ${doc.shots.filter(shot => shot.cast.length).length} shots`;
    grid.innerHTML = doc.cast.map(member => {
      const master = member.master || member.images[0];
      const uses = doc.shots.filter(shot => shot.cast.includes(member.id)).length;
      return `<article class="cast-card" data-cast="${member.id}">
        <div class="cast-hero">${master ? `<img src="${actions.file(master)}" alt="">` : `<div class="ph" style="${art(member.id, member.name)}"><div>${icon('image')}<div style="color:#fff;margin-top:6px">Drop a photo here</div></div></div>`}
          <button class="btn small icon lock ${member.locked ? 'active' : ''}" data-lock title="${member.locked ? 'Locked' : 'Lock: the showrunner keeps this look'}">${icon(member.locked ? 'lock' : 'unlock')}</button></div>
        <div class="cast-views">${member.images.map(path => `<span class="view"><img src="${actions.file(path)}" data-view="${esc(path)}" class="${path === member.master ? 'on' : ''}" title="${path === member.master ? 'The reference picture of every shot' : 'Click: make it the reference picture'}" alt=""><button class="x" data-drop-image="${esc(path)}" title="Remove this picture">${icon('x')}</button></span>`).join('')}
          <button class="btn small icon" data-add-image title="Add pictures" style="width:54px;height:54px;flex:none">${icon('plus')}</button></div>
        <div class="cast-body">
          <input class="input cast-name" data-field="name" value="${esc(member.name)}" placeholder="Name" title="Name: click to edit">
          <input class="input" data-field="role" value="${esc(member.role)}" placeholder="Role (lead singer, villain, the red car…)">
          <textarea class="textarea" data-field="description" rows="3" placeholder="Look, age, clothes, what must stay the same in every shot…">${esc(member.description)}</textarea>
          ${member.looks?.length ? `<div class="looks">${member.looks.map(look => `<span class="look" title="${esc(look.description || look.name)}">${look.master ? `<img src="${actions.file(look.master)}" alt="">` : ''}<b>${esc(look.name || 'Look')}</b><span class="muted">${esc([...(look.sections || []), ...(look.shots || []).map(id => `shot ${state.doc.shots.findIndex(shot => shot.id === id) + 1}`)].join(', ') || 'no sequence')}</span></span>`).join('')}</div>` : ''}
          <textarea class="textarea" data-field="voice_description" rows="2" placeholder="Voice: age, timbre, accent, pace (the showrunner makes the voice sample from it; every line is spoken in it)">${esc(member.voice_description || '')}</textarea>
          <div class="voice">${icon('audio')}${member.voice ? `<audio controls src="${actions.file(member.voice)}" style="height:30px;flex:1"></audio>` : `<span class="muted grow">No voice sample · drop audio here</span>`}<button class="btn small" data-add-voice>${member.voice ? 'Replace' : 'Add'}</button>${member.voice ? `<button class="btn small ghost icon" data-drop-voice title="Remove the voice sample (the showrunner makes a new one from the description)">${icon('x')}</button>` : ''}</div>
          <div class="row"><span class="muted" style="font-size:12.5px">In ${uses} ${uses === 1 ? 'shot' : 'shots'}</span><span class="grow"></span>
            <button class="btn small" data-new-picture title="The showrunner makes a new reference picture from the description (after a recast, or when the picture no longer matches it)">${icon('wand')} New picture</button>
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
      card.querySelector('[data-add-voice]').onclick = async () => { const files = await pickFiles(AUDIO_ACCEPT, false); if (files.length) actions.upload(files, 'cast', id); };
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
