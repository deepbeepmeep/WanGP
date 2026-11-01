// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Media bin: uploads and the director's untracked media, ready to be dragged onto shots (as takes) or cast members.
import { state, actions } from './app.js';
import { icon, esc, art, pickFiles, menu, AUDIO_ACCEPT } from './ui.js';
import { BIN_TYPE } from './storyboard.js';

// Generated files are named "<date>_seed<n>_<start of the prompt>": the bin shows the prompt's words.
const readable = name => name.replace(/^\d{4}-\d\d-\d\d-\d\dh\d\dm\d\ds_seed\d+_/, '').replace(/\.\w+$/, '');

export function mountBin(root) {
  function render() {
    const items = state.doc?.bin || [];
    const open = state.binOpen && (items.length || state.binOpenedEmpty);  // an empty bin stays folded: the board keeps its room
    root.className = `bin ${open ? '' : 'collapsed'}`;
    root.innerHTML = `<div class="bin-head" id="bin-toggle">${icon('bin')} Media bin <span class="muted" style="font-weight:500">${items.length ? `${items.length} · drag onto a shot or a cast member` : 'drop files here, then drag them onto a shot or a cast member'}</span><span class="grow"></span>${icon(open ? 'chevronDown' : 'chevron')}</div>
      <div class="bin-items">${items.map(item => `<div class="bin-item" draggable="true" data-bin="${item.id}" title="${esc(item.name || item.path)}">
          ${item.poster ? `<img src="${actions.file(item.poster)}" alt="" draggable="false">` : `<div style="position:absolute;inset:0;${art(item.id, item.kind)}"></div>`}
          <span class="tag bin-kind">${icon(item.kind === 'audio' ? 'audio' : item.kind === 'image' ? 'image' : 'video')}${item.by === 'deepy' ? 'Showrunner' : ''}</span>
          <span class="bin-name">${esc(readable(item.name || item.path.split('/').pop()))}</span></div>`).join('')}
        <div class="bin-drop" id="bin-drop">${icon('upload')}<br>Add media</div></div>`;
    root.querySelector('#bin-toggle').onclick = () => {
      state.binOpen = !open; state.binOpenedEmpty = state.binOpen && !items.length;
      try { localStorage.setItem('sr.bin', state.binOpen ? '1' : '0'); } catch { /* a convenience */ }
      render();
    };
    const drop = root.querySelector('#bin-drop');
    drop.onclick = async () => { const files = await pickFiles('image/*,video/*,' + AUDIO_ACCEPT); if (files.length) actions.upload(files, 'bin'); };
    root.querySelectorAll('[data-bin]').forEach(element => {
      element.ondragstart = event => { event.dataTransfer.setData(BIN_TYPE, element.dataset.bin); event.dataTransfer.effectAllowed = 'copy'; };
      element.oncontextmenu = event => {
        event.preventDefault();
        const item = state.doc.bin.find(entry => entry.id === element.dataset.bin);
        const shot = state.selection.size === 1 ? [...state.selection][0] : null;
        menu({ x: event.clientX, y: event.clientY }, [
          ...(shot && item.kind !== 'audio' ? [{ label: `Use as a take of the selected shot`, icon: 'layers', action: () => actions.ops([{ op: 'bin_to_take', item: item.id, shot }]) }] : []),
          { label: 'Open', icon: 'expand', action: () => window.open(actions.file(item.path), '_blank') },
          '-',
          { label: 'Remove from the bin', icon: 'trash', danger: true, action: () => actions.ops([{ op: 'bin_remove', item: item.id }]) },
        ]);
      };
    });
  }
  render();
  return { update: render };
}
