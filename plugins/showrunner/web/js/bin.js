// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Media bin: uploads and the director's untracked media, ready to be dragged onto shots (as takes) or cast members.
import { state, actions } from './app.js';
import { icon, esc, art, h, pickFiles, menu, toast, AUDIO_ACCEPT } from './ui.js';
import { BIN_TYPE, SOUND_LAID, shotFromBin, afterSelection } from './storyboard.js';

// Generated files are named "<date>_seed<n>_<start of the prompt>": the bin shows the prompt's words.
const readable = name => name.replace(/^\d{4}-\d\d-\d\d-\d\dh\d\dm\d\ds_seed\d+_/, '').replace(/\.\w+$/, '');

// The zoom of WanGP's image gallery, the same script, loaded when a picture first opens.
let galleryZoom = null;
function loadZoom() {
  if (!galleryZoom) {
    document.head.append(Object.assign(document.createElement('link'), { rel: 'stylesheet', href: 'assets/image_zoom.css' }));
    galleryZoom = import('../image_zoom.js');
  }
  return galleryZoom;
}

// A bin item in full: a video or a sound plays at once; a picture zooms with the wheel or the buttons and moves when held, as in the gallery.
export function showItem(item) {
  const src = item.url || actions.file(item.path), name = readable(item.name || item.path.split('/').pop()), picture = item.kind === 'image';  // url: a media id the showrunner named
  actions.pausePlayback();
  const media = item.kind === 'audio' ? `<audio src="${src}" controls autoplay></audio>`
    : picture ? `<div class="bin-image modal-image-viewport"><img src="${src}" alt="" draggable="false"></div>`
    : `<video src="${src}" ${item.poster ? `poster="${actions.file(item.poster)}"` : ''} controls autoplay playsinline></video>`;
  const back = h(`<div class="modal-back"><div class="modal bin-viewer" role="dialog" aria-modal="true" aria-label="${esc(name)}"><div class="row"><b class="grow">${esc(name)}</b>${picture ? '<span class="muted" style="font-size:12px">Scroll or + and − to zoom, hold and drag to move</span>' : ''}<button class="btn ghost icon" data-close title="Close (Escape)">${icon('x')}</button></div>${media}</div></div>`);
  const image = back.querySelector('.bin-image img');
  let open = true, pressedOutside = false;
  if (image) loadZoom().then(() => { if (open) window.WanGPImageZoom.mount(image, { viewport: image.parentElement }); });
  const close = () => {
    open = false;
    if (image && window.WanGPImageZoom) window.WanGPImageZoom.destroy(image);
    back.querySelector('video, audio')?.pause();
    back.remove();
    document.removeEventListener('keydown', keys);
  };
  const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
  back.addEventListener('pointerdown', event => { pressedOutside = event.target === back; });  // moving a picture and letting go outside it keeps the preview open
  back.addEventListener('click', event => { if (event.target === back && pressedOutside || event.target.closest('[data-close]')) close(); });
  document.addEventListener('keydown', keys);
  document.getElementById('overlay-root').append(back);
}

export function mountBin(root) {
  let known = null;  // the items seen so far: a new one comes into view
  function render() {
    const items = state.doc?.bin || [], position = root.querySelector('.bin-items')?.scrollLeft || 0;
    const open = state.binOpen && (items.length || state.binOpenedEmpty);  // an empty bin stays folded: the board keeps its room
    root.className = `bin ${open ? '' : 'collapsed'}`;
    root.innerHTML = `<div class="bin-head" id="bin-toggle">${icon('bin')} Media bin <span class="muted" style="font-weight:500">${items.length ? `${items.length} · click to play, drag onto a shot or a cast member` : 'drop files here, then drag them onto a shot or a cast member'}</span><span class="grow"></span>${icon(open ? 'chevronDown' : 'chevron')}</div>
      <div class="bin-items">${items.map(item => `<div class="bin-item" draggable="true" data-bin="${item.id}" title="${esc(item.name || item.path)}">
          ${item.poster ? `<img src="${actions.file(item.poster)}" alt="" draggable="false">` : `<div style="position:absolute;inset:0;${art(item.id, item.kind)}"></div>`}
          <span class="tag bin-kind">${icon(item.kind === 'audio' ? 'audio' : item.kind === 'image' ? 'image' : 'video')}${item.by === 'deepy' ? 'Showrunner' : ''}</span>
          ${item.kind === 'image' ? '' : `<span class="bin-play">${icon('play')}</span>`}
          <button class="bin-remove" data-remove-bin title="Remove from the bin (the file stays in the production's folder; Ctrl+Z brings it back)" aria-label="Remove from the bin">${icon('x')}</button>
          <span class="bin-name">${esc(readable(item.name || item.path.split('/').pop()))}</span></div>`).join('')}
        <div class="bin-drop" id="bin-drop">${icon('upload')}<br>Add media</div></div>`;
    const list = root.querySelector('.bin-items'), added = known && items.some(item => !known.has(item.id));
    known = new Set(items.map(item => item.id));
    list.scrollLeft = position;  // a redraw keeps the place
    if (added && !document.querySelector('.bin-viewer')) list.scrollTo({ left: list.scrollWidth, behavior: 'smooth' });  // not while the user looks at an item
    root.querySelector('#bin-toggle').onclick = () => {
      state.binOpen = !open; state.binOpenedEmpty = state.binOpen && !items.length;
      try { localStorage.setItem('sr.bin', state.binOpen ? '1' : '0'); } catch { /* a convenience */ }
      render();
    };
    const drop = root.querySelector('#bin-drop');
    drop.onclick = async () => { const files = await pickFiles('image/*,video/*,' + AUDIO_ACCEPT); if (files.length) actions.upload(files, 'bin'); };
    root.querySelectorAll('[data-bin]').forEach(element => {
      element.ondragstart = event => { event.dataTransfer.setData(BIN_TYPE, element.dataset.bin); event.dataTransfer.effectAllowed = 'copy'; };
      element.onclick = event => {  // a drag fires no click
        if (event.target.closest('[data-remove-bin]')) actions.ops([{ op: 'bin_remove', item: element.dataset.bin }]);
        else showItem(state.doc.bin.find(entry => entry.id === element.dataset.bin));
      };
      element.oncontextmenu = event => {
        event.preventDefault();
        const item = state.doc.bin.find(entry => entry.id === element.dataset.bin);
        const shot = state.selection.size === 1 ? [...state.selection][0] : null;
        menu({ x: event.clientX, y: event.clientY }, [
          { label: item.kind === 'image' ? 'View' : 'Play', icon: item.kind === 'image' ? 'eye' : 'play', action: () => showItem(item) },
          ...(item.kind !== 'audio' && state.doc.brief.kind !== 'remake' ? [{ label: state.selection.size ? 'New shot from it, after the selected shot' : 'New shot from it, at the end', icon: 'plus', action: () => shotFromBin(item.id, afterSelection()) }] : []),
          ...(shot && item.kind !== 'audio' ? [{ label: `Use as a take of the selected shot`, icon: 'layers', action: () => actions.ops([{ op: 'bin_to_take', item: item.id, shot }]) }] : []),
          ...(shot && item.kind === 'audio' ? [{ label: `Play it from the selected shot on`, icon: 'audio', action: () => actions.ops([{ op: 'bin_to_track', item: item.id, shot }]).then(() => toast(SOUND_LAID, 'ok'), () => {}) }] : []),
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
