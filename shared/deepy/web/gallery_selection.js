// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* Only user gestures write gallery selection; Gradio refresh events are read-only. */
(() => {
  function view() {
    const value = document.querySelector('#wangp-gallery-view textarea')?.value;
    return value ? JSON.parse(value) : null;
  }
  let sequence = 0, pendingView = null, acceptedView = null, viewing = null, repairedView = null;
  // True when the gallery shows other files than the view its clicks resolve through
  // (a gallery value written by another response than the view refresh). Thumbnails are compared by
  // the listed file's name (data-wangp-name), not by their URL: Gradio's cache copy strips characters
  // such as brackets from file names and fast video previews are served under another name.
  function stale(state = view()) {
    if (!state || !document.querySelector('#gallery')) return false;
    const thumb = document.querySelector('#gallery .thumbnail-item');
    const names = thumb ? [...thumb.parentElement.querySelectorAll('.thumbnail-item')].map(node => node.dataset.wangpName) : [];
    if (names.some(name => !name)) return false;
    return names.length !== state.video.length || names.some((name, index) => name !== state.video[index].split(/[\\/]/).pop());
  }
  // `index`: position in the whole gallery `source`.
  function request(state, source, index, path) {
    pendingView = state;
    window.__wangpAssistantChatNS.setBridgeValue('#wangp-gallery-interaction textarea', JSON.stringify({session: state.session, workspace: state.workspace, source, index, path, sequence: ++sequence}));
  }
  function select(source, index = null) {
    const state = view();
    if (!state) return;
    const paths = state[source];
    if (index !== null && (index < 0 || index >= paths.length)) return;
    if (source === 'video' && index !== null && stale(state)) {
      // The clicked thumbnail is not view.video[index]: render the gallery again rather than select
      // another file. Once per view, so that an unforeseen difference never blocks selection.
      const value = document.querySelector('#wangp-gallery-view textarea').value;
      if (value !== repairedView && window.__wangpAssistantChatNS?.repairGallery) {
        repairedView = value;
        window.__wangpAssistantChatNS.repairGallery();
        return;
      }
    }
    request(state, source, index === null ? null : index + state[source + '_offset'], index === null ? null : paths[index]);
  }
  function selectedSource() {
    const tab = document.querySelector('#wangp-gallery-tabs [role=tab][aria-selected=true]');
    return tab ? tabSource(tab) : 'video';
  }
  // Enlarges the gallery once it shows the media requested by view(). Tabs hiding
  // the gallery are displayed only while it is full screen, without selecting them.
  function present() {
    const item = viewing, state = acceptedView || view();
    if (!item) return;
    if (state.workspace !== item.workspace || state.video[state.selected] !== item.path) {
      // Full screen needs the click's user activation, which expires.
      if (performance.now() - item.requested > 5000) viewing = null;
      return;
    }
    viewing = null;
    const container = document.querySelector('#gallery .gallery-container');
    const hosts = [];
    for (let node = container.parentElement; node; node = node.parentElement) if (node.matches('.tabitem') && getComputedStyle(node).display === 'none') hosts.push(node);
    hosts.forEach(node => node.classList.add('wangp-gallery-viewer-host'));
    const focused = document.activeElement;
    const close = () => {
      if (document.fullscreenElement === container) return;
      document.removeEventListener('fullscreenchange', close);
      hosts.forEach(node => node.classList.remove('wangp-gallery-viewer-host'));
      window.scrollTo(item.pageX, item.pageY);
      if (selectedSource() !== item.source) select(item.source);
      focused?.focus({preventScroll: true});
      item.onClose();
    };
    document.addEventListener('fullscreenchange', close);
    // The gallery's keyboard navigation listens on its preview, as after a click there.
    container.requestFullscreen().then(() => container.querySelector('.preview')?.focus({preventScroll: true}), close);
  }
  // A gallery tab (or its entry in Gradio's overflow menu, which has no tab attributes) is identified by its label.
  function tabSource(tab, step = 0) {
    const sources = view()?.sources || [{id: 'video', label: ''}];
    const index = Math.max(0, sources.findIndex(source => source.label.trim() === tab.textContent.trim()));
    return sources[(index + step + sources.length) % sources.length].id;
  }
  window.WanGPGallerySelection = {
    stale: () => stale(),
    // A strip gallery thumbnail click: index in the visible window of gallery `source`.
    select: (source, index) => {const state = view(); if (state) select(source, Number(index) - state[source + '_offset']);},
    acceptView(value) {
      const state = JSON.parse(value);
      const current = acceptedView || view();
      // Network completion order is not publication order. Revision counters
      // belong to a Gradio session and can restart with a new server.
      if (state.session === current?.session && state.revision < current.revision) return false;
      const accept = state.session !== pendingView?.session || state.workspace !== pendingView?.workspace || state.gallery_sequence >= sequence;
      // Several responses may share one Gradio flush, before its DOM updates.
      if (accept) acceptedView = state;
      if (accept && viewing) requestAnimationFrame(present);
      return accept;
    },
    // Chat media in the gallery's own full-screen viewer: `item` ({workspace, index, path}) is selected as a
    // thumbnail click would. Leaving full screen restores the page scroll and gallery tab, then calls onClose.
    view(item, onClose) {
      const state = view();
      if (!state || state.workspace !== item.workspace) return false;
      viewing = {...item, onClose, source: selectedSource(), requested: performance.now(), pageX: window.scrollX, pageY: window.scrollY};
      if (state.video[state.selected] === item.path) present();
      else request(state, 'video', item.index, item.path);
      return true;
    },
    // A chat 3D asset in the 3D gallery's viewer (`markup`), full screen. The viewer exists only while it is
    // full screen: removing it on leaving stops its rendering and releases its WebGL resources.
    viewMesh(markup, onClose) {
      const host = document.createElement('div');
      host.innerHTML = markup;
      const viewer = host.firstElementChild;
      viewer.classList.add('wangp-chat-mesh-viewer');
      const focused = document.activeElement;
      const close = () => {
        if (document.fullscreenElement === viewer) return;
        document.removeEventListener('fullscreenchange', close);
        viewer.remove();
        focused?.focus({preventScroll: true});
        onClose();
      };
      document.addEventListener('fullscreenchange', close);
      document.body.append(viewer);
      viewer.requestFullscreen().catch(close);
    }
  };
  function indexOf(thumb) {
    return [...thumb.parentElement.querySelectorAll('.thumbnail-item')].indexOf(thumb);
  }
  document.addEventListener('click', event => {
    const tab = event.target.closest('#wangp-gallery-tabs [role=tab], #wangp-gallery-tabs .overflow-dropdown button');
    if (tab) {
      select(tabSource(tab));
      return;
    }
    const thumb = event.target.closest('#gallery .thumbnail-item');
    if (thumb) {select('video', indexOf(thumb)); return;}
    if (event.target.closest('#gallery .preview')) {
      requestAnimationFrame(() => {
        const selected = document.querySelector('#gallery .thumbnail-item.selected');
        if (selected && indexOf(selected) !== view().selected) select('video', indexOf(selected));
      });
    }
  }, true);
  document.addEventListener('keydown', event => {
    const tab = event.target.closest('#wangp-gallery-tabs [role=tab]');
    if (tab && ['ArrowLeft', 'ArrowRight'].includes(event.code)) {
      select(tabSource(tab, event.code === 'ArrowRight' ? 1 : -1));
      return;
    }
    if (!event.target.closest('#gallery') || !['ArrowLeft', 'ArrowRight'].includes(event.code)) return;
    const selected = document.querySelector('#gallery .thumbnail-item.selected'), state = view();
    if (selected && state.video.length) select('video', (indexOf(selected) + (event.code === 'ArrowRight' ? 1 : -1) + state.video.length) % state.video.length);
  }, true);
})();
