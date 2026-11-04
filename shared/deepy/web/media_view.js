// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* Shared presentation for gallery media and the existing Python property formatter. */
(() => {
  const documentHtml = html => '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'"><style>body{margin:0;color:#24485d;font:13px/1.5 system-ui}table{width:100%;table-layout:fixed}td{overflow-wrap:anywhere}td:first-child{width:28%;white-space:normal!important}#video_info td{font-size:12px}#video_info .selected-media-prompt{display:block;max-height:20lh;overflow-y:auto;overflow-wrap:anywhere}b{font-weight:500}.copy-swap__full{display:none}.copy-swap:focus .copy-swap__trunc{display:none}.copy-swap:focus .copy-swap__full{display:inline}</style>' + html;
  function media(item) {
    if (item.kind === '3d') return thumbnail(item);  // 3D assets are viewed interactively in WanGP's 3D gallery
    const node = document.createElement(item.kind === 'image' ? 'img' : item.kind);
    node.setAttribute('aria-label', item.name);
    if (item.kind === 'image') {node.alt = item.name; node.loading = 'lazy';}
    else {node.controls = true; node.preload = 'metadata'; if (item.kind === 'video') {node.playsInline = true; if (item.poster) node.poster = item.poster;}}
    node.src = item.url;
    return node;
  }
  function thumbnail(item) {
    if (item.kind === 'audio') {
      const node = document.createElement('div'); node.className = 'media-audio-thumbnail';
      node.textContent = '♫'; node.setAttribute('aria-label', 'Audio');
      return node;
    }
    const node = document.createElement('img');
    node.alt = item.name; node.loading = 'lazy'; node.decoding = 'async';
    node.dataset.mediaKind = item.kind;
    node.src = item.thumbnail;
    return node;
  }
  function mesh(item) {
    const container = document.createElement('wangp-mesh-viewer');
    container.className = 'mesh-viewer'; container.dataset.preview = item.thumbnail;
    const viewer = document.createElement(item.format === 'ply' ? 'wangp-splat-viewer' : 'model-viewer');
    viewer.setAttribute('src', item.url); viewer.setAttribute('aria-label', item.name);
    if (item.navigation) viewer.dataset.navigation = item.navigation;
    if (item.walk_start) viewer.dataset.walkStart = item.walk_start;
    if (item.level) viewer.dataset.level = 'free';
    if (item.format !== 'ply') {
      viewer.setAttribute('alt', item.name);
      for (const [key, value] of Object.entries({'camera-controls': '', 'touch-action': 'pan-y', 'interaction-prompt': 'none', 'shadow-intensity': '0.6', 'environment-image': 'neutral', exposure: '1', 'tone-mapping': 'neutral'})) viewer.setAttribute(key, value);
    }
    const actions = document.createElement('div'); actions.className = 'mesh-viewer-actions';
    actions.innerHTML = '<button type="button" data-mesh-action="play" title="Play the Animation" aria-label="Play the Animation" hidden><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg></button>'
      + (item.format === 'ply' ? '<button type="button" data-mesh-action="walk" title="Walk Through the Scene" aria-label="Walk Through the Scene" aria-pressed="false"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="13" cy="4.5" r="1.8"/><path d="M10 21l2.5-6.5-2.5-3 1.5-4 3 2.5 3 1M11.5 7.5 8 9.5l-1.5 4M12.5 14.5l3 2.5 1 4"/></svg></button>' : '');
    container.append(viewer, actions);
    return container;
  }
  function playButton(play) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'media-play';
    button.title = 'Play video'; button.setAttribute('aria-label', button.title);
    // Keep native scrolling and dragging; a moved/cancelled touch is not a tap.
    let pointer = null;
    button.onpointerdown = event => {pointer = {x: event.clientX, y: event.clientY, moved: false};};
    button.onpointermove = event => {if (pointer && Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 8) pointer.moved = true;};
    button.onpointercancel = () => {if (pointer) pointer.moved = true;};
    button.onclick = event => {
      event.stopPropagation();
      const moved = pointer?.moved; pointer = null;
      if (event.detail && moved) return;
      play();
    };
    return button;
  }
  window.WanGPMediaView = {documentHtml, media, thumbnail, mesh, playButton};
})();
