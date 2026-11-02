// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* The app player: web apps made with Deepy, HTML pages and document previews, framed isolated from WanGP (shared/apps/server.py serves
   them with an opaque origin). One app plays at a time, in a panel beside the chat that moves by its bar and can fill the window or the
   screen. Minimized, it keeps only its bar and pauses the app, as while the page is hidden. It shows the app's console and frame rate,
   reloads it when Deepy changes it (WanGPAppPlayer.sync, called after chat updates), and can hand a summary of what went wrong to the chat
   composer. */
(() => {
  if (window.WanGPAppPlayer) return;
  const icon = paths => `<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  const ICONS = {
    reload: icon('<path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5"/>'),
    pause: icon('<path d="M9 5v14M15 5v14"/>'),
    play: icon('<path d="M8 5v14l11-7z"/>'),
    console: icon('<path d="m5 8 4 4-4 4M11 17h8"/>'),
    expand: icon('<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>'),
    collapse: icon('<path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5"/>'),
    screen: icon('<rect x="3" y="5" width="18" height="12" rx="2"/><path d="M8 21h8M12 17v4"/>'),
    send: icon('<path d="m4 4 17 8-17 8 3-8-3-8Z"/><path d="M7 12h14"/>'),
    download: icon('<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14"/>'),
    close: icon('<path d="m6 6 12 12M6 18 18 6"/>'),
    minimize: icon('<path d="M5 12h14"/>'),
    restore: icon('<rect x="5" y="5" width="14" height="14" rx="2"/>'),
  };
  const VIEWPORTS = [['Fit', ''], ['Desktop', '1280px'], ['Tablet', '820px'], ['Phone', '390px']];
  let panel = null, current = null;

  const button = (name, title, onClick) => {
    const node = document.createElement('button');
    node.type = 'button'; node.className = 'wangp-app-player__button'; node.dataset.action = name; node.title = title; node.setAttribute('aria-label', title);
    node.innerHTML = ICONS[name]; node.addEventListener('click', onClick);
    return node;
  };

  function build() {
    panel = document.createElement('section');
    panel.className = 'wangp-app-player'; panel.hidden = true; panel.setAttribute('aria-label', 'App player');
    panel.innerHTML = '<header class="wangp-app-player__bar"><span class="wangp-app-player__title"></span><span class="wangp-app-player__fps" hidden></span><span class="wangp-app-player__tools"></span></header>'
      + '<div class="wangp-app-player__stage"><div class="wangp-app-player__viewport"></div><div class="wangp-app-player__notice" hidden></div></div>'
      + '<footer class="wangp-app-player__controls" hidden></footer><div class="wangp-app-player__console" hidden><ol></ol></div>';
    const tools = panel.querySelector('.wangp-app-player__tools');
    const viewport = document.createElement('select');
    viewport.className = 'wangp-app-player__select'; viewport.title = 'Viewport width'; viewport.setAttribute('aria-label', 'Viewport width');
    for (const [label, width] of VIEWPORTS) viewport.append(new Option(label, width));
    viewport.addEventListener('change', () => { panel.querySelector('.wangp-app-player__viewport').style.maxWidth = viewport.value || ''; });
    const download = document.createElement('a');
    download.className = 'wangp-app-player__button'; download.dataset.action = 'download'; download.title = 'Download the App (.wgpapp)'; download.innerHTML = ICONS.download; download.hidden = true;
    tools.append(viewport,
      button('reload', 'Reload', () => reload()),
      button('pause', 'Pause', () => setPaused(!current.userPaused)),
      button('console', 'Console', () => toggleConsole()),
      button('send', 'Tell Deepy What Happened', () => sendToDeepy()),
      download,
      button('screen', 'Full Screen', () => (panel.querySelector('.wangp-app-player__stage').requestFullscreen?.() || Promise.resolve()).catch(() => {})),
      button('expand', 'Fill the Window', () => setLayout(panel.classList.contains('is-full') ? '' : 'is-full')),
      button('minimize', 'Minimize (pauses the app)', () => setLayout(panel.classList.contains('is-minimized') ? '' : 'is-minimized')),
      button('close', 'Close', () => close()));
    const bar = panel.querySelector('.wangp-app-player__bar');
    bar.addEventListener('dblclick', event => { if (!event.target.closest('button, select, a')) setLayout(panel.classList.contains('is-full') ? '' : 'is-full'); });
    bar.addEventListener('pointerdown', startDrag);
    document.body.append(panel);
    window.addEventListener('message', onMessage);
    document.addEventListener('visibilitychange', applyPause);
    document.addEventListener('wangp-viewer-scope', applyPause);
    document.addEventListener('close', applyPause, true);
    document.addEventListener('focusin', applyPause);
    document.addEventListener('fullscreenchange', applyPause);
    document.addEventListener('scroll', applyPause, {capture: true, passive: true});
    document.addEventListener('click', () => queueMicrotask(applyPause));
    window.addEventListener('blur', () => queueMicrotask(applyPause));
    window.addEventListener('focus', applyPause);
    window.addEventListener('resize', () => { keepOnScreen(); applyPause(); });
    window.addEventListener('pagehide', () => applyPause(true));
    window.addEventListener('pageshow', () => applyPause());
  }

  function setLayout(layout) {
    panel.classList.remove('is-full', 'is-minimized');
    if (layout) panel.classList.add(layout);
    const full = panel.classList.contains('is-full');
    const expand = panel.querySelector('[data-action="expand"]');
    expand.innerHTML = full ? ICONS.collapse : ICONS.expand; expand.title = full ? 'Restore the Panel' : 'Fill the Window';
    const minimized = panel.classList.contains('is-minimized'), minimize = panel.querySelector('[data-action="minimize"]');
    minimize.innerHTML = minimized ? ICONS.restore : ICONS.minimize; minimize.title = minimized ? 'Restore (the app resumes)' : 'Minimize (pauses the app)';
    minimize.setAttribute('aria-label', minimize.title);
    keepOnScreen();  // minimized and restored panels change width
    applyPause();
  }

  // The floating panel moves by its bar (not by its controls) and keeps the bar on screen; filling the window, a phone-sized window and the
  // workspace viewer's embedded player ignore the place.
  function floating() {
    return !panel.classList.contains('is-full') && !current?.owner && !matchMedia('(max-width: 700px)').matches;
  }

  function startDrag(event) {
    if (event.button !== 0 || event.target.closest('button, select, a') || !floating()) return;
    const bar = event.currentTarget, box = panel.getBoundingClientRect(), dx = event.clientX - box.left, dy = event.clientY - box.top;
    bar.setPointerCapture(event.pointerId);
    panel.classList.add('is-dragging');  // the frame takes no pointer events meanwhile: it would swallow the moves
    const move = moved => place(moved.clientX - dx, moved.clientY - dy);
    const end = () => { panel.classList.remove('is-dragging'); bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', end); bar.removeEventListener('pointercancel', end); };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', end); bar.addEventListener('pointercancel', end);
  }

  function place(x, y) {  // the panel stays on screen (its top and left part when it is larger than the window)
    panel.style.setProperty('--player-x', `${Math.round(Math.max(0, Math.min(x, innerWidth - panel.offsetWidth)))}px`);
    panel.style.setProperty('--player-y', `${Math.round(Math.max(0, Math.min(y, innerHeight - panel.offsetHeight)))}px`);
    panel.classList.add('is-placed');
  }

  function keepOnScreen() {
    if (panel.classList.contains('is-placed') && floating()) { const box = panel.getBoundingClientRect(); place(box.left, box.top); }
  }

  function applyPause(pageHidden = false) {
    if (!current) return;
    const modal = document.querySelector('dialog[open]:focus-within') || [...document.querySelectorAll('dialog[open]')].at(-1), box = panel.getBoundingClientRect();
    const full = document.fullscreenElement;
    const hidden = pageHidden === true || document.hidden || !document.hasFocus() || panel.hidden || !panel.isConnected || panel.classList.contains('is-minimized')
      || (modal && !modal.contains(panel)) || (full && !panel.contains(full) && !full.contains(panel))
      || !box.width || !box.height || box.bottom <= 0 || box.top >= innerHeight || box.right <= 0 || box.left >= innerWidth
      || getComputedStyle(panel).visibility === 'hidden';
    // Keep the browsing context: the runtime suspends scheduled work in place.
    // A launch finishing after the panel hides waits until it is shown.
    if (!current.frame && current.launch) {
      if (hidden) return;
      current.frame = frameFor(current.launch);
      panel.querySelector('.wangp-app-player__viewport').replaceChildren(current.frame);
    }
    const paused = !!(hidden || current.userPaused);
    if (current.frame && current.paused !== paused) {
      current.paused = paused;
      current.frame.contentWindow?.postMessage({wangp: paused ? 'pause' : 'resume'}, '*');
    }
  }

  function setPaused(paused) {
    current.userPaused = paused;
    const node = panel.querySelector('[data-action="pause"]');
    node.innerHTML = paused ? ICONS.play : ICONS.pause; node.title = paused ? 'Resume' : 'Pause';
    applyPause();
  }

  function toggleConsole(show) {
    const drawer = panel.querySelector('.wangp-app-player__console');
    drawer.hidden = show === undefined ? !drawer.hidden : !show;
  }

  function log(level, text) {
    const list = panel.querySelector('.wangp-app-player__console ol');
    const item = document.createElement('li');
    item.className = 'is-' + level; item.textContent = text;
    list.append(item);
    while (list.children.length > 200) list.firstChild.remove();
    list.parentElement.scrollTop = list.parentElement.scrollHeight;
    if (level === 'error' || level === 'exception') {
      current.errors += 1;
      const consoleButton = panel.querySelector('[data-action="console"]');
      consoleButton.dataset.count = String(current.errors);
    }
  }

  function onMessage(event) {
    if (!current || !current.frame || event.source !== current.frame.contentWindow || !event.data || typeof event.data.wangp !== 'string') return;
    const data = event.data;
    if (data.wangp === 'log') log(String(data.level || 'log'), String(data.text || ''));
    else if (data.wangp === 'fps') { const fps = panel.querySelector('.wangp-app-player__fps'); fps.hidden = false; fps.textContent = `${data.fps} fps`; }
    else if (data.wangp === 'ready') { current.paused = undefined; applyPause(); }
  }

  function notice(text) {
    const node = panel.querySelector('.wangp-app-player__notice');
    node.textContent = text || ''; node.hidden = !text;
  }

  function frameFor(launch) {
    const frame = document.createElement('iframe');
    frame.className = 'wangp-app-player__frame';
    // No allow-same-origin: the app keeps an opaque origin (the server enforces it too, with its CSP sandbox directive).
    frame.setAttribute('sandbox', launch.kind === 'document' ? '' : 'allow-scripts allow-forms allow-modals allow-downloads allow-pointer-lock');
    frame.setAttribute('allow', 'fullscreen; gamepad; autoplay');
    frame.setAttribute('referrerpolicy', 'no-referrer');
    frame.title = launch.name;
    frame.name = 'wangp-managed-app';
    frame.src = launch.url;
    return frame;
  }

  async function launch(appId) {
    const response = await fetch('/apps/api/launch', {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({app_id: appId})});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `The app could not start (${response.status}).`);
    return payload;
  }

  async function open(link) {
    if (!panel) build();
    const appId = link.dataset.appId;
    if (current && current.appId === appId && current.owner === link.closest('.wv-preview')) { panel.hidden = false; setLayout(panel.classList.contains('is-full') ? 'is-full' : ''); return true; }
    close();
    const owner = link.closest('.wv-preview');
    (owner || document.body).append(panel);
    current = {appId, owner, version: link.dataset.appVersion || '', errors: 0, userPaused: false, frame: null, token: '', name: link.dataset.appName || (link.querySelector('.chat__attachment-title') || link).textContent.trim()};
    panel.hidden = false; setLayout(panel.classList.contains('is-full') ? 'is-full' : '');
    panel.querySelector('.wangp-app-player__title').textContent = current.name;
    panel.querySelector('.wangp-app-player__fps').hidden = true;
    panel.querySelector('.wangp-app-player__select').value = '';
    panel.querySelector('.wangp-app-player__viewport').style.maxWidth = '';
    panel.querySelector('.wangp-app-player__console ol').replaceChildren();
    delete panel.querySelector('[data-action="console"]').dataset.count;
    const download = panel.querySelector('[data-action="download"]');
    download.hidden = !link.dataset.downloadUrl;
    if (link.dataset.downloadUrl) { download.href = link.dataset.downloadUrl; download.download = link.dataset.download || ''; }
    setPaused(false);
    notice('Starting…');
    const opening = current;
    try {
      const started = await launch(appId);
      if (current !== opening) {release(started.token); return true;}
      Object.assign(current, {token: started.token, url: started.url, kind: started.kind, launch: started});
      panel.dataset.kind = started.kind;
      panel.querySelector('.wangp-app-player__title').textContent = started.name || current.name;
      const controls = Object.entries(started.controls || {});
      const footer = panel.querySelector('.wangp-app-player__controls');
      footer.hidden = controls.length === 0;
      footer.replaceChildren(...controls.map(([key, action]) => { const item = document.createElement('span'); const kbd = document.createElement('kbd'); kbd.textContent = key; item.append(kbd, ' ' + action); return item; }));
      applyPause();
      notice('');
    } catch (error) {
      if (current === opening) notice(error.message);
    }
    return true;
  }

  function reload() {
    if (!current || !current.frame) return;
    current.errors = 0;
    delete panel.querySelector('[data-action="console"]').dataset.count;
    log('info', '— reloaded —');
    current.paused = undefined;
    current.frame.src = current.url + (current.url.includes('?') ? '&' : '?') + 'r=' + Date.now();
  }

  function close() {
    if (!current) { if (panel) panel.hidden = true; return; }
    const token = current.token;
    current.frame?.remove();
    current = null;
    if (panel) panel.hidden = true;
    release(token);
  }

  function release(token) {
    if (token) fetch('/apps/api/close', {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token})}).catch(() => {});
  }

  function sendToDeepy() {
    if (!current) return;
    const errors = [...panel.querySelectorAll('.wangp-app-player__console li.is-error, .wangp-app-player__console li.is-exception')].map(node => node.textContent);
    const fps = panel.querySelector('.wangp-app-player__fps');
    const facts = [errors.length ? `${errors.length} error(s) in its console, the last: ${errors[errors.length - 1].slice(0, 300)}` : 'no error in its console', fps.hidden ? '' : `running at ${fps.textContent}`].filter(Boolean).join(', ');
    const text = `About the app "${current.name}" (${facts}; app_logs has the details): `;
    if (window.WAC && typeof WAC.setRequestInputValue === 'function') {
      WAC.setRequestInputValue(text);
      const input = WAC.requestInput && WAC.requestInput();
      if (input) { input.focus(); input.setSelectionRange?.(text.length, text.length); }
    }
  }

  // Cards rendered again after Deepy changed an app carry a new version: the open player reloads it.
  function sync(root) {
    if (!current || !root || !root.querySelectorAll) return;
    for (const card of root.querySelectorAll(`[data-app-id="${CSS.escape(current.appId)}"]`)) {
      const version = card.dataset.appVersion || '';
      if (version && version !== current.version) { current.version = version; reload(); break; }
    }
  }

  function openLink(link) {
    if (!link || !link.dataset || !link.dataset.appId) return false;
    open(link).catch(error => notice(error.message));
    return true;
  }

  window.WanGPAppPlayer = {open: openLink, close, closeOwner: owner => {if (current?.owner === owner) close();}, refresh: () => applyPause(), sync};
})();
