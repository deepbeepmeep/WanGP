// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Deepy's chat window as a component, for a page to show in an iframe: Deepy's own code and styles (streaming, scrolling, thoughts and steps), fed by that page.
(() => {
  const WAC = window.__wangpAssistantChatNS;
  const query = new URLSearchParams(location.search), body = document.body;
  // What the URL can leave out with <name>=0.
  for (const name of ['contour', 'input', 'stats', 'avatars', 'stop', 'actions']) body.classList.toggle(`no-${name}`, query.get(name) === '0');
  body.classList.toggle('wide', query.get('wide') === '1');  // messages use the whole width
  body.classList.toggle('dark', query.get('theme') === 'dark');
  if (query.get('font')) document.querySelector('#assistant_chat_dock').style.setProperty('--dock-font-scale', query.get('font'));
  WAC.ensureShell();
  WAC.setDockOpen(true, false);
  WAC.applyDisplaySettings({compact_actions: query.get('compact') !== '0', gallery_media_viewer: false});
  // The window has no connection of its own: a missing update and a request typed in it go to its page.
  WAC.requestCanonicalSync = () => { window.DeepyWindow.onGap?.(); return false; };
  const submit = text => { WAC.clearRequestInput(text); window.DeepyWindow.onSubmit?.(text); return true; };
  WAC.submitRequest = submit;
  WAC.queueBusyRequest = submit;
  WAC.steerRequest = submit;
  // A picture, video or sound of the conversation opens over the page showing the window, when that page shows media: in a new tab otherwise.
  WAC.openAttachment = link => {
    const kind = link.dataset.mediaKind;
    if (link.hasAttribute('download') || !['image', 'video', 'audio'].includes(kind) || !window.DeepyWindow.onMedia) return false;
    window.DeepyWindow.onMedia({kind, url: link.href, name: (link.querySelector('.chat__attachment-title') || link).textContent.trim()});
    return true;
  };
  window.DeepyWindow = {
    onSubmit: null,
    onGap: null,
    onMedia: null,  // ({kind, url, name}): the page shows that media
    replay(updates) { WAC.consumePayload({batch: updates, replay: true}); },  // a conversation at once, the window at its end
    update(update) { WAC.consumePayload(update); },  // one update as Deepy publishes it: the window follows it when it is at its end
    rejoin() { WAC.chatSequence = -1; },  // before updates again after some were skipped
    shown() { WAC.applyAutoscrollState(WAC.lastScrollState); },  // back in view: where the reader was
    reset() { WAC.reset(); },
    theme(name) { body.classList.toggle('dark', name === 'dark'); },
  };
})();
