// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
  const WAC = window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
  let noticeTimer = null;
  let currentTab = 'chat', chatScrollState = null, lastAnswer = '', dismissedAnswer = '', hasSnapshot = false;
  let settingsSessionId = null;
  let settingsBaseline = null, lastSettingsForm = null;
  let backgroundChatState = null;
  let mediaScrollState = null;
  let loadingSession = null, generationProgress = null;
  let activeSessionTitle = '';
  const $ = selector => document.querySelector(selector);
  function notice(text) {
    const node = $('#app-notice'); node.textContent = text; node.hidden = !text;
    clearTimeout(noticeTimer); if (text) noticeTimer = setTimeout(() => {node.hidden = true;}, 7000);
  }
  function connection(connected) {
    $('#connection').hidden = connected;
    $('#connection').dataset.state = connected ? 'connected' : 'disconnected';
  }
  function restoreBackgroundChatScroll() {
    if (backgroundChatState && currentTab === 'chat' && backgroundChatState.sessionId === WAC.chatSessionId) {
      WAC.applyAutoscrollState(backgroundChatState.scroll);
      WAC.scheduleComposerLayout(backgroundChatState.scroll);
    }
    backgroundChatState = null;
  }
  const transport = new DeepyConnection('.', {
    snapshot: state => {
      snapshot(state, hasSnapshot); hasSnapshot = true;
      restoreBackgroundChatScroll();
    },
    event: message => {
      if (message.type === 'snapshot') snapshot(message.data, true);
      else if (message.type === 'chat') consumeChat(message.data);
      else if (message.type === 'gallery') gallery(message.data);
      else if (message.type === 'workspaces') workspaces.render(message.data);
      else if (message.type === 'display_settings') WAC.applyDisplaySettings(message.data);
      else if (message.type === 'voice_config') WAC.setVoiceMode?.(message.data.mode);
      else if (message.type === 'restoration') { WAC.setRestoration(message.data?.pending ? message.data : null); progress(generationProgress); }
      else if (message.type === 'progress') progress(message.data);
      else if (message.type === 'sessions') sessions(message.data);
      else if (message.type === 'settings') {
        if (currentTab === 'settings') renderSettings(message.data);
        else settingsSessionId = null;
      }
      else if (message.type === 'error' && message.data.visible) notice(message.data.message);
    },
    connection,
    unauthorized: () => {window.location.assign('/auth/login?next=' + encodeURIComponent(location.pathname));},
  });
  const api = (path, body, signal) => transport.request(path, body, signal);
  WAC.saveDisplaySettings = values => api('display-settings', values);
  const workspaceHost = document.createElement('div'); workspaceHost.id = 'workspace-toolbar'; workspaceHost.hidden = true;
  $('.app-heading').append(workspaceHost);
  const workspaces = new WanGPWorkspacePicker(workspaceHost, api, notice, {deepy: true});
  function progress(data) {
    if (data?.type === 'download') {
      const download = data.data;
      let description = data.description;
      if (download && !/^Downloading/i.test(description)) {
        description = /^Loading\s/i.test(description) ? description.replace(/^Loading\b/i, 'Downloading') : description ? `Downloading files — ${description}` : 'Downloading files';
      }
      data = download
        ? {...data, type: 'progress', data: [download.total ? download.completed / download.total : null, description, download.total]}
        : {...data, type: 'status', data: data.description};
    }
    generationProgress = data;
    const sessionStatus = WAC.restorationStatus || loadingSession || WAC.state.status;
    const loading = sessionStatus?.kind === 'session_loading' && currentTab !== 'chat';
    if (loading) data = {type: 'status', data: sessionStatus.text, aborting: false};
    $('#abort-generation').hidden = loading;
    const node = $('#generation-progress'); node.hidden = !data;
    if (!data) return;
    const bar = $('#generation-bar');
    const position = data.type === 'progress' ? data.data[0] : null;
    const statusOnly = data.type !== 'progress';
    bar.hidden = statusOnly;
    node.classList.toggle('status-only', statusOnly);
    const label = ((data.type === 'progress' ? data.data[1] : data.data) ?? '').replace(/^(Loading model\b.*?|Downloading.*?)\s*(?:\.{3}|…)$/i, '$1');
    $('#generation-status').textContent = label;
    $('#generation-status').title = label;
    $('#generation-count').textContent = '';
    if (Array.isArray(position) && position[1] > 0) {
      const [step, total] = position;
      bar.max = total;
      bar.value = step;
      $('#generation-count').textContent = `${step} / ${total} ${data.data[3] || 'steps'}` + (step > 0 ? ` · ${Math.round(100 * step / total)}%` : '');
    } else if (typeof position === 'number' && (position > 0 || data.data[2] > 0)) {
      bar.max = 1; bar.value = position;
      $('#generation-count').textContent = Math.round(position * 100) + '%';
    } else bar.removeAttribute('value');
    $('#abort-generation').disabled = data.aborting;
    $('#abort-generation span').textContent = data.aborting ? 'Aborting…' : 'Abort';
  }
  function tab(name) {
    if (currentTab === 'chat' && name !== 'chat') chatScrollState = WAC.captureAutoscrollState();
    currentTab = name;
    workspaceHost.hidden = !['visual', 'audio', 'mesh'].includes(name);
    $('.session-bar').hidden = name !== 'chat' || !WAC.multiSessionEnabled;
    if (name === 'chat') $('#answer-notice').hidden = true;
    document.querySelectorAll('.view').forEach(node => {node.hidden = node.id !== name + '-view';});
    document.querySelectorAll('[data-tab]').forEach(node => node.setAttribute('aria-selected', String(node.dataset.tab === name)));
    if (name === 'chat') { WAC.setDockOpen(true, false); WAC.scheduleComposerLayout(chatScrollState); chatScrollState = null; }
    gallery(galleryItems);
    if (name === 'settings' && settingsSessionId !== WAC.chatSessionId) loadSettings();
    progress(generationProgress);
  }
  function scrollGallerySelection(view) {
    if (view.hidden || !view.dataset.pendingMedia) return;
    requestAnimationFrame(() => {
      if (view.hidden) return;
      const card = [...view.querySelectorAll('.media-card')].find(node => node.dataset.mediaId === view.dataset.pendingMedia);
      if (card) card.scrollIntoView({block: 'nearest', behavior: 'instant'});
      delete view.dataset.pendingMedia;
    });
  }
  function installGalleryDivider(divider) {
    const view = divider.closest('.gallery-view');
    const browser = view.querySelector('.gallery-browser');
    const grid = view.querySelector('.media-grid');
    const jumps = ['last', 'first'].map(destination => {
      const jump = document.createElement('button'); jump.type = 'button'; jump.className = 'gallery-jump gallery-jump-' + destination; jump.hidden = true;
      jump.title = `Go to ${destination} media`; jump.setAttribute('aria-label', jump.title);
      jump.setAttribute('aria-controls', grid.id);
      jump.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${destination === 'last' ? 'M12 4v16m-6-6 6 6 6-6' : 'M12 20V4m-6 6 6-6 6 6'}"/></svg>`;
      divider.append(jump);
      jump.onclick = () => browser.scrollTo({top: destination === 'last' ? browser.scrollHeight : 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      return jump;
    });
    const updateJump = () => {
      const empty = !grid.querySelector('.media-card');
      jumps[0].hidden = empty || browser.scrollHeight - browser.clientHeight - browser.scrollTop <= 8;
      jumps[1].hidden = empty || browser.scrollTop <= 8;
    };
    browser.addEventListener('scroll', updateJump, {passive: true});
    const observer = new ResizeObserver(updateJump); observer.observe(browser); observer.observe(grid);
    const storageKey = 'deepy-gallery-split-' + view.id;
    const toggle = divider.querySelector('.gallery-details-toggle');
    const collapse = closed => {
      view.classList.toggle('details-collapsed', closed);
      toggle.setAttribute('aria-expanded', String(!closed));
      toggle.title = closed ? 'Show media information' : 'Hide media information';
      toggle.setAttribute('aria-label', toggle.title);
      localStorage.setItem(storageKey + '-collapsed', String(closed));
      const info = view.querySelector('iframe');
      if (closed && info.pendingRequest) {info.pendingRequest.abort(); delete info.dataset.mediaId;}
      if (!closed && !view.hidden) gallery(galleryItems);
    };
    toggle.onclick = () => collapse(!view.classList.contains('details-collapsed'));
    if (localStorage.getItem(storageKey + '-collapsed') === 'true') collapse(true);
    const resize = value => {
      const percent = Math.max(20, Math.min(80, value));
      view.style.setProperty('--gallery-split', percent + '%');
      divider.setAttribute('aria-valuenow', String(Math.round(percent)));
    };
    const saved = localStorage.getItem(storageKey);
    if (saved !== null && Number.isFinite(Number(saved))) resize(Number(saved));
    divider.onpointerdown = event => {
      if (event.button !== 0 || event.target.closest('button')) return;
      if (view.classList.contains('details-collapsed')) collapse(false);
      event.preventDefault(); divider.setPointerCapture(event.pointerId); view.classList.add('is-resizing');
    };
    divider.onpointermove = event => {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      const box = view.getBoundingClientRect(); resize(100 * (event.clientY - box.top) / box.height);
    };
    divider.onlostpointercapture = () => {view.classList.remove('is-resizing'); localStorage.setItem(storageKey, divider.getAttribute('aria-valuenow'));};
    divider.onpointerup = event => { if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId); };
    divider.onkeydown = event => {
      if (event.target.closest('button')) return;
      const current = Number(divider.getAttribute('aria-valuenow'));
      const next = {ArrowUp: current - 5, ArrowDown: current + 5, Home: 20, End: 80}[event.key];
      if (next === undefined) return;
      event.preventDefault(); collapse(false); resize(next); localStorage.setItem(storageKey, divider.getAttribute('aria-valuenow'));
    };
  }
  function settingsValues() {
    return Object.fromEntries([...$('#deepy-settings-form').querySelectorAll('[name]')].map(input => [input.name, input.name === 'use_template_properties' ? input.value === 'true' : input.type === 'number' ? Number(input.value) : input.value]));
  }
  function renderSettings(form, acknowledged = null) {
    const local = settingsValues();
    if (settingsBaseline?.instance === form.sync.instance && form.sync.revision < settingsBaseline.revision) return WanGPFormSync.merge(settingsBaseline.values, local, settingsBaseline.values).dirty.length;
    const base = acknowledged || settingsBaseline?.values;
    const merging = settingsBaseline?.scope === form.sync.scope && Object.keys(local).length;
    const merged = merging ? WanGPFormSync.merge(base, local, form.values) : {values: form.values, conflicts: [], dirty: []};
    settingsBaseline = {...form.sync, values: {...form.values}};
    for (const key of merged.conflicts) settingsBaseline.values[key] = base[key];
    lastSettingsForm = form;
    function field(def, container) {
      let input = container.querySelector('[name="'+def.key+'"]');
      if (!input) {
        const label = document.createElement('label'); label.textContent = def.label;
        input = document.createElement(def.choices ? 'select' : 'input'); input.name = def.key;
        label.append(input); container.append(label);
      }
      if (def.choices) {
        const choices = JSON.stringify(def.choices);
        if (input.dataset.choices !== choices) {
          input.replaceChildren(); input.dataset.choices = choices;
          for (const [text, value] of def.choices) {const option = document.createElement('option'); option.value = String(value); option.textContent = text; input.append(option);}
        }
      } else {input.type = 'number'; input.min = def.minimum; input.max = def.maximum; input.step = def.step; input.required = true;}
      if (input.value !== String(merged.values[def.key])) input.value = String(merged.values[def.key]);
      input.setAttribute('aria-invalid', String(merged.conflicts.includes(def.key)));
    }
    field({key: 'use_template_properties', label: 'Default Dimensions / Durations / Seed', choices: form.property_modes}, $('#property-mode'));
    for (const def of form.properties) field(def, $('#property-fields'));
    for (const def of [...form.preferences, ...form.templates]) field(def, $('#template-fields'));
    const mode = $('#property-mode select');
    mode.onchange = () => { for (const input of $('#property-fields').querySelectorAll('input')) input.disabled = mode.value === 'true'; };
    mode.onchange(); $('#save-settings').disabled = false;
    let controls = $('#settings-conflict-controls');
    if (!controls) {
      controls = document.createElement('div'); controls.id = 'settings-conflict-controls';
      for (const [label, action] of [['Load saved values', () => {settingsBaseline = null; renderSettings(lastSettingsForm);}], ['Keep my edits', () => {settingsBaseline = lastSettingsForm.sync; controls.hidden = true; $('#settings-status').textContent = 'Unsaved changes';}]]) {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = label; button.onclick = action; controls.append(button);
      }
      $('#settings-status').after(controls);
    }
    controls.hidden = !merged.conflicts.length;
    const labels = Object.fromEntries([...form.properties, ...form.preferences, ...form.templates, {key:'use_template_properties',label:'Default properties'}].map(field => [field.key,field.label]));
    $('#settings-status').textContent = merged.conflicts.length ? 'Changed in another window: '+merged.conflicts.map(key => labels[key]).join(', ')+'. Your edits have been kept.' : merged.dirty.length ? 'Unsaved changes' : '';
    return merged.dirty.length;
  }
  async function loadSettings() {
    $('#settings-status').textContent = 'Loading…'; $('#save-settings').disabled = true;
    try { renderSettings(await api('settings')); settingsSessionId = WAC.chatSessionId; }
    catch (error) { $('#settings-status').textContent = error.message; }
  }
  function consumeChat(payload, notify = true) {
    const wasLoading = WAC.state.status?.kind === 'session_loading';
    WAC.consumePayload(payload);
    if (loadingSession) WAC.setStatus(loadingSession);
    if (wasLoading || WAC.state.status?.kind === 'session_loading') progress(generationProgress);
    const messageId = [...WAC.state.order].reverse().find(id => WAC.state.messages[id].role === 'assistant');
    const message = messageId && WAC.messageNode(messageId);
    const text = message ? [...message.querySelectorAll('[data-block-type="markdown"]')].map(node => node.textContent).join(' ').replace(/\s+/g, ' ').trim() : '';
    const answerId = WAC.chatSessionId + ':' + messageId;
    const preview = text.slice(0, 160) + (text.length > 160 ? '…' : '');
    const signature = answerId + ':' + preview;
    if (!notify || !text) $('#answer-notice').hidden = true;
    else if (currentTab !== 'chat' && signature !== lastAnswer && answerId !== dismissedAnswer) {
      $('#answer-title').textContent = $('#deepy-name').textContent + ' replied';
      $('#answer-preview').textContent = preview;
      $('#answer-notice').dataset.answerId = answerId;
      $('#answer-notice').hidden = false;
    }
    lastAnswer = signature;
  }
  function openMedia(item) {
    const viewer = $('#media-viewer');
    if (currentTab === 'chat') mediaScrollState = {sessionId: WAC.chatSessionId, scroll: WAC.captureAutoscrollState()};
    viewer.querySelector('wangp-mesh-viewer')?.remove();
    WanGPImageZoom.destroy(viewer.querySelector('img'));
    const media = item.kind === '3d' ? WanGPMediaView.mesh(item) : viewer.querySelector(item.kind === 'image' ? 'img' : item.kind);
    for (const node of viewer.querySelectorAll('img, video, audio')) node.hidden = node !== media;
    if (item.kind === 'video') media.poster = item.thumbnail || item.poster || '';
    if (item.kind !== '3d') media.src = item.url;
    media.setAttribute('aria-label', item.name);
    if (item.kind === 'image') media.alt = item.name;
    viewer.showModal();
    if (item.kind === 'image') WanGPImageZoom.mount(media, {viewport: viewer});
    if (item.kind === '3d') viewer.append(media);
    else if (item.kind !== 'image') media.play().catch(() => {});
  }
  WAC.openAttachment = link => {
    const kind = link.dataset.mediaKind;
    // Without the gallery viewer, images and videos open in a new tab.
    if (!['image', 'video', 'audio', '3d'].includes(kind) || (!WAC.galleryMediaViewer && ['image', 'video'].includes(kind))) return false;
    const thumbnail = link.querySelector('img')?.src || '';
    openMedia({kind, format: link.dataset.mediaFormat, navigation: link.dataset.mediaNavigation, walk_start: link.dataset.mediaWalkStart, level: link.dataset.mediaLevel, url: link.href, name: (link.querySelector('.chat__attachment-title') || link).textContent.trim(), thumbnail, poster: thumbnail});
    return true;
  };
  let galleryItems = [];
  function gallery(items) {
    galleryItems = items;
    for (const [scope, kinds] of [['visual', ['image', 'video']], ['audio', ['audio']], ['mesh', ['3d']]]) {
      const container = $('#' + scope + '-gallery');
      const wanted = items.filter(item => kinds.includes(item.kind));
      const previous = [...container.querySelectorAll('.media-card')];
      const followedLast = !previous.length || previous[previous.length - 1].classList.contains('selected');
      const view = $('#' + scope + '-view');
      const info = $('#' + scope + '-info');
      if (view.hidden) {
        if (info.pendingRequest) {info.pendingRequest.abort(); delete info.dataset.mediaId;}
        continue;
      }
      const selected = wanted.find(item => item.selected);
      const mediaId = selected ? selected.id : '';
      if (view.dataset.pendingMedia !== mediaId) delete view.dataset.pendingMedia;
      if (mediaId && followedLast && !previous.some(card => card.dataset.mediaId === mediaId)) view.dataset.pendingMedia = mediaId;
      if (!view.classList.contains('details-collapsed') && (info.dataset.mediaId !== mediaId || info.dataset.mediaUrl !== selected?.url)) {
        info.pendingRequest?.abort();
        info.pendingRequest = new AbortController();
        const request = info.pendingRequest;
        info.dataset.mediaId = mediaId;
        info.dataset.mediaUrl = selected?.url || '';
        const documentHtml = WanGPMediaView.documentHtml;
        info.srcdoc = documentHtml(mediaId ? 'Loading media information…' : 'Select media to view its information.');
        if (mediaId) api('media/' + encodeURIComponent(mediaId) + '/info', undefined, request.signal).then(result => {
          if (!request.signal.aborted && info.dataset.mediaId === mediaId) info.srcdoc = documentHtml(result.html);
        }).catch(error => { if (!request.signal.aborted && info.dataset.mediaId === mediaId) {info.srcdoc = documentHtml('Could not load media information.'); notice(error.message);} }).finally(() => {if (info.pendingRequest === request) delete info.pendingRequest;});
      }
      const ids = new Set(wanted.map(item => item.id));
      for (const card of container.querySelectorAll('[data-media-id]')) if (!ids.has(card.dataset.mediaId)) card.remove();
      container.querySelector('.gallery-empty')?.remove();
      const cards = new Map([...container.children].map(card => [card.dataset.mediaId, card]));
      for (const [index, item] of wanted.entries()) {
        let card = cards.get(item.id);
        if (card && card.dataset.mediaUrl !== item.url) { card.remove(); card = null; }
        if (!card) {
          card = document.createElement('article'); card.className = 'media-card'; card.dataset.mediaId = item.id;
          card.dataset.mediaUrl = item.url;
          card.dataset.mediaName = item.name;
          card.tabIndex = 0; card.setAttribute('aria-label', item.name);
          card.onclick = async event => {
            if (event.target.closest('a, button')) return;
            try { const result = await api('media/' + encodeURIComponent(item.id) + '/select', {}); gallery(result.gallery); }
            catch (error) { notice(error.message); }
          };
          card.onkeydown = event => { if (event.target === card && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); card.click(); } };
          const media = WanGPMediaView.thumbnail(item);
          const title = document.createElement('div'); title.className = 'media-name'; title.textContent = item.name;
          const stats = document.createElement('div'); stats.className = 'media-stats';
          stats.textContent = item.kind.toUpperCase() + ' · ' + (item.size < 1048576 ? Math.ceil(item.size / 1024) + ' KB' : (item.size / 1048576).toFixed(1) + ' MB');
          const actions = document.createElement('div'); actions.className = 'media-actions';
          {
            const expand = document.createElement('button'); expand.type = 'button'; expand.title = ['image', '3d'].includes(item.kind) ? 'View full screen' : 'Play ' + item.kind; expand.setAttribute('aria-label', expand.title + ': ' + item.name);
            expand.innerHTML = ['image', '3d'].includes(item.kind) ? '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 3H3v6M15 3h6v6M3 15v6h6M21 15v6h-6"/></svg>' : '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m8 4 12 8-12 8Z"/></svg>';
            expand.onclick = () => openMedia(card.mediaItem); actions.append(expand);
          }
          const download = document.createElement('a'); const downloadUrl = new URL(item.url, location.href); downloadUrl.searchParams.set('download', 'true'); download.href = downloadUrl.href; download.download = item.name; download.title = 'Download'; download.setAttribute('aria-label', 'Download ' + item.name);
          download.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/></svg>';
          actions.append(download); card.append(media, title, stats, actions); container.append(card);
          if (item.kind === 'video') card.append(WanGPMediaView.playButton(() => openMedia(card.mediaItem)));
        }
        card.mediaItem = item;
        if (card.dataset.mediaName !== item.name) {
          card.dataset.mediaName = item.name; card.setAttribute('aria-label', item.name);
          card.querySelector('.media-name').textContent = item.name;
          const download = card.querySelector('.media-actions a'); download.download = item.name; download.setAttribute('aria-label', 'Download ' + item.name);
          const expand = card.querySelector('.media-actions button'); expand.setAttribute('aria-label', expand.title + ': ' + item.name);
        }
        if (container.children[index] !== card) container.insertBefore(card, container.children[index] || null);
        card.classList.toggle('selected', item.selected);
        card.setAttribute('aria-current', String(item.selected));
      }
      if (!wanted.length) { const empty = document.createElement('div'); empty.className = 'gallery-empty'; empty.textContent = 'Your imports and creations will appear here.'; container.append(empty); }
      scrollGallerySelection(view);
    }
  }
  function snapshot(state, notify = false) {
    workspaces.render(state.workspaces, true);
    WAC.applyDisplaySettings(state.display_settings);
    WAC.setRestoration(state.restoration?.pending ? state.restoration : null);
    $('#deepy-name').textContent = state.deepy_type === 'prime' ? 'Deepy Prime' : 'Deepy Zero';
    WAC.multiSessionEnabled = state.multi_session;
    consumeChat(state.chat, notify); gallery(state.gallery); progress(state.progress);
    sessions(state);
    if (currentTab === 'settings') loadSettings();
  }
  function sessions(state) {
    WAC.multiSessionEnabled = state.multi_session;
    WAC.activeSessionId = state.active_session_id;
    activeSessionTitle = state.active_session_title;
    const picker = $('#saved-session'); picker.replaceChildren();
    if (!WAC.activeSessionId) { const current = document.createElement('option'); current.value = ''; current.textContent = activeSessionTitle || 'Current conversation'; current.disabled = true; picker.append(current); }
    for (const item of state.sessions) { const option = document.createElement('option'); option.value = item.id; option.textContent = item.title; picker.append(option); }
    picker.value = WAC.activeSessionId;
    $('#delete-session').disabled = !WAC.activeSessionId;
    $('.session-bar').hidden = currentTab !== 'chat' || !state.multi_session;
    WAC.setQueuedEditButtonLabels(!!WAC.queuedEditMessageId);
  }
  const connect = () => transport.connect().then(restoreBackgroundChatScroll);
  async function control(action, payload = {}) {
    const changingSession = action === 'resume' || action === 'reset';
    if (changingSession && loadingSession) return;
    if (action === 'resume' && (!payload.id || payload.id === WAC.activeSessionId)) return;
    const generating = !!generationProgress;
    if (changingSession && (generating || WAC.isAssistantBusy())) {
      $('#saved-session').value = WAC.activeSessionId;
      notice(generating ? 'A generation is in progress. Wait for it to finish before changing conversation.' : 'Deepy is active in this conversation. Wait for it to finish before changing conversation.');
      return;
    }
    if (changingSession) {
      const title = action === 'resume' ? [...$('#saved-session').options].find(option => option.value === payload.id).textContent : '';
      loadingSession = {visible: true, kind: 'session_loading', text: action === 'reset' ? 'Creating new session…' : 'Loading Session ' + title};
      WAC.setStatus(loadingSession); progress(generationProgress);
      $('#assistant_chat_reset_button').disabled = true;
      $('#assistant_chat_reset_button').setAttribute('aria-busy', 'true');
    }
    try { const state = await api('control', {action, payload}); if (changingSession) loadingSession = null; snapshot(state); }
    catch (error) {
      if (changingSession) { loadingSession = null; WAC.setStatus(null); progress(generationProgress); }
      if (changingSession) $('#saved-session').value = WAC.activeSessionId;
      notice(error.message);
    }
    finally {
      if (changingSession) {
        $('#assistant_chat_reset_button').disabled = !!WAC.restorationStatus;
        $('#assistant_chat_reset_button').removeAttribute('aria-busy');
      }
    }
  }
  WAC.submitRequest = async (text, submissionId, steering = false) => {
    WAC.clearRequestInput(text);
    try {
      await WAC.waitForChatUpload();
      WAC.acceptRestorationSubmission(submissionId, await api('messages', {text, submission_id: submissionId, steering}));
    }
    catch (error) { WAC.dropOptimisticSubmit(submissionId); if (!WAC.requestInput().value) WAC.setRequestInputValue(text); notice(error.message); }
  };
  WAC.queueBusyRequest = (text, id) => {WAC.submitRequest(text, id); return true;};
  WAC.steerRequest = (text, id) => {WAC.submitRequest(text, id, true); return true;};
  WAC.queuedRequestAction = (action, message_id, text) => {control('queued', {action, message_id, text}); return true;};
  WAC.stopBridgeTargets = () => [{click: () => control('stop')}];
  WAC.pauseBridgeTargets = () => [{click: () => control('pause')}];
  WAC.requestCanonicalSync = () => {api('state').then(state => snapshot(state, true)).catch(error => notice(error.message)); return true;};
  WAC.resumeSelectedSession = trigger => {const picker = trigger.closest('.chat__session-picker').querySelector('select'); control('resume', {id: picker.value});};
  WAC.prefillResumedSession = () => {};
  WAC.handleEventNodeMutation = () => {};
  WAC.readEventSource = () => {};
  WAC.installEventBridge = () => {window.addEventListener('resize', () => WAC.scheduleComposerLayout());};
  document.addEventListener('DOMContentLoaded', () => {
    WAC.emptyMarkup = mode => {
      const prime = mode === 'prime';
      return `<div class="chat__empty-card chat__empty-card--web">
        <header class="chat__empty-header"><h2 class="chat__empty-title">Deepy ${prime ? 'Prime' : 'Zero'}</h2></header>
        <p class="chat__empty-intro">${prime ? 'Describe your idea. Deepy takes it from there.' : 'Create with a simple request.'}</p>
        <section class="chat__empty-section"><ul>
          <li>Create and edit images, videos, speech and music.</li>
          <li>${prime ? 'Let Deepy choose the models and plan the steps.' : 'Use your preferred models and settings.'}</li>
          <li>${prime ? 'Build on your creations, one request at a time.' : 'Make quick edits to your latest creation.'}</li>
        </ul></section>
        <p class="chat__empty-tip">Try: “${prime ? 'Turn my selfie into a superhero.' : 'Animate this photo.'}”</p>
        ${WAC.sessionPickerMarkup()}
      </div>`;
    };
    WAC.empty().innerHTML = WAC.emptyMarkup(WAC.host().dataset.deepyType);
    const request = WAC.requestInput();
    // Input keeps the pending or beforeinput intent; a fresh capture here would see the previous keystroke's growth.
    const resizeRequest = scrollState => {
      request.style.height = 'auto';
      request.style.height = request.scrollHeight + 'px';
      WAC.scheduleComposerLayout(scrollState);
    };
    request.addEventListener('input', () => resizeRequest());
    // Resize events arrive after the viewport has changed; retain the preceding scroll intent.
    let viewportScrollState = WAC.captureAutoscrollState();
    WAC.scroll().addEventListener('scroll', () => { viewportScrollState = WAC.captureAutoscrollState(); }, {passive: true});
    window.addEventListener('resize', () => resizeRequest(viewportScrollState));
    const composer = $('#assistant_chat_controls');
    WAC.mountChatUpload(api, notice, gallery);
    // Preserve input focus without cancelling WebKit's synthesized touch click.
    composer.addEventListener('mousedown', event => { if (event.button === 0 && event.target.closest('button')) event.preventDefault(); });
    const statusNode = WAC.statusNode();
    const composerObserver = new ResizeObserver(() => {
      if (!composer.offsetHeight) return;
      const scrollState = WAC.composerResizeScrollState || WAC.captureAutoscrollState();
      WAC.panel().style.setProperty('--deepy-composer-height', composer.offsetHeight + 'px');
      statusNode.style.setProperty('--deepy-status-limit', composer.getBoundingClientRect().top - statusNode.parentElement.getBoundingClientRect().top - 8 + 'px');
      statusNode.style.setProperty('--deepy-status-height', statusNode.offsetHeight + 'px');
      WAC.scheduleComposerLayout(scrollState);
    });
    for (const node of [composer, statusNode, statusNode.parentElement]) composerObserver.observe(node);
    resizeRequest();
    document.querySelectorAll('.gallery-divider').forEach(installGalleryDivider);
    const settingsTabs = [...document.querySelectorAll('[data-settings-tab]')];
    for (const button of settingsTabs) {
      button.onclick = () => {
        for (const item of settingsTabs) { const active = item === button; item.setAttribute('aria-selected', String(active)); item.tabIndex = active ? 0 : -1; $('#' + item.dataset.settingsTab + '-settings').hidden = !active; }
      };
      button.onkeydown = event => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {event.preventDefault(); const next = settingsTabs.find(item => item !== button); next.click(); next.focus();} };
    }
    $('#deepy-settings-form').oninput = () => {$('#settings-status').textContent = 'Unsaved changes';};
    $('#deepy-settings-form').onsubmit = async event => {
      event.preventDefault(); $('#save-settings').disabled = true;
      const values = settingsValues();
      try { if (!renderSettings(await api('settings', {values, baseline:settingsBaseline}), values)) $('#settings-status').textContent = 'Settings saved'; }
      catch (error) {
        try {renderSettings(await api('settings'));} catch (_) { /* Keep the draft while disconnected. */ }
        $('#settings-status').textContent = error.message;
      }
      finally { $('#save-settings').disabled = false; }
    };
    $('#abort-generation').onclick = async () => {
      $('#abort-generation').disabled = true;
      try { const state = await api('control', {action: 'abort', payload: {}}); progress(state.progress); }
      catch (error) { $('#abort-generation').disabled = false; notice(error.message); }
    };
    $('#read-answer').onclick = () => {chatScrollState = {atBottom: true, top: 0}; tab('chat');};
    $('#dismiss-answer').onclick = () => {dismissedAnswer = $('#answer-notice').dataset.answerId; $('#answer-notice').hidden = true;};
    $('#close-media-viewer').onclick = () => $('#media-viewer').close();
    $('#media-viewer').addEventListener('close', () => {
      WanGPImageZoom.destroy($('#media-viewer img'));
      $('#media-viewer wangp-mesh-viewer')?.remove();
      for (const media of $('#media-viewer').querySelectorAll('img, video, audio')) {
        if (media.tagName !== 'IMG') media.pause();
        media.removeAttribute('src');
        if (media.tagName !== 'IMG') media.load();
      }
      if (mediaScrollState && currentTab === 'chat' && mediaScrollState.sessionId === WAC.chatSessionId) WAC.scheduleComposerLayout(mediaScrollState.scroll);
      mediaScrollState = null;
    });
    document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => tab(button.dataset.tab));
    $('#assistant_chat_reset_button').onclick = () => control('reset');
    $('#saved-session').onchange = () => control('resume', {id: $('#saved-session').value});
    const openSessionDialog = action => {
      const dialog = $('#rename-session-dialog');
      const deleting = action === 'delete';
      dialog.dataset.action = action;
      dialog.dataset.sessionId = WAC.activeSessionId;
      $('#rename-session-heading').textContent = deleting ? 'Delete session' : 'Rename session';
      $('#session-title').disabled = deleting; $('#session-title').closest('label').hidden = deleting;
      $('#session-title').value = activeSessionTitle;
      $('#session-delete-description').hidden = !deleting;
      $('#session-delete-description').textContent = `Delete “${activeSessionTitle}”? The session will be moved to recoverable trash.`;
      $('#save-session-title').textContent = deleting ? 'Delete' : 'Save';
      $('#rename-session-error').hidden = true;
      dialog.showModal();
      if (deleting) $('#cancel-session-rename').focus(); else $('#session-title').select();
    };
    $('#rename-session').onclick = () => openSessionDialog('rename');
    $('#delete-session').onclick = () => openSessionDialog('delete');
    $('#cancel-session-rename').onclick = () => $('#rename-session-dialog').close();
    $('#rename-session-form').onsubmit = async event => {
      event.preventDefault();
      const dialog = $('#rename-session-dialog'), save = $('#save-session-title');
      save.disabled = true;
      try {
        const deleting = dialog.dataset.action === 'delete';
        const state = await api('control', {action: dialog.dataset.action, payload: {id: dialog.dataset.sessionId, ...(deleting ? {confirmed: true} : {title: $('#session-title').value})}});
        if (deleting) snapshot(state); else sessions(state);
        dialog.close();
        if (deleting) notice('Session moved to recoverable trash.');
      } catch (error) { $('#rename-session-error').textContent = error.message; $('#rename-session-error').hidden = false; }
      finally { save.disabled = false; }
    };
    for (const input of document.querySelectorAll('#visual-upload, #audio-upload, #mesh-upload')) input.onchange = async () => {
      input.disabled = true;
      try { await WAC.uploadMediaFiles(Array.from(input.files), api, notice, gallery); notice('Import complete.'); }
      catch (error) {notice(error.message);} finally {input.disabled = false; input.value = '';}
    };
    const rememberBackgroundScroll = () => {
      if (currentTab === 'chat' && !backgroundChatState) backgroundChatState = {sessionId: WAC.chatSessionId, scroll: WAC.captureAutoscrollState()};
    };
    const viewport = $('meta[name=viewport]');
    const viewportContent = viewport.content;
    let resumeFrame = 0;
    const resume = () => {
      if (document.hidden) return;
      rememberBackgroundScroll();
      if (!window.matchMedia('(pointer: coarse)').matches) { connect(); return; }
      cancelAnimationFrame(resumeFrame);
      // WebKit can discard the mobile viewport on resume (bug 262207).
      // Reapply it across layout frames before restoring the chat scroll position.
      viewport.content = viewportContent.replace('viewport-fit=cover', 'viewport-fit=contain');
      resumeFrame = requestAnimationFrame(() => {
        viewport.content = viewportContent;
        resumeFrame = requestAnimationFrame(() => { resumeFrame = 0; if (!document.hidden) connect(); });
      });
    };
    document.addEventListener('visibilitychange', () => { if (document.hidden) rememberBackgroundScroll(); else resume(); });
    for (const type of ['gesturestart', 'gesturechange']) document.addEventListener(type, event => event.preventDefault(), {passive: false});
    for (const type of ['contextmenu', 'dragstart']) document.addEventListener(type, event => {
      if (window.matchMedia('(pointer: coarse)').matches && event.target.closest('.media-card img, .media-actions a, #media-viewer img')) event.preventDefault();
    });
    window.addEventListener('online', connect);
    window.addEventListener('offline', () => { transport.close(); connection(false); });
    window.addEventListener('pagehide', () => {rememberBackgroundScroll(); transport.close();});
    window.addEventListener('pageshow', event => {if (event.persisted) resume();});
    window.addEventListener('blur', rememberBackgroundScroll);
    window.addEventListener('focus', resume);
    tab('chat'); connect();
  });
})();
