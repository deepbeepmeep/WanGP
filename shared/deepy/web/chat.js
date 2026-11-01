// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

window.__wangpAssistantChatNS = window.__wangpAssistantChatNS || {};
window.__wangpAssistantChatPending = window.__wangpAssistantChatPending || [];
const WAC = window.__wangpAssistantChatNS;
WAC.replayDepth = Number(WAC.replayDepth || 0);
window.WAC = WAC;

WAC.state = WAC.state || { order: [], messages: {}, status: null, stats: null };
WAC.blockState = WAC.blockState || {};
WAC.streamingReveals = WAC.streamingReveals || new Map();
WAC.streamingRevealFrame = WAC.streamingRevealFrame || 0;
WAC.init = WAC.init || false;
WAC.observer = WAC.observer || null;
WAC.eventNode = WAC.eventNode || null;
WAC.lastPayloadId = WAC.lastPayloadId || '';
WAC.lastPayloadText = WAC.lastPayloadText || '';
WAC.dockBridgeInstalled = WAC.dockBridgeInstalled || false;
WAC.dockOpen = typeof WAC.dockOpen === 'boolean' ? WAC.dockOpen : false;
WAC.settingsOpen = typeof WAC.settingsOpen === 'boolean' ? WAC.settingsOpen : false;
WAC.disclosureNode = WAC.disclosureNode || null;
WAC.disclosureState = WAC.disclosureState || {};
WAC.composerLayoutMode = WAC.composerLayoutMode || '';
WAC.composerResizeScrollState = WAC.composerResizeScrollState || null;
WAC.composerResizeFrame = WAC.composerResizeFrame || 0;
WAC.followSubmissionId = WAC.followSubmissionId || '';
WAC.followSubmissionScrollFrame = WAC.followSubmissionScrollFrame || 0;
WAC.sessionCatalog = Array.isArray(WAC.sessionCatalog) ? WAC.sessionCatalog : [];
WAC.activeSessionId = WAC.activeSessionId || '';
WAC.multiSessionEnabled = !!WAC.multiSessionEnabled;
WAC.sessionCatalogInitialized = !!WAC.sessionCatalogInitialized;

WAC.dock = function () {
  return document.querySelector('#assistant_chat_dock');
};

WAC.panel = function () {
  return document.querySelector('#assistant_chat_panel');
};

WAC.launcher = function () {
  return document.querySelector('#assistant_chat_toggle');
};

WAC.settingsPanel = function () {
  return document.querySelector('#assistant_chat_settings_panel');
};

WAC.settingsLauncher = function () {
  return document.querySelector('#assistant_chat_settings_toggle');
};

WAC.requestInput = function () {
  return document.querySelector('#assistant_chat_request textarea, #assistant_chat_request input');
};

WAC.resetComposerLayout = function () {
  const panel = WAC.panel();
  if (!panel) return;
  panel.classList.remove('has-fixed-composer-layout');
  panel.style.removeProperty('height');
  panel.style.removeProperty('--chat-request-max-height');
  panel.dataset.composerLayoutMode = '';
  WAC.composerLayoutMode = '';
};

WAC.syncComposerLayout = function () {
  if (WAC.replayDepth > 0) return;
  const dock = WAC.dock();
  const panel = WAC.panel();
  const shellBlock = document.querySelector('#assistant_chat_shell_block');
  const input = WAC.requestInput();
  if (!dock || !panel || !shellBlock || !input || !WAC.dockOpen || window.getComputedStyle(panel).display === 'none') return;
  const mode = window.innerWidth <= 900 ? 'mobile' : 'desktop';
  if (panel.dataset.composerLayoutMode === mode && panel.classList.contains('has-fixed-composer-layout')) return;
  WAC.resetComposerLayout();
  // Measure layout sizes, independent of the dock's opening transform.
  const naturalHeight = panel.offsetHeight;
  const shellHeight = shellBlock.offsetHeight;
  const inputHeight = input.offsetHeight;
  if (naturalHeight <= 0 || shellHeight <= 0 || inputHeight <= 0) return;
  const historyMinHeight = parseFloat(window.getComputedStyle(dock).getPropertyValue('--chat-history-min-height')) || 112;
  const viewportLimit = Math.max(320, window.innerHeight - 36);
  const panelHeight = Math.min(naturalHeight, viewportLimit);
  const nonHistoryHeight = Math.max(0, naturalHeight - shellHeight);
  const availableHistoryHeight = Math.max(historyMinHeight, panelHeight - nonHistoryHeight);
  const maxRequestHeight = Math.max(inputHeight, inputHeight + availableHistoryHeight - historyMinHeight);
  panel.style.height = `${panelHeight}px`;
  panel.style.setProperty('--chat-request-max-height', `${maxRequestHeight}px`);
  panel.dataset.composerLayoutMode = mode;
  panel.classList.add('has-fixed-composer-layout');
  WAC.composerLayoutMode = mode;
};

WAC.scheduleComposerLayout = function (scrollState) {
  if (scrollState) WAC.composerResizeScrollState = scrollState;
  else if (!WAC.composerResizeScrollState) WAC.composerResizeScrollState = WAC.captureAutoscrollState();
  if (WAC.composerResizeFrame) window.cancelAnimationFrame(WAC.composerResizeFrame);
  const scroll = WAC.scroll(), initialTop = scroll?.scrollTop;
  WAC.composerResizeFrame = window.requestAnimationFrame(() => {
    WAC.composerResizeFrame = window.requestAnimationFrame(() => {
      WAC.composerResizeFrame = 0;
      const state = scroll?.scrollTop !== initialTop ? WAC.captureAutoscrollState() : WAC.composerResizeScrollState;
      WAC.syncComposerLayout();
      // Pasting scrolls the caret line into view but not the field's bottom padding.
      const input = WAC.requestInput();
      if (input && input.selectionStart === input.value.length) input.scrollTop = input.scrollHeight;
      WAC.composerResizeScrollState = null;
      WAC.applyAutoscrollState(state);
    });
  });
};

WAC.escapeHtml = function (value) {
  return String(value || '').replace(/[&<>\"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[char] || char));
};

WAC.timeLabel = function (timestamp) {
  const value = Number(timestamp);
  return Number.isFinite(value) && value > 0 ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
};

WAC.bottomThreshold = function () {
  return 1;
};

WAC.captureAutoscrollState = function () {
  if (WAC.replayDepth > 0) return null;
  if (WAC.compactScrollState) return WAC.compactScrollState;
  const scroll = WAC.scroll();
  if (!scroll) return { atBottom: true, top: 0 };
  // Follow the first message, but retain the welcome page's reading position.
  return {
    atBottom: WAC.state.order.length === 0 || WAC.isNearBottom(),
    top: Math.max(0, scroll.scrollTop),
  };
};

WAC.applyAutoscrollState = function (state) {
  if (WAC.replayDepth > 0 || WAC.compactScrollState) return;
  const scroll = WAC.scroll();
  if (!scroll) return;
  if (state) {
    const top = state.atBottom && WAC.state.order.length > 0 ? Math.max(0, scroll.scrollHeight - scroll.clientHeight) : Math.max(0, Number(state.top || 0));
    // Writing even the current position can interrupt native touch/smooth scrolling.
    if (scroll.scrollTop !== top) scroll.scrollTop = top;
  }
  if (scroll.clientHeight > 0) WAC.lastScrollState = WAC.captureAutoscrollState();
  WAC.syncJumpToBottom();
};

const optimisticProtocolVersion = 2;
if (WAC.optimisticProtocolVersion !== optimisticProtocolVersion) {
  WAC.optimisticSubmits = [];
  WAC.state.order = WAC.state.order.filter((messageId) => {
    const optimistic = String(messageId || '').startsWith('optimistic_');
    if (optimistic) delete WAC.state.messages[messageId];
    return !optimistic;
  });
  WAC.optimisticProtocolVersion = optimisticProtocolVersion;
}
WAC.optimisticSubmits = Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits : [];
WAC.serverInstanceId = WAC.serverInstanceId || '';
WAC.chatSessionId = WAC.chatSessionId || '';
WAC.chatRevision = Number.isFinite(Number(WAC.chatRevision)) ? Number(WAC.chatRevision) : -1;
WAC.chatSequence = Number.isFinite(Number(WAC.chatSequence)) ? Number(WAC.chatSequence) : -1;
WAC.syncRequired = !!WAC.syncRequired;
WAC.syncRecoveryPending = !!WAC.syncRecoveryPending;
WAC.optimisticMaxAgeMs = 30000;
WAC.pendingSteeringId = WAC.pendingSteeringId || '';
WAC.queuedEditMessageId = WAC.queuedEditMessageId || '';
WAC.queuedEditDraft = typeof WAC.queuedEditDraft === 'string' ? WAC.queuedEditDraft : '';

WAC.normalizeText = function (value) {
  return String(value || '').replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ').trim();
};

WAC.gradioConfig = function () {
  return window.gradio_config || window.__gradio_config__ || null;
};

WAC.componentNode = function (id) {
  if (id === null || typeof id === 'undefined') return null;
  return document.getElementById(`component-${id}`);
};

WAC.isVisibleNode = function (node) {
  if (!node) return false;
  const style = window.getComputedStyle(node);
  if (style.display === 'none' || style.visibility === 'hidden') return false;
  const rect = node.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};

WAC.dropdownChoiceTexts = function (component) {
  const rawChoices = component && component.props ? component.props.choices : [];
  if (!Array.isArray(rawChoices)) return [];
  const texts = [];
  for (const choice of rawChoices) {
    if (Array.isArray(choice)) {
      texts.push(String(choice[0] || '').toLowerCase());
      texts.push(String(choice[1] || '').toLowerCase());
      continue;
    }
    if (choice && typeof choice === 'object') {
      texts.push(String(choice.label || choice.name || '').toLowerCase());
      texts.push(String(choice.value || '').toLowerCase());
      continue;
    }
    texts.push(String(choice || '').toLowerCase());
  }
  return texts;
};

WAC.findWanGpSettingsDropdown = function () {
  const cfg = WAC.gradioConfig();
  const components = cfg && Array.isArray(cfg.components) ? cfg.components : [];
  let fallback = null;
  for (const component of components) {
    if (!component || String(component.type || '').toLowerCase() !== 'dropdown') continue;
    const texts = WAC.dropdownChoiceTexts(component);
    const hasSettings = texts.some((text) => text.includes('>settings'));
    const hasProfiles = texts.some((text) => text.includes('>profiles'));
    const hasLoraPresetHint = texts.some((text) => text.includes('lora preset'));
    if (!hasSettings || (!hasProfiles && !hasLoraPresetHint)) continue;
    const node = WAC.componentNode(component.id);
    if (node && WAC.isVisibleNode(node)) return { component, node };
    if (!fallback) fallback = { component, node };
  }
  return fallback;
};

WAC.getWanGpSettingsSelection = function () {
  const located = WAC.findWanGpSettingsDropdown();
  if (!located || !located.component) return { value: '', label: '' };
  const component = located.component;
  const node = located.node || WAC.componentNode(component.id);
  const input = node ? node.querySelector('input[role="listbox"], input, textarea') : null;
  const label = WAC.normalizeText(input ? (input.value || input.getAttribute('value') || '') : '');
  const value = WAC.normalizeText(component && component.props ? component.props.value : '');
  return { value, label };
};

WAC.buildOptimisticUserMessage = function (optimisticId, content, timestamp, badgeText) {
  const contentHtml = WAC.escapeHtml(content).replace(/\n/g, '<br>');
  const badge = WAC.normalizeText(badgeText);
  const badgeHtml = badge ? `<span class='chat__badge'>${WAC.escapeHtml(badge)}</span>` : '';
  const copyButton = `<button type='button' class='chat__copy-button' data-copy-source='user' data-copy-text='${WAC.escapeHtml(content)}' aria-label='Copy request' title='Copy request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><rect x='5' y='5' width='8' height='8' rx='1.5'></rect><path d='M3.5 10.5H3A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5'></path></svg></button>`;
  const queuedActions = badge === 'Queued' ? `<button type='button' class='chat__message-action-button' data-message-action='steer' aria-label='Steer with this queued request' title='Steer with this queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M2.5 8h9M8.5 4l4 4-4 4'></path></svg></button><button type='button' class='chat__message-action-button' data-message-action='edit' aria-label='Edit queued request' title='Edit queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 11.8 3.5 9l6.8-6.8a1.4 1.4 0 0 1 2 0l1.5 1.5a1.4 1.4 0 0 1 0 2L7 12.5l-2.8.5Z'></path><path d='m9.4 3.1 3.5 3.5'></path></svg></button><button type='button' class='chat__message-action-button' data-message-action='remove' aria-label='Remove queued request' title='Remove queued request'><svg viewBox='0 0 16 16' aria-hidden='true' focusable='false'><path d='M3 4.5h10M6 4.5V2.7h4v1.8M4.7 4.5l.6 8.3h5.4l.6-8.3M7 7v3.4M9 7v3.4'></path></svg></button>` : '';
  const html = [
    `<article class='chat__message chat__message--user' data-message-id='${optimisticId}'>`,
    "<div class='chat__avatar'>You</div>",
    "<div class='chat__message-card'>",
    `<div class='chat__meta'><div class='chat__meta-left'>${badgeHtml}</div>`,
    `<div class='chat__meta-right'><div class='chat__message-actions'>${copyButton}${queuedActions}</div><div class='chat__time'>${WAC.escapeHtml(WAC.timeLabel(timestamp))}</div></div></div>`,
    `<div class='chat__body'><p>${contentHtml}</p></div>`,
    "</div></article>",
  ].join('');
  return { id: optimisticId, role: 'user', html, badge, queued: badge === 'Queued' || badge === 'Steered', client_submission_id: optimisticId };
};

WAC.dropOptimisticSubmit = function (optimisticId) {
  const targetId = String(optimisticId || '');
  WAC.optimisticSubmits = (WAC.optimisticSubmits || []).filter((item) => String(item && item.id || '') !== targetId);
  if (WAC.followSubmissionId === targetId) WAC.followSubmissionId = '';
};

WAC.queuedTailInsertIndex = function () {
  let index = WAC.state.order.length;
  while (index > 0) {
    const message = WAC.state.messages[WAC.state.order[index - 1]];
    if (!message || message.role !== 'user' || !message.queued) break;
    index -= 1;
  }
  return index;
};

WAC.clearRequestInput = function (expectedText) {
  const input = WAC.requestInput();
  if (!input) return;
  const current = WAC.normalizeText(input.value || '');
  const expected = WAC.normalizeText(expectedText || '');
  if (expected && current && current !== expected) return;
  if (window.matchMedia('(pointer: coarse)').matches && document.activeElement === input) input.blur();
  input.value = '';
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
};

WAC.reconcileOptimisticSubmits = function (acknowledgedSubmissionIds) {
  const acknowledged = new Set((Array.isArray(acknowledgedSubmissionIds) ? acknowledgedSubmissionIds : []).map((value) => String(value || '').trim()).filter(Boolean));
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    const submissionId = String(message && message.client_submission_id || '').trim();
    if (submissionId) acknowledged.add(submissionId);
  }
  const now = Date.now();
  WAC.optimisticSubmits = (Array.isArray(WAC.optimisticSubmits) ? WAC.optimisticSubmits : []).filter((item) => {
    const submissionId = String(item && item.id || '').trim();
    const timestamp = Number(item && item.ts || 0);
    return submissionId && !acknowledged.has(submissionId) && timestamp > 0 && (item.accepted || now - timestamp < WAC.optimisticMaxAgeMs);
  });
  for (const item of WAC.optimisticSubmits) {
    const optimisticId = String(item && item.id || '').trim();
    const content = WAC.normalizeText(item && item.text || '');
    if (!optimisticId || !content || WAC.state.messages[optimisticId]) continue;
    const message = WAC.buildOptimisticUserMessage(optimisticId, content, item.ts, item.badge);
    const messageIndex = message.badge === 'Steered' ? WAC.queuedTailInsertIndex() : WAC.state.order.length;
    WAC.state.order.splice(messageIndex, 0, optimisticId);
    WAC.state.messages[optimisticId] = message;
  }
};

WAC.newSubmissionId = function () {
  const randomId = window.crypto && typeof window.crypto.randomUUID === 'function' ? window.crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return `optimistic_${randomId}`;
};

WAC.pushOptimisticUserMessage = function (text, badgeText) {
  const content = WAC.normalizeText(text);
  if (!content) return '';
  const now = Date.now();
  const optimisticId = WAC.newSubmissionId();
  const badge = WAC.normalizeText(badgeText);
  WAC.followSubmissionId = optimisticId;
  WAC.optimisticSubmits.push({ id: optimisticId, text: content, ts: now, badge });
  const message = WAC.buildOptimisticUserMessage(optimisticId, content, now, badge);
  WAC.upsertMessage(message, false, badge === 'Steered' ? WAC.queuedTailInsertIndex() : undefined);
  WAC.scrollToBottomAfterLayout();
  window.setTimeout(() => {
    if (!(WAC.optimisticSubmits || []).some((item) => String(item && item.id || '') === optimisticId && !item.accepted)) return;
    WAC.dropOptimisticSubmit(optimisticId);
    WAC.removeMessage(optimisticId);
  }, WAC.optimisticMaxAgeMs);
  return optimisticId;
};

WAC.acceptRestorationSubmission = function (submissionId, response) {
  if (!response.queued_for_restoration) return;
  const item = WAC.optimisticSubmits.find(item => item.id === submissionId);
  if (item) item.accepted = true;
};

WAC.host = function () {
  return document.querySelector('#assistant_chat_html');
};

WAC.shell = function () {
  return document.querySelector('#assistant_chat_html .chat');
};

WAC.scroll = function () {
  return document.querySelector('#assistant_chat_html .chat__scroll');
};

WAC.transcript = function () {
  return document.querySelector('#assistant_chat_html .chat__transcript');
};

WAC.empty = function () {
  return document.querySelector('#assistant_chat_html .chat__empty');
};

WAC.acknowledgeOptimisticSubmits = function (submissionIds) {
  const acknowledged = new Set((Array.isArray(submissionIds) ? submissionIds : []).map((value) => String(value || '').trim()).filter(Boolean));
  if (acknowledged.size === 0) return;
  WAC.optimisticSubmits = (WAC.optimisticSubmits || []).filter((item) => !acknowledged.has(String(item && item.id || '').trim()));
  for (const submissionId of acknowledged) delete WAC.state.messages[submissionId];
  WAC.state.order = WAC.state.order.filter((messageId) => !acknowledged.has(String(messageId || '')));
};

WAC.mergeStaleSync = function (messages, acknowledgedSubmissionIds) {
  WAC.ensureShell();
  const followSubmittedRequest = WAC.syncAcknowledgesFollowedSubmission(messages, acknowledgedSubmissionIds);
  const scrollState = followSubmittedRequest ? { atBottom: true, top: 0 } : WAC.captureAutoscrollState();
  WAC.acknowledgeOptimisticSubmits(acknowledgedSubmissionIds);
  const snapshotOrder = [];
  for (const message of (Array.isArray(messages) ? messages : [])) {
    if (!message || !message.id) continue;
    const messageId = String(message.id);
    snapshotOrder.push(messageId);
    if (!WAC.state.messages[messageId]) WAC.state.messages[messageId] = message;
  }
  const snapshotIds = new Set(snapshotOrder);
  WAC.state.order = snapshotOrder.concat(WAC.state.order.filter((messageId) => !snapshotIds.has(String(messageId))));
  WAC.hydrate(scrollState);
  if (followSubmittedRequest) {
    WAC.followSubmissionId = '';
    WAC.scrollToBottomAfterLayout();
  }
};

WAC.readSessionCatalogFromHost = function () {
  if (WAC.sessionCatalogInitialized) return;
  const host = WAC.host();
  if (!host) return;
  try {
    const catalog = JSON.parse(String(host.dataset.sessionCatalog || '[]'));
    if (Array.isArray(catalog)) WAC.sessionCatalog = catalog;
  } catch (_error) {}
  WAC.activeSessionId = String(host.dataset.activeSessionId || WAC.activeSessionId || '');
  WAC.multiSessionEnabled = String(host.dataset.multiSessionEnabled || '').toLowerCase() === 'true';
  WAC.sessionCatalogInitialized = true;
};

WAC.syncSessionActionTooltips = function () {
  const labels = {
    assistant_chat_session_resume_button: 'Resume the selected session',
    assistant_chat_session_rename_button: 'Rename the selected session',
    assistant_chat_session_duplicate_button: 'Duplicate the selected session',
    assistant_chat_session_export_button: 'Export the selected session',
    assistant_chat_session_import_button: 'Import a session archive',
    assistant_chat_session_delete_button: 'Delete the selected session',
  };
  for (const [id, label] of Object.entries(labels)) {
    const host = document.getElementById(id);
    if (!host) continue;
    const button = host.matches && host.matches('button') ? host : host.querySelector('button');
    if (!button) continue;
    button.title = label;
    button.setAttribute('aria-label', label);
  }
};

WAC.sessionPickerMarkup = function () {
  if (!WAC.multiSessionEnabled) return '';
  const options = [];
  for (const item of WAC.sessionCatalog) {
    const id = String(item && item.id || '').trim();
    if (!id) continue;
    const title = String(item && item.title || 'Deepy session').trim();
    const selected = id === WAC.activeSessionId ? ' selected' : '';
    options.push(`<option value="${WAC.escapeHtml(id)}"${selected}>${WAC.escapeHtml(title)}</option>`);
  }
  const disabled = options.length === 0 ? ' disabled' : '';
  const choices = options.length ? options.join('') : '<option value="">No saved sessions</option>';
  return `<div class="chat__session-picker"><div class="chat__session-picker-spacer" aria-hidden="true"></div><label><span>Saved sessions</span><span class="chat__session-picker-controls"><select aria-label="Deepy session" data-wac-session-picker${disabled}>${choices}</select><button type="button" data-wac-session-resume aria-label="Resume selected session" title="Resume selected session"${disabled}>Resume</button></span></label></div>`;
};



WAC.bindSessionPickerControls = function (root) {
  const scope = root && root.querySelectorAll ? root : document;
  for (const button of scope.querySelectorAll('[data-wac-session-resume]')) {
    if (button.dataset.wacSessionResumeBound === 'true') continue;
    button.dataset.wacSessionResumeBound = 'true';
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      WAC.resumeSelectedSession(button);
    });
  }
};

WAC.refreshSessionPickers = function () {
  const markup = WAC.sessionPickerMarkup();
  const existing = document.querySelectorAll('.chat__session-picker');
  if (!markup) {
    for (const node of existing) node.remove();
    return;
  }
  if (existing.length) {
    for (const node of existing) node.outerHTML = markup;
    WAC.bindSessionPickerControls();
    return;
  }
  const card = document.querySelector('#assistant_chat_html .chat__empty-card');
  if (card) {
    card.insertAdjacentHTML('beforeend', markup);
    WAC.bindSessionPickerControls(card);
  }
};

WAC.emptyMarkup = function (mode) {
  if (mode === 'prime') {
    return `<div class="chat__empty-card">
      <header class="chat__empty-header">
        <span class="chat__empty-eyebrow">Current assistant</span>
        <h2 class="chat__empty-title">Deepy Prime</h2>
        <span class="chat__empty-mode">Advanced creative orchestration</span>
      </header>
      <p class="chat__empty-intro">Describe the result you want and Deepy Prime can plan the work, choose suitable models and tools, and connect multiple image, video, and audio steps into one creative workflow.</p>
      <div class="chat__empty-grid">
        <section class="chat__empty-section"><h3>What it does for you</h3><ul>
          <li>Plan and complete multi-step projects that create, inspect, edit, and combine several pieces of media.</li>
          <li>Choose among available WanGP models and settings according to your goal, quality preference, and source media.</li>
          <li>Build on Gallery items or existing files, then extract, transcribe, resize, add sound, upscale, or continue generating.</li>
          <li>Extend the workflow with other connected services when they are available.</li>
        </ul></section>
        <section class="chat__empty-section chat__empty-section--examples"><h3>Try asking</h3><ul>
          <li>Create a character portrait and related keyframes, then turn them into a longer video with a soundtrack.</li>
          <li>Inspect the selected video, improve the weak sections, upscale it, and prepare a subtitled version.</li>
          <li>Design an album cover, write a matching song, and create a short promotional video from both.</li>
        </ul></section>
      </div>
      <p class="chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
      ${WAC.sessionPickerMarkup()}
    </div>`;
  }
  return `<div class="chat__empty-card">
    <header class="chat__empty-header">
      <span class="chat__empty-eyebrow">Current assistant</span>
      <h2 class="chat__empty-title">Deepy Zero</h2>
      <span class="chat__empty-mode">Fast, focused creation</span>
    </header>
    <p class="chat__empty-intro">Deepy Zero is the lightweight assistant for straightforward requests. It uses the models and templates selected in Deepy Settings, making it a good match for smaller LLMs, quick responses, and familiar results.</p>
    <div class="chat__empty-grid">
      <section class="chat__empty-section"><h3>What it does for you</h3><ul>
        <li>Generate an image, video, speech clip, or song with your preferred templates and defaults.</li>
        <li>Handle focused edits and practical media tasks without requiring a complex workflow.</li>
        <li>Refer naturally to the selected, latest, or previous Gallery item.</li>
        <li>See each generation and completed result in the normal WanGP queue and Galleries.</li>
      </ul></section>
      <section class="chat__empty-section chat__empty-section--examples"><h3>Try asking</h3><ul>
        <li>Generate a square album cover showing a robot jazz band.</li>
        <li>Animate the selected image as a five-second cinematic shot.</li>
        <li>Transcribe the last video or resize it for social media.</li>
      </ul></section>
    </div>
    <p class="chat__empty-tip">Start with the outcome you want. Deepy will ask only when an important choice is missing.</p>
    ${WAC.sessionPickerMarkup()}
  </div>`;
};

WAC.syncDeepyTypePreview = function () {
  const canonical = document.querySelector('#deepy_type_value');
  const value = String((canonical && canonical.textContent) || '').trim().toLowerCase();
  if (!value) return;
  const mode = value === 'prime' ? 'prime' : 'zero';
  const host = WAC.host();
  const empty = WAC.empty();
  if (!host || !empty || host.dataset.deepyType === mode) return;
  host.dataset.deepyType = mode;
  empty.innerHTML = WAC.emptyMarkup(mode);
};

WAC.statusNode = function () {
  return document.querySelector('#assistant_chat_html .chat__status');
};

WAC.jumpBottomNode = function () {
  return document.querySelector('#assistant_chat_html .chat__jump-bottom');
};

WAC.statsNode = function () {
  return document.getElementById('assistant_chat_stats');
};

WAC.disclosureKey = function (node) {
  if (!node || !node.getAttribute) return '';
  const reasoningId = String(node.getAttribute('data-reasoning-id') || '').trim();
  if (reasoningId) return `reasoning:${reasoningId}`;
  const toolId = String(node.getAttribute('data-tool-id') || '').trim();
  if (toolId) return `tool:${toolId}`;
  const contextSummaryId = String(node.getAttribute('data-context-summary-id') || '').trim();
  if (contextSummaryId) return `context-summary:${contextSummaryId}`;
  return '';
};

WAC.captureDisclosureState = function (root) {
  if (WAC.replayDepth > 0) return;
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  const nodes = [...scope.querySelectorAll('.chat__disclosure')];
  if (scope.matches && scope.matches('.chat__disclosure')) nodes.unshift(scope);
  nodes.forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key) return;
    WAC.disclosureState[key] = !!node.open;
  });
};

WAC.applyDisclosureState = function (root) {
  if (WAC.replayDepth > 0) return;
  const scope = root || WAC.transcript();
  if (!scope || !scope.querySelectorAll) return;
  const nodes = [...scope.querySelectorAll('.chat__disclosure')];
  if (scope.matches && scope.matches('.chat__disclosure')) nodes.unshift(scope);
  nodes.forEach((node) => {
    const key = WAC.disclosureKey(node);
    if (!key || !(key in WAC.disclosureState)) return;
    node.open = !!WAC.disclosureState[key];
  });
};

WAC.handleDisclosureToggle = function (event) {
  if (WAC.replayDepth > 0) return;
  const node = event && event.target;
  if (node && node.tagName === 'DETAILS' && !node.open) WAC.clearStreamingReveals(node, true);
  if (!node || !node.classList || !node.classList.contains('chat__disclosure')) return;
  const key = WAC.disclosureKey(node);
  if (!key) return;
  WAC.disclosureState[key] = !!node.open;
};

WAC.toggleDisclosure = function (node, pointerY, fromBottom = false) {
  if (!node || !node.matches('.chat__disclosure, .chat__compact-section')) return;
  const scroll = WAC.scroll();
  const followBottom = (!node.open || fromBottom) && WAC.isNearBottom();
  const summary = node.querySelector(':scope > summary');
  const before = summary.getBoundingClientRect();
  let targetY = before.top + before.height / 2;
  if (node.open && Number.isFinite(pointerY) && scroll) {
    const bounds = scroll.getBoundingClientRect();
    if (before.top < bounds.top) targetY = Math.min(pointerY, bounds.bottom - before.height / 2);
  }
  node.open = !node.open;
  WAC.clearStreamingReveals(node, true);
  const key = node.matches('.chat__disclosure') ? WAC.disclosureKey(node) : '';
  if (key) WAC.disclosureState[key] = !!node.open;
  // Settle compact media/statement placement before anchoring, in the same frame.
  // Opening or closing from below preserves active bottom tracking; otherwise anchor the heading.
  node._syncDisclosureLayout?.();
  if (scroll) {
    const after = summary.getBoundingClientRect();
    const top = followBottom ? Math.max(0, scroll.scrollHeight - scroll.clientHeight) : scroll.scrollTop + after.top + after.height / 2 - targetY;
    if (scroll.scrollTop !== top) scroll.scrollTop = top; // Native bounds avoid blank space below the chat.
  }
  WAC.handleScroll();
};

WAC.closeDisclosure = function (node, pointerY) {
  if (!node || !node.open) return;
  WAC.toggleDisclosure(node, pointerY, true);
  node.querySelector(':scope > summary').focus({ preventScroll: true });
};

WAC.markCopyButton = function (button, state) {
  if (!button) return;
  const originalLabel = String(button.dataset.copyLabel || button.getAttribute('aria-label') || 'Copy');
  button.dataset.copyLabel = originalLabel;
  button.classList.toggle('is-copied', state === 'copied');
  button.classList.toggle('is-copy-error', state === 'error');
  const label = state === 'copied' ? 'Copied' : state === 'error' ? 'Copy failed' : originalLabel;
  button.setAttribute('aria-label', label);
  button.setAttribute('title', label);
  if (button.copyResetTimer) window.clearTimeout(button.copyResetTimer);
  if (state) button.copyResetTimer = window.setTimeout(() => { WAC.markCopyButton(button, ''); }, 1200);
};

WAC.handleCopyButtonClick = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('.chat__copy-button') : null;
  if (!button) return false;
  event.preventDefault();
  event.stopPropagation();
  const source = String(button.getAttribute('data-copy-source') || '');
  const jsonNode = source === 'json' || source === 'stdout' ? button.closest('.chat__tool-json') : null;
  // 'stdout' copies a Python run's printed output as it is now (it keeps growing while the script runs)
  const pre = jsonNode ? (source === 'stdout' && jsonNode.querySelector('.chat__tool-stdout')) || jsonNode.querySelector('pre') : null;
  const text = jsonNode ? String((pre && pre.textContent) || '') : String(button.getAttribute('data-copy-text') || '');
  if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
    WAC.markCopyButton(button, 'error');
    return true;
  }
  navigator.clipboard.writeText(text).then(() => { WAC.markCopyButton(button, 'copied'); }).catch(() => { WAC.markCopyButton(button, 'error'); });
  return true;
};

WAC.handleCollapseButtonClick = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('[data-disclosure-action="collapse"]') : null;
  if (!button) return false;
  event.preventDefault();
  event.stopPropagation();
  WAC.closeDisclosure(button.closest('.chat__disclosure'), event.detail ? event.clientY : undefined);
  return true;
};

WAC.handleCollapseButtonPointerDown = function (event) {
  const button = event && event.target && event.target.closest ? event.target.closest('[data-disclosure-action="collapse"]') : null;
  if (!button) return false;
  const isPrimaryPointer = event.button === 0 || event.pointerType === 'touch' || event.pointerType === 'pen';
  if (!isPrimaryPointer) return false;
  if (WAC.startDisclosurePointer(event, button, true)) return true;
  event.preventDefault();
  event.stopPropagation();
  return true;
};

WAC.handleDisclosurePointerDown = function (event) {
  const summary = event && event.target && event.target.closest ? event.target.closest('summary') : null;
  if (!summary) return false;
  const disclosureNode = summary.parentElement;
  if (!disclosureNode || !disclosureNode.matches('.chat__disclosure, .chat__compact-section')) return false;
  if (WAC.startDisclosurePointer(event, summary, false)) return true;
  if (event.button !== 0) return false;
  event.preventDefault();
  event.stopPropagation();
  return true;
};

WAC.startDisclosurePointer = function (event, node, collapse) {
  if (!['mouse', 'touch', 'pen'].includes(event.pointerType) || (event.pointerType === 'mouse' && event.button !== 0)) return false;
  WAC.disclosurePointer = {id: event.pointerId, x: event.clientX, y: event.clientY, node, collapse, moved: !event.isPrimary};
  if (event.pointerType === 'mouse') {
    event.preventDefault();
    event.stopPropagation();
  }
  return true;
};

WAC.trackDisclosurePointer = function (event) {
  const pointer = WAC.disclosurePointer;
  if (!pointer || pointer.id !== event.pointerId) return;
  if (event.type === 'pointercancel' || Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 10) pointer.moved = true;
  if (event.type !== 'pointerup' || pointer.moved || !pointer.node.isConnected) return;
  // A live title update can suppress the browser's click. Complete the original
  // press here, then consume its compatibility click even if layout moved its target.
  const node = pointer.node.closest('.chat__disclosure, .chat__compact-section');
  if (pointer.collapse) WAC.closeDisclosure(node, event.clientY);
  else WAC.toggleDisclosure(node);
};

// Media links in replies open like the attachment thumbnails.
WAC.attachmentSelector = 'a.chat__attachment, .chat__transcript a[data-media-kind], a.chat__pending-upload-link';

WAC.handleAttachmentPointerDown = function (event) {
  WAC.attachmentTouch = null;
  const link = event && event.target && event.target.closest ? event.target.closest(WAC.attachmentSelector) : null;
  if (!link) return false;
  WAC.attachmentTouch = event.pointerType === 'touch' || event.pointerType === 'pen'
    ? {id: event.pointerId, x: event.clientX, y: event.clientY, link, moved: !event.isPrimary}
    : null;
  return true;
};

WAC.trackAttachmentTouch = function (event) {
  const touch = WAC.attachmentTouch;
  if (!touch || touch.id !== event.pointerId) return;
  if (event.type === 'pointercancel' || Math.hypot(event.clientX - touch.x, event.clientY - touch.y) > 10) touch.moved = true;
  if (event.type === 'pointerup' && !touch.moved) {
    if (window.WanGPAppPlayer && WanGPAppPlayer.open(touch.link)) return;  // web apps and previews play in the app player
    if (WAC.openAttachment && WAC.openAttachment(touch.link)) return;
    const target = touch.link.target || '_blank';
    if (target === '_blank') window.open(touch.link.href, '_blank', 'noopener');
    else window.location.assign(touch.link.href);
  }
};

WAC.handleAttachmentClick = function (event) {
  const link = event.target.closest(WAC.attachmentSelector);
  if (!link) return false;
  const touch = WAC.attachmentTouch;
  WAC.attachmentTouch = null;
  if ((event.detail !== 0 && touch) || (window.WanGPAppPlayer && WanGPAppPlayer.open(link)) || (WAC.openAttachment && WAC.openAttachment(link))) {
    event.preventDefault();
    event.stopPropagation();
  }
  return true;
};




WAC.setBridgeValue = function (selector, value) {
  const input = document.querySelector(selector);
  if (!input) return false;
  input.value = String(value || '');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
};




WAC.setRequestInputValue = function (value) {
  const input = WAC.requestInput();
  if (!input) return;
  input.value = String(value || '');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
};

WAC.setComposerButtonIcon = function (button, action) {
  if (!button || (button.dataset.actionIcon === action && button.querySelector('svg'))) return;
  const icons = {
    Ask: ['Send request', '<path d="m4 4 17 8-17 8 3-8-3-8Z"/><path d="M7 12h14"/>'],
    Reset: ['Reset conversation', '<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>'],
    New: ['New conversation', '<path d="M20 11v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h8M16 5h6M19 2v6"/><path d="M7 10h5M7 14h9"/>'],
    Save: ['Save changes', '<path d="m5 12 4 4L19 6"/>'],
    Cancel: ['Cancel editing', '<path d="m6 6 12 12M6 18 18 6"/>'],
  };
  const [label, paths] = icons[action];
  button.innerHTML = `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  button.title = label; button.setAttribute('aria-label', label); button.dataset.actionIcon = action;
};

WAC.setQueuedEditButtonLabels = function (editing) {
  const askButton = document.querySelector('#assistant_chat_ask_button button, #assistant_chat_ask_button');
  const resetButton = document.querySelector('#assistant_chat_reset_button button, #assistant_chat_reset_button');
  WAC.setComposerButtonIcon(askButton, editing ? 'Save' : 'Ask');
  WAC.setComposerButtonIcon(resetButton, editing ? 'Cancel' : WAC.multiSessionEnabled ? 'New' : 'Reset');
};

WAC.finishQueuedRequestEdit = function () {
  const messageId = String(WAC.queuedEditMessageId || '');
  if (messageId) {
    const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
    if (messageNode) messageNode.classList.remove('is-editing');
  }
  WAC.setRequestInputValue(WAC.queuedEditDraft);
  WAC.queuedEditMessageId = '';
  WAC.queuedEditDraft = '';
  WAC.setQueuedEditButtonLabels(false);
};

WAC.startQueuedRequestEdit = function (messageNode) {
  if (!messageNode) return;
  const copyButton = messageNode.querySelector('[data-copy-source="user"]');
  const input = WAC.requestInput();
  if (!copyButton || !input || !messageNode.querySelector('[data-message-action="edit"]')) return;
  if (WAC.queuedEditMessageId) WAC.finishQueuedRequestEdit();
  WAC.queuedEditMessageId = String(messageNode.getAttribute('data-message-id') || '');
  WAC.queuedEditDraft = String(input.value || '');
  const text = String(copyButton.getAttribute('data-copy-text') || '');
  messageNode.classList.add('is-editing');
  WAC.setRequestInputValue(text);
  WAC.setQueuedEditButtonLabels(true);
  input.focus({ preventScroll: true });
  input.setSelectionRange(input.value.length, input.value.length);
};

WAC.syncQueuedRequestEdit = function () {
  if (WAC.replayDepth > 0) return;
  const messageId = String(WAC.queuedEditMessageId || '');
  if (!messageId) return;
  const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(messageId)}"]`);
  if (!messageNode || !messageNode.querySelector('[data-message-action="edit"]')) {
    WAC.finishQueuedRequestEdit();
    return;
  }
  messageNode.classList.add('is-editing');
  WAC.setQueuedEditButtonLabels(true);
};

WAC.submitQueuedRequestAction = function (messageNode, action, text) {
  if (!messageNode) return;
  const messageId = String(messageNode.getAttribute('data-message-id') || '');
  const normalizedAction = String(action || '').trim().toLowerCase();
  const content = String(text || '').trim();
  if (!messageId || !['edit', 'remove', 'steer'].includes(normalizedAction) || (normalizedAction === 'edit' && !content)) return;
  if (WAC.queuedEditMessageId === messageId) WAC.finishQueuedRequestEdit();
  messageNode.classList.add('is-pending-queue-action');
  if (!WAC.queuedRequestAction(normalizedAction, messageId, content)) {
    messageNode.classList.remove('is-pending-queue-action');
    return;
  }
  window.setTimeout(() => {
    if (messageNode.isConnected) messageNode.classList.remove('is-pending-queue-action');
  }, 3000);
};

WAC.handleQueuedRequestClick = function (event) {
  const messageAction = event && event.target && event.target.closest ? event.target.closest('[data-message-action]') : null;
  if (!messageAction) return false;
  const messageNode = messageAction.closest('.chat__message--user');
  if (!messageNode) return false;
  event.preventDefault();
  event.stopPropagation();
  const action = String(messageAction.getAttribute('data-message-action') || '');
  if (action === 'edit') WAC.startQueuedRequestEdit(messageNode);
  else if (action === 'remove') WAC.submitQueuedRequestAction(messageNode, 'remove', '');
  else if (action === 'steer') WAC.submitQueuedRequestAction(messageNode, 'steer', '');
  return true;
};

WAC.isAssistantBusy = function () {
  if (WAC.restorationStatus) return true;
  if (WAC.state && WAC.state.status && WAC.state.status.visible && WAC.state.status.text) return true;
  const stopButton = document.querySelector('#assistant_chat_html .chat__status-stop');
  return !!(stopButton && !stopButton.disabled);
};

WAC.setBusyInputHelper = function (visible) {
  const stats = WAC.statsNode();
  if (!stats) return;
  let helper = stats.querySelector('.chat__input-helper');
  if (!helper) {
    helper = document.createElement('span');
    helper.className = 'chat__input-helper';
    helper.textContent = 'Press Enter to Queue Requests / CTRL Enter to Steer Deepy';
    stats.prepend(helper);
  }
  helper.classList.toggle('is-visible', !!visible);
  helper.setAttribute('aria-hidden', visible ? 'false' : 'true');
  stats.classList.toggle('has-input-helper', !!visible);
  stats.setAttribute('aria-hidden', visible || stats.classList.contains('is-visible') ? 'false' : 'true');
};


WAC.consumePayload = function (payload) {
  if (!payload) return [];
  let envelope = payload;
  if (typeof payload === 'string') {
    if (payload === WAC.lastPayloadText) return [];
    try {
      envelope = JSON.parse(payload);
    } catch (_error) {
      return [];
    }
  }
  const payloadId = envelope && typeof envelope.event_id === 'string' ? envelope.event_id : '';
  const payloadText = typeof payload === 'string' ? payload : (payloadId ? '' : JSON.stringify(envelope));
  if ((payloadId && payloadId === WAC.lastPayloadId) || (!payloadId && payloadText === WAC.lastPayloadText)) return [];
  WAC.lastPayloadId = payloadId;
  WAC.lastPayloadText = payloadText;
  if (Array.isArray(envelope.batch)) {
    const replaying = !!envelope.replay;
    let transcript = null;
    let shell = null;
    let previousVisibility = '';
    if (replaying) {
      WAC.ensureShell();
      shell = typeof WAC.shell === 'function' ? WAC.shell() : null;
      transcript = WAC.transcript();
      previousVisibility = transcript ? transcript.style.visibility : '';
      if (transcript) transcript.style.visibility = 'hidden';
      if (shell) {
        shell.classList.add('is-replaying');
        shell.setAttribute('aria-busy', 'true');
      }
      WAC.replayDepth += 1;
    }
    try {
      for (const item of envelope.batch) WAC.consumePayload(item);
    } finally {
      if (replaying) {
        WAC.replayDepth = Math.max(0, WAC.replayDepth - 1);
        transcript = WAC.transcript() || transcript;
        WAC.applyDisclosureState(transcript);
        WAC.showEmptyIfNeeded();
        const scroll = WAC.scroll();
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
        WAC.syncJumpToBottom();
        if (transcript) transcript.style.visibility = previousVisibility;
        if (shell) {
          shell.classList.remove('is-replaying');
          shell.removeAttribute('aria-busy');
        }
      }
    }
    WAC.lastPayloadId = payloadId;
    WAC.lastPayloadText = payloadText;
    return [];
  }
  const instanceId = envelope && typeof envelope.instance_id === 'string' ? envelope.instance_id : '';
  if (instanceId) {
    if (WAC.serverInstanceId && WAC.serverInstanceId !== instanceId) {
      WAC.reset();
    }
    WAC.serverInstanceId = instanceId;
  }
  const event = envelope && envelope.event ? envelope.event : envelope;
  if (!event || typeof event !== 'object') return [];
  if (event.type === 'session_catalog') {
    WAC.sessionCatalog = Array.isArray(event.sessions) ? event.sessions : [];
    WAC.activeSessionId = String(event.active_session_id || '');
    WAC.multiSessionEnabled = !!event.multi_session_enabled;
    WAC.refreshSessionPickers();
    return [];
  }
  const chatSessionId = typeof event.chat_session_id === 'string' ? event.chat_session_id : '';
  if (chatSessionId && WAC.chatSessionId && chatSessionId !== WAC.chatSessionId) {
    // Saving the first request assigns a persistent session ID. Keep following it
    // when this event acknowledges that request, despite resetting the old session.
    const followedSubmission = WAC.syncAcknowledgesFollowedSubmission(event.messages || [event.message], event.acknowledged_submission_ids) ? WAC.followSubmissionId : '';
    WAC.reset();
    WAC.followSubmissionId = followedSubmission;
  }
  if (chatSessionId) WAC.chatSessionId = chatSessionId;
  if (event.type === 'session_resume_ready') {
    WAC.prefillResumedSession(event.request_id);
    return [];
  }
  const blockEventTypes = ['upsert_block', 'append_block_text', 'replace_block_text', 'finalize_block', 'remove_block'];
  const transcriptEvent = event.type === 'sync' || event.type === 'upsert_message' || event.type === 'remove_message' || event.type === 'pending_uploads' || blockEventTypes.includes(event.type);
  const revision = Number(event.revision);
  const hasRevision = transcriptEvent && Number.isFinite(revision);
  const sequence = Number(event.sequence);
  const sequenceStart = Number.isFinite(Number(event.sequence_start)) ? Number(event.sequence_start) : sequence;
  const hasSequence = transcriptEvent && Number.isFinite(sequence);
  const acknowledgedSubmissionIds = event.type === 'sync' ? (event.acknowledged_submission_ids || []) : [];
  const canonicalSubmissionIds = event.type === 'sync' ? (event.messages || []).map((message) => String(message && message.client_submission_id || '').trim()).filter(Boolean) : [];
  const steeringAcknowledged = !!WAC.pendingSteeringId && [...acknowledgedSubmissionIds, ...canonicalSubmissionIds].some((submissionId) => String(submissionId || '').trim() === WAC.pendingSteeringId);
  if (hasRevision && revision < WAC.chatRevision) {
    if (event.type === 'sync') {
      WAC.mergeStaleSync(event.messages || [], acknowledgedSubmissionIds);
      if (steeringAcknowledged) {
        WAC.pendingSteeringId = '';
        WAC.setStatus(event.status || null);
      }
    }
    return [];
  }
  if (event.type !== 'sync' && hasSequence) {
    if (sequence <= WAC.chatSequence) return [];
    if (WAC.chatSequence >= 0 && sequenceStart > WAC.chatSequence + 1) {
      // Recover missing earlier blocks without stalling a self-contained update.
      WAC.markSyncRequired(event);
      if (event.type !== 'upsert_block' && event.type !== 'finalize_block') return [];
    }
    WAC.chatSequence = sequence;
  }
  if (event.type === 'sync' && hasSequence) {
    WAC.chatSequence = sequence;
    WAC.syncRequired = false;
    WAC.syncRecoveryPending = false;
  }
  if (hasRevision) WAC.chatRevision = revision;
  if (event.pending_uploads !== undefined) WAC.setPendingUploads(event.pending_uploads);
  if (event.type === 'reset') {
    WAC.reset();
    if (chatSessionId) WAC.chatSessionId = chatSessionId;
    if (Number.isFinite(revision)) WAC.chatRevision = revision;
    if (Number.isFinite(sequence)) WAC.chatSequence = sequence;
    return [];
  }
  if (event.type === 'upsert_message') {
    const message = event.message || {};
    const submissionId = String(message.client_submission_id || '').trim();
    const followSubmittedRequest = !!submissionId && submissionId === WAC.followSubmissionId;
    if (submissionId) {
      WAC.acknowledgeOptimisticSubmits([submissionId]);
      if (submissionId === WAC.pendingSteeringId) {
        WAC.pendingSteeringId = '';
        WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering accepted. Waiting for the current thought/action boundary...' });
      }
    }
    WAC.upsertMessage(message, false, event.message_index);
    if (followSubmittedRequest) WAC.scrollToBottomAfterLayout();
    return [];
  }
  if (event.type === 'remove_message') {
    WAC.removeMessage(event.message_id);
    return [];
  }
  if (event.type === 'upsert_block') {
    WAC.upsertBlock(event);
    return [];
  }
  if (event.type === 'append_block_text') {
    WAC.appendBlockText(event);
    return [];
  }
  if (event.type === 'replace_block_text') {
    WAC.replaceBlockText(event);
    return [];
  }
  if (event.type === 'finalize_block') {
    WAC.finalizeBlock(event);
    return [];
  }
  if (event.type === 'remove_block') {
    WAC.removeBlock(event);
    return [];
  }
  if (event.type === 'status') {
    if (WAC.pendingSteeringId) return [];
    WAC.setStatus(event.status || null);
    if (Object.prototype.hasOwnProperty.call(event, 'stats')) WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'stats') {
    WAC.setStats(event.stats || null);
    return [];
  }
  if (event.type === 'sync') {
    const status = WAC.pendingSteeringId && !steeringAcknowledged ? WAC.state.status : (event.status || null);
    if (steeringAcknowledged) WAC.pendingSteeringId = '';
    WAC.sync(event.messages || [], status, Object.prototype.hasOwnProperty.call(event, 'stats') ? (event.stats || null) : WAC.state.stats, acknowledgedSubmissionIds);
    return [];
  }
  return [];
};




WAC.replaceState = function (messages, status, stats) {
  const nextState = { order: [], messages: {}, status: status || null, stats: typeof stats === 'undefined' ? (WAC.state ? (WAC.state.stats || null) : null) : (stats || null) };
  const items = Array.isArray(messages) ? messages : [];
  for (const message of items) {
    if (!message || !message.id) continue;
    const key = String(message.id);
    nextState.order.push(key);
    nextState.messages[key] = message;
  }
  WAC.state = nextState;
};

WAC.syncDockVisibility = function () {
  document.querySelectorAll('#assistant_chat_dock').forEach((dock) => {
    const hasLauncher = !!dock.querySelector('#assistant_chat_toggle');
    dock.style.display = hasLauncher ? 'flex' : 'none';
  });
};

WAC.parseThemeColor = function (value) {
  const match = String(value || '').trim().match(/^rgba?\(([^)]+)\)$/i);
  if (!match) return null;
  const parts = match[1].split(',').map((part) => parseFloat(part.trim()));
  if (parts.length < 3 || parts.slice(0, 3).some((part) => !Number.isFinite(part))) return null;
  const alpha = Number.isFinite(parts[3]) ? parts[3] : 1;
  if (alpha <= 0.01) return null;
  return { r: parts[0], g: parts[1], b: parts[2], a: alpha };
};

WAC.resolveThemeBackground = function (node) {
  let current = node;
  while (current) {
    const resolved = WAC.parseThemeColor(window.getComputedStyle(current).backgroundColor);
    if (resolved) return resolved;
    current = current.parentElement;
  }
  return WAC.parseThemeColor(window.getComputedStyle(document.body).backgroundColor);
};

WAC.relativeLuminance = function (rgb) {
  if (!rgb) return 1;
  const normalize = function (value) {
    const channel = Math.max(0, Math.min(255, Number(value || 0))) / 255;
    return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * normalize(rgb.r) + 0.7152 * normalize(rgb.g) + 0.0722 * normalize(rgb.b);
};

WAC.isDarkTheme = function () {
  const nodes = [
    document.querySelector('.gradio-container'),
    document.body,
    document.documentElement,
    document.querySelector('gradio-app'),
  ].filter(Boolean);
  if (nodes.some((node) => node.classList && node.classList.contains('dark'))) return true;
  if (nodes.some((node) => String(node.getAttribute('data-theme') || node.getAttribute('theme') || '').toLowerCase().includes('dark'))) return true;
  const sample = document.querySelector('.gradio-container') || document.body;
  const background = WAC.resolveThemeBackground(sample);
  const foreground = WAC.parseThemeColor(window.getComputedStyle(sample).color) || WAC.parseThemeColor(window.getComputedStyle(document.body).color);
  const backgroundLuminance = WAC.relativeLuminance(background);
  const foregroundLuminance = WAC.relativeLuminance(foreground);
  return backgroundLuminance < 0.18 || (foreground && backgroundLuminance < foregroundLuminance);
};

WAC.syncThemeState = function () {
  const dock = WAC.dock();
  if (!dock) return;
  dock.classList.toggle('is-dark', !!WAC.isDarkTheme());
};

WAC.syncDockState = function () {
  WAC.syncDockVisibility();
  WAC.syncThemeState();
  const dock = WAC.dock();
  const launcher = WAC.launcher();
  if (dock) dock.classList.toggle('is-open', !!WAC.dockOpen);
  if (launcher) launcher.setAttribute('aria-expanded', WAC.dockOpen ? 'true' : 'false');
  WAC.syncSettingsState();
};

WAC.syncSettingsState = function () {
  const panel = WAC.panel();
  const launcher = WAC.settingsLauncher();
  const open = !!WAC.dockOpen && !!WAC.settingsOpen;
  if (panel) panel.classList.toggle('is-settings-open', open);
  if (launcher) launcher.setAttribute('aria-expanded', open ? 'true' : 'false');
};

WAC.syncSettingsDropdowns = function () {
  const panel = WAC.settingsPanel();
  if (!panel) return;
  for (const menu of panel.querySelectorAll('.chat__settings-scroll .wrap > ul.options[role="listbox"]')) {
    const input = menu.parentElement.getBoundingClientRect();
    const scroll = menu.closest('.chat__settings-scroll').getBoundingClientRect();
    const above = Math.max(0, input.top - Math.max(0, scroll.top) - 4);
    const below = Math.max(0, Math.min(window.innerHeight, scroll.bottom) - input.bottom - 4);
    const openUp = below < Math.min(280, menu.scrollHeight) && above > below;
    menu.style.setProperty('--chat-options-top', openUp ? 'auto' : '100%');
    menu.style.setProperty('--chat-options-bottom', openUp ? '100%' : 'auto');
    menu.style.setProperty('--chat-options-height', `${Math.min(280, openUp ? above : below)}px`);
  }
};

WAC.syncDockLayout = function () {
  const dock = WAC.dock();
  if (!dock) return;
  if (window.innerWidth <= 900) {
    dock.style.removeProperty('--dock-panel-width');
    dock.style.removeProperty('--dock-settings-panel-width');
    return;
  }
  const candidates = [
    dock.parentElement,
    dock.parentElement ? dock.parentElement.closest('.column') : null,
    dock.parentElement && dock.parentElement.parentElement ? dock.parentElement.parentElement.closest('.column') : null,
  ].filter((node) => node && node !== dock);
  const flowColumn = candidates
    .map((node) => ({ node, rect: node.getBoundingClientRect() }))
    .filter((entry) => entry.rect.width > 180)
    .sort((a, b) => a.rect.width - b.rect.width)[0];
  const flowRect = flowColumn ? flowColumn.rect : null;
  const dockStyle = window.getComputedStyle(dock);
  const launcherWidth = parseFloat(dockStyle.getPropertyValue('--dock-launcher-width')) || 41;
  const dockGap = parseFloat(dockStyle.getPropertyValue('--dock-gap')) || 14;
  const panelLeft = launcherWidth + dockGap;
  const measuredWidth = flowRect ? Math.round(flowRect.width) : 0;
  const columnBoundWidth = flowRect ? Math.round(flowRect.right - panelLeft - 12) : 0;
  const maxWidth = Math.max(320, window.innerWidth - panelLeft - 28);
  const panelWidth = Math.max(Math.min(320, maxWidth), Math.min(measuredWidth || 548, columnBoundWidth || measuredWidth || 548, maxWidth));
  const settingsPanelOffset = parseFloat(dockStyle.getPropertyValue('--dock-settings-panel-offset')) || 44;
  const maxSettingsWidth = Math.max(320, window.innerWidth - panelLeft - panelWidth - settingsPanelOffset - 12);
  const settingsWidth = Math.min(maxSettingsWidth, Math.max(320, Math.min(panelWidth + 112, 660)));
  dock.style.setProperty('--dock-panel-width', `${panelWidth}px`);
  dock.style.setProperty('--dock-settings-panel-width', `${settingsWidth}px`);
};

WAC.setDockOpen = function (open, focusInput = true) {
  const firstOpen = open && !WAC.hasOpenedDock;
  if (open) WAC.hasOpenedDock = true;
  WAC.dockOpen = !!open;
  WAC.syncDockState();
  WAC.syncDockLayout();
  if (WAC.dockOpen) {
    WAC.syncComposerLayout();
    if (firstOpen && WAC.state.order.length > 0) WAC.scrollToBottomAfterLayout();
    window.setTimeout(() => {
      const input = WAC.requestInput();
      if (input && focusInput) input.focus();
    }, 140);
  }
};

WAC.toggleDock = function (forceOpen) {
  const nextOpen = typeof forceOpen === 'boolean' ? forceOpen : !WAC.dockOpen;
  WAC.setDockOpen(nextOpen);
};

WAC.setSettingsOpen = function (open) {
  WAC.settingsOpen = !!open;
  if (WAC.settingsOpen && !WAC.dockOpen) WAC.dockOpen = true;
  WAC.syncDockState();
  WAC.syncDockLayout();
  if (WAC.settingsOpen) {
    const refreshButton = document.querySelector('#assistant_chat_session_refresh_button button, #assistant_chat_session_refresh_button');
    if (refreshButton && typeof refreshButton.click === 'function') refreshButton.click();
  }
};

WAC.toggleSettings = function (forceOpen) {
  const nextOpen = typeof forceOpen === 'boolean' ? forceOpen : !WAC.settingsOpen;
  WAC.setSettingsOpen(nextOpen);
};

WAC.ensureShell = function () {
  const host = WAC.host();
  if (!host) return false;
  WAC.readSessionCatalogFromHost();
  WAC.syncSessionActionTooltips();
  WAC.bindSessionPickerControls(host);
  if (host.dataset.wangpAssistantChatMounted === 'true' && WAC.shell()) {
    if (WAC.replayDepth <= 0) {
      WAC.showEmptyIfNeeded();
      WAC.syncDockState();
      WAC.syncDockLayout();
      WAC.syncComposerLayout();
    }
    return true;
  }
  host.innerHTML = `
    <section class="chat">
      <div class="chat__scroll">
        <div class="chat__empty">
          ${WAC.emptyMarkup('zero')}
        </div>
        <div class="chat__transcript"></div>
      </div>
      <div class="chat__status" aria-live="polite">
        <div class="chat__status-dots" aria-hidden="true"><span></span><span></span><span></span></div>
        <div class="chat__status-main"><div class="chat__status-text"></div><div class="chat__status-detail" hidden></div><div class="chat__status-progress" hidden><span></span></div></div>
        <button class="chat__status-pause" type="button" aria-label="Pause Deepy" disabled>Pause</button>
        <button class="chat__status-stop" type="button" aria-label="Stop Deepy" disabled>Stop</button>
      </div>
      <button class="chat__jump-bottom" type="button" aria-label="Jump to latest messages" aria-hidden="true" tabindex="-1">
        <span aria-hidden="true"></span>
      </button>
    </section>
  `;
  host.dataset.wangpAssistantChatMounted = 'true';
  WAC.bindSessionPickerControls(host);
  WAC.hydrate();
  WAC.syncDockVisibility();
  WAC.syncDockState();
  WAC.syncDockLayout();
  WAC.syncComposerLayout();
  WAC.syncDisclosureBridge();
  WAC.syncScrollBridge();
  return true;
};

WAC.isNearBottom = function () {
  const scroll = WAC.scroll();
  if (!scroll) return true;
  return (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight) <= WAC.bottomThreshold();
};

WAC.syncJumpToBottom = function () {
  if (WAC.replayDepth > 0) return;
  const node = WAC.jumpBottomNode();
  if (!node) return;
  const show = WAC.state.order.length > 0 && !WAC.isNearBottom();
  node.classList.toggle('is-visible', show);
  node.setAttribute('aria-hidden', show ? 'false' : 'true');
  node.tabIndex = show ? 0 : -1;
};

WAC.scrollToBottom = function () {
  const scroll = WAC.scroll();
  if (!scroll) return;
  scroll.scrollTop = scroll.scrollHeight;
  WAC.syncJumpToBottom();
};

WAC.scrollToBottomAfterLayout = function () {
  WAC.scrollToBottom();
  if (WAC.followSubmissionScrollFrame) window.cancelAnimationFrame(WAC.followSubmissionScrollFrame);
  WAC.followSubmissionScrollFrame = window.requestAnimationFrame(() => {
    WAC.followSubmissionScrollFrame = window.requestAnimationFrame(() => {
      WAC.followSubmissionScrollFrame = 0;
      WAC.scrollToBottom();
    });
  });
};

WAC.hideEmpty = function () {
  if (WAC.replayDepth > 0) return;
  const empty = WAC.empty();
  const transcript = WAC.transcript();
  if (empty) empty.style.display = 'none';
  // Show the transcript in the same frame, so the first message is laid out before scrolling to it.
  if (transcript) transcript.style.display = 'flex';
};

WAC.showEmptyIfNeeded = function () {
  if (WAC.replayDepth > 0) return;
  const empty = WAC.empty();
  const transcript = WAC.transcript();
  const isEmpty = WAC.state.order.length === 0;
  if (empty) empty.style.display = isEmpty ? 'flex' : 'none';
  if (transcript) transcript.style.display = isEmpty ? 'none' : 'flex';
  WAC.syncJumpToBottom();
};

WAC.createMessageNode = function (message) {
  const tpl = document.createElement('template');
  tpl.innerHTML = (message && message.html) ? String(message.html).trim() : '';
  return tpl.content.firstElementChild;
};

WAC.markSyncRequired = function (event) {
  WAC.syncRequired = true;
  window.dispatchEvent(new CustomEvent('wangp-assistant-chat-sync-required', { detail: { chatSessionId: WAC.chatSessionId, sequence: WAC.chatSequence, event: event || null } }));
  if (!WAC.syncRecoveryPending) {
    WAC.syncRecoveryPending = true;
    if (!WAC.requestCanonicalSync()) WAC.syncRecoveryPending = false;
  }
};

WAC.syncAcknowledgesFollowedSubmission = function (messages, acknowledgedSubmissionIds) {
  const followed = String(WAC.followSubmissionId || '').trim();
  if (!followed) return false;
  if ((Array.isArray(acknowledgedSubmissionIds) ? acknowledgedSubmissionIds : []).some((value) => String(value || '').trim() === followed)) return true;
  return (Array.isArray(messages) ? messages : []).some((message) => String(message && message.client_submission_id || '').trim() === followed);
};

WAC.messageNode = function (messageId) {
  const transcript = WAC.transcript();
  return transcript ? transcript.querySelector(`[data-message-id="${CSS.escape(String(messageId || ''))}"]`) : null;
};

WAC.blockNode = function (messageId, blockId) {
  const messageNode = WAC.messageNode(messageId);
  return messageNode ? messageNode.querySelector(`[data-block-id="${CSS.escape(String(blockId || ''))}"]`) : null;
};

WAC.liveTextNode = function (blockNode) {
  return blockNode && blockNode.querySelector ? blockNode.querySelector('.chat__stream-text') : null;
};

WAC.safeStreamingMarkdownUrl = function (value, image) {
  const raw = String(value || '').trim().replace(/^<|>$/g, '');
  if (!raw || /^javascript:/i.test(raw) || /^data:/i.test(raw)) return '';
  if (raw.startsWith('/wangp_api/gallery/media/') || (!image && raw.startsWith('/wangp_api/download/'))) return raw;
  try {
    const resolved = new URL(raw, document.baseURI);
    return resolved.protocol === 'http:' || resolved.protocol === 'https:' ? resolved.href : '';
  } catch (_error) {
    return '';
  }
};

WAC.streamingMarkdownDelimiterFlags = function (text, index, width, marker) {
  const before = index > 0 ? text[index - 1] : '';
  const after = index + width < text.length ? text[index + width] : '';
  const beforeWhitespace = !before || /\s/u.test(before);
  const afterWhitespace = !after || /\s/u.test(after);
  const beforePunctuation = !!before && /[\p{P}\p{S}]/u.test(before);
  const afterPunctuation = !!after && /[\p{P}\p{S}]/u.test(after);
  const leftFlanking = !afterWhitespace && (!afterPunctuation || beforeWhitespace || beforePunctuation);
  const rightFlanking = !beforeWhitespace && (!beforePunctuation || afterWhitespace || afterPunctuation);
  if (marker === '_') {
    return { open: leftFlanking && (!rightFlanking || beforePunctuation), close: rightFlanking && (!leftFlanking || afterPunctuation) };
  }
  return { open: leftFlanking, close: rightFlanking };
};

WAC.findStreamingMarkdownCloser = function (text, marker, start) {
  const width = marker.length;
  const markerChar = marker[0];
  let index = text.indexOf(marker, start);
  while (index >= 0) {
    let backslashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) backslashes += 1;
    const exactRun = text[index - 1] !== markerChar && text[index + width] !== markerChar;
    if (backslashes % 2 === 0 && exactRun && WAC.streamingMarkdownDelimiterFlags(text, index, width, markerChar).close) return index;
    index = text.indexOf(marker, index + width);
  }
  return -1;
};

WAC.appendStreamingInlineMarkdown = function (parent, value) {
  const text = String(value || '');
  let plainStart = 0;
  let index = 0;
  const flushPlain = (end) => {
    if (end > plainStart) parent.appendChild(document.createTextNode(text.slice(plainStart, end)));
  };
  while (index < text.length) {
    let consumed = 0;
    let rendered = null;
    if (text[index] === '\\' && index + 1 < text.length) {
      flushPlain(index);
      parent.appendChild(document.createTextNode(text[index + 1]));
      index += 2;
      plainStart = index;
      continue;
    }
    if ((text.startsWith('![', index) || text[index] === '[')) {
      const image = text.startsWith('![', index);
      const labelStart = index + (image ? 2 : 1);
      const labelEnd = text.indexOf('](', labelStart);
      const targetEnd = labelEnd < 0 ? -1 : text.indexOf(')', labelEnd + 2);
      if (labelEnd >= 0 && targetEnd >= 0) {
        const label = text.slice(labelStart, labelEnd);
        const rawTarget = text.slice(labelEnd + 2, targetEnd).trim().split(/\s+/)[0];
        const href = WAC.safeStreamingMarkdownUrl(rawTarget, image);
        if (href) {
          if (image) {
            rendered = document.createElement('img');
            rendered.src = href;
            rendered.alt = label;
            rendered.loading = 'lazy';
          } else {
            rendered = document.createElement('a');
            rendered.href = href;
            rendered.target = '_blank';
            rendered.rel = 'noopener noreferrer';
            WAC.appendStreamingInlineMarkdown(rendered, label);
          }
          consumed = targetEnd - index + 1;
        }
      }
    }
    if (!rendered && text[index] === '`') {
      const end = text.indexOf('`', index + 1);
      if (end > index + 1) {
        rendered = document.createElement('code');
        rendered.textContent = text.slice(index + 1, end);
        consumed = end - index + 1;
      }
    }
    if (!rendered && (text.startsWith('**', index) || text.startsWith('__', index)) && text[index - 1] !== text[index] && text[index + 2] !== text[index] && WAC.streamingMarkdownDelimiterFlags(text, index, 2, text[index]).open) {
      const marker = text.slice(index, index + 2);
      const end = WAC.findStreamingMarkdownCloser(text, marker, index + 2);
      if (end > index + 2) {
        rendered = document.createElement('strong');
        WAC.appendStreamingInlineMarkdown(rendered, text.slice(index + 2, end));
        consumed = end - index + 2;
      }
    }
    if (!rendered && (text[index] === '*' || text[index] === '_') && text[index - 1] !== text[index] && text[index + 1] !== text[index] && WAC.streamingMarkdownDelimiterFlags(text, index, 1, text[index]).open) {
      const end = WAC.findStreamingMarkdownCloser(text, text[index], index + 1);
      if (end > index + 1) {
        rendered = document.createElement('em');
        WAC.appendStreamingInlineMarkdown(rendered, text.slice(index + 1, end));
        consumed = end - index + 1;
      }
    }
    if (!rendered) {
      index += 1;
      continue;
    }
    flushPlain(index);
    parent.appendChild(rendered);
    index += consumed;
    plainStart = index;
  }
  flushPlain(text.length);
};

WAC.resetStreamingMarkdown = function (node) {
  node.replaceChildren();
  delete node.__wangpStreamingArrival;
  const state = { source: '', buffer: '', inFence: false, fence: '', code: null, list: null, listType: '', table: null, tableHeader: null, blockBoundary: true, tail: null };
  node.__wangpStreamingMarkdown = state;
  return state;
};

WAC.appendStreamingListItem = function (node, state, line) {
  const match = line.match(/^[ \t]*(?:([-+*])|(\d+)\.)[ \t]+(.*)$/);
  if (!match) return null;
  const listType = match[1] ? 'ul' : 'ol';
  const continuing = state.listType === listType && state.list && state.list === node.lastElementChild;
  if (!state.blockBoundary && !continuing) return null;
  const list = continuing ? state.list : document.createElement(listType);
  if (!continuing) {
    if (listType === 'ol' && Number(match[2]) !== 1) list.start = Number(match[2]);
    node.appendChild(list);
  }
  const item = document.createElement('li');
  WAC.appendStreamingInlineMarkdown(item, match[3]);
  list.appendChild(item);
  state.list = list;
  state.listType = listType;
  state.blockBoundary = false;
  return item;
};

WAC.splitStreamingTableRow = function (line) {
  const text = line.trim();
  const cells = [];
  let cell = '';
  let code = '';
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '\\' && index + 1 < text.length) {
      cell += char + text[++index];
    } else if (char === '`') {
      const marker = text.slice(index).match(/^\x60+/)[0];
      if (code === marker) code = '';
      else if (!code && text.indexOf(marker, index + marker.length) >= 0) code = marker;
      cell += marker;
      index += marker.length - 1;
    } else if (char === '|' && !code) {
      cells.push(cell.trim());
      cell = '';
    } else {
      cell += char;
    }
  }
  if (!cells.length) return null;
  cells.push(cell.trim());
  if (text.startsWith('|')) cells.shift();
  if (cells[cells.length - 1] === '' && text.endsWith('|')) cells.pop();
  return cells;
};

WAC.appendStreamingTableRow = function (parent, cells, alignments, header) {
  const row = document.createElement('tr');
  alignments.forEach((alignment, index) => {
    const cell = document.createElement(header ? 'th' : 'td');
    if (alignment) cell.style.textAlign = alignment;
    WAC.appendStreamingInlineMarkdown(cell, cells[index] || '');
    row.appendChild(cell);
  });
  parent.appendChild(row);
  return row;
};

WAC.renderStreamingMarkdownLine = function (node, state, line) {
  const header = state.tableHeader;
  state.tableHeader = null;
  const fence = line.match(/^ {0,3}(\x60{3,}|~{3,})(.*)$/);
  if (state.inFence) {
    if (fence && fence[1][0] === state.fence[0] && fence[1].length >= state.fence.length && !fence[2].trim()) {
      state.inFence = false;
      state.fence = '';
      state.code = null;
      state.blockBoundary = true;
    } else {
      state.code.appendChild(document.createTextNode(`${line}\n`));
    }
    return;
  }
  if (fence) {
    state.table = null;
    state.list = null;
    state.listType = '';
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    if (fence[2].trim()) code.className = `language-${fence[2].trim().replace(/[^a-z0-9_-]/gi, '')}`;
    pre.appendChild(code);
    node.appendChild(pre);
    state.inFence = true;
    state.fence = fence[1];
    state.code = code;
    state.blockBoundary = true;
    return;
  }
  const cells = WAC.splitStreamingTableRow(line);
  if (state.table && line.trim()) {
    WAC.appendStreamingTableRow(state.table.body, cells || [line.trim()], state.table.alignments, false);
    return;
  }
  state.table = null;
  const separators = cells || [line.trim()];
  if (header && separators.length === header.cells.length && separators.every((cell) => /^:?-+:?$/.test(cell))) {
    const alignments = separators.map((cell) => cell.startsWith(':') ? (cell.endsWith(':') ? 'center' : 'left') : (cell.endsWith(':') ? 'right' : ''));
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const body = document.createElement('tbody');
    WAC.appendStreamingTableRow(head, header.cells, alignments, true);
    table.appendChild(head);
    table.appendChild(body);
    header.span.remove();
    header.break.remove();
    node.appendChild(table);
    state.table = { body, alignments };
    state.blockBoundary = false;
    return;
  }
  const heading = line.match(/^(#{1,6})\s+(.+)$/);
  if (heading) {
    state.list = null;
    state.listType = '';
    const element = document.createElement(`h${heading[1].length}`);
    WAC.appendStreamingInlineMarkdown(element, heading[2]);
    node.appendChild(element);
    state.blockBoundary = true;
    return;
  }
  const quote = line.match(/^\s*>\s?(.*)$/);
  if (quote) {
    state.list = null;
    state.listType = '';
    const element = document.createElement('blockquote');
    WAC.appendStreamingInlineMarkdown(element, quote[1]);
    node.appendChild(element);
    state.blockBoundary = true;
    return;
  }
  if (WAC.appendStreamingListItem(node, state, line)) return;
  if (!line.trim()) {
    // A paragraph break measures what the final rendering gives it (a paragraph margin), and nothing after a block with its own margin (code, heading, list), so the text does not shift when the block is finalized.
    if (!state.list && node.lastElementChild && node.lastElementChild.tagName === 'BR') {
      const gap = document.createElement('span');
      gap.className = 'chat__stream-gap';
      node.appendChild(gap);
    }
    state.blockBoundary = true;
    return;
  }
  state.list = null;
  state.listType = '';
  const rule = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  if (rule) {
    node.appendChild(document.createElement('hr'));
    state.blockBoundary = true;
    return;
  }
  const span = document.createElement('span');
  WAC.appendStreamingInlineMarkdown(span, line);
  node.appendChild(span);
  const lineBreak = document.createElement('br');
  node.appendChild(lineBreak);
  if (cells && cells.length && !/^(?: {4}|\t)/.test(line)) state.tableHeader = { cells, span, break: lineBreak };
  state.blockBoundary = false;
};

WAC.renderStreamingMarkdown = function (node, value) {
  if (!node) return;
  const text = String(value || '');
  let state = node.__wangpStreamingMarkdown;
  if (state && text === state.source) return;
  if (!state || !text.startsWith(state.source)) state = WAC.resetStreamingMarkdown(node);
  const delta = text.slice(state.source.length);
  WAC.appendStreamingMarkdown(node, delta);
};

// The reveal loop supplies only new text; completed Markdown stays untouched.
WAC.appendStreamingMarkdown = function (node, delta) {
  const state = node.__wangpStreamingMarkdown || WAC.resetStreamingMarkdown(node);
  if (state.tablePending) {
    const pending = state.tablePending;
    state.table.body.parentNode.remove();
    node.appendChild(pending.tableHeader.span);
    node.appendChild(pending.tableHeader.break);
    Object.assign(state, pending);
  }
  const previousBuffer = state.buffer;
  state.buffer += delta;
  state.source += delta;
  let completedLine = false;
  let newline = state.buffer.indexOf('\n');
  while (newline >= 0) {
    completedLine = true;
    if (state.tail) state.tail.remove();
    state.tail = null;
    WAC.renderStreamingMarkdownLine(node, state, state.buffer.slice(0, newline));
    state.buffer = state.buffer.slice(newline + 1);
    newline = state.buffer.indexOf('\n');
  }
  if (!state.inFence && state.tableHeader) {
    const separators = WAC.splitStreamingTableRow(state.buffer) || [state.buffer.trim()];
    if (separators.length === state.tableHeader.cells.length && separators.every((cell) => /^:?-{3,}:?$/.test(cell))) {
      if (state.tail) state.tail.remove();
      state.tail = null;
      const pending = { ...state, tablePending: null };
      WAC.renderStreamingMarkdownLine(node, state, state.buffer);
      state.tablePending = pending;
      return;
    }
  }
  const delimiterArrived = ['\\', '`', '*', '_', '[', ']', '(', ')', '!'].some((marker) => delta.includes(marker)) || previousBuffer.endsWith('\\');
  const listMarker = /^[ \t]*(?:[-+*]|\d+\.)[ \t]+/;
  const listMarkerArrived = listMarker.test(state.buffer) && !listMarker.test(previousBuffer);
  if (state.tail && !state.table && !completedLine && !delimiterArrived && !listMarkerArrived && state.buffer === previousBuffer + delta) {
    const target = ['OL', 'UL'].includes(state.tail.tagName) ? state.tail.lastElementChild : state.tail;
    if (target.lastChild && target.lastChild.nodeType === 3) target.lastChild.appendData(delta);
    else target.appendChild(document.createTextNode(delta));
    return;
  }
  if (state.tail) state.tail.remove();
  if (state.table && state.buffer.trim()) {
    state.tail = WAC.appendStreamingTableRow(state.table.body, WAC.splitStreamingTableRow(state.buffer) || [state.buffer.trim()], state.table.alignments, false);
    return;
  }
  if (!state.inFence) {
    const preview = { ...state };
    const item = WAC.appendStreamingListItem(node, preview, state.buffer);
    if (item) {
      state.tail = preview.list === state.list ? item : preview.list;
      return;
    }
  }
  state.tail = document.createElement('span');
  state.tail.className = 'chat__stream-tail';
  if (state.inFence && state.code) {
    state.tail.textContent = state.buffer;
    state.code.appendChild(state.tail);
  } else {
    WAC.appendStreamingInlineMarkdown(state.tail, state.buffer);
    node.appendChild(state.tail);
  }
};

WAC.shouldRevealStreamingText = function (live) {
  return !WAC.replayDepth && !document.hidden && !window.matchMedia('(prefers-reduced-motion: reduce)').matches && !live.closest('details:not([open])') && live.getClientRects().length > 0;
};

WAC.clearStreamingReveals = function (root, flush = false) {
  for (const [live, reveal] of WAC.streamingReveals) {
    if (root && root !== live && !root.contains(live)) continue;
    WAC.streamingReveals.delete(live);
    if (flush && live.isConnected) {
      WAC.appendStreamingMarkdown(live, reveal.text.slice(live.__wangpStreamingMarkdown.source.length));
      if (reveal.finalEvent) WAC.finalizeBlock(reveal.finalEvent);
    }
  }
  if (!WAC.streamingReveals.size && WAC.streamingRevealFrame) {
    window.cancelAnimationFrame(WAC.streamingRevealFrame);
    WAC.streamingRevealFrame = 0;
  }
};

WAC.queueStreamingReveal = function (live, text) {
  const now = performance.now();
  const start = live.__wangpStreamingMarkdown.source.length;
  if (text.length <= start) return;
  const reveal = WAC.streamingReveals.get(live) || { last: now, target: start };
  reveal.text = text;
  // Learn visible throughput per block, independently of the model, action and tokenizer.
  // Coalesced events within 50 ms form one sample; empty/final events are not arrivals.
  let arrival = live.__wangpStreamingArrival;
  if (!arrival) arrival = live.__wangpStreamingArrival = { at: now, length: text.length, interval: 250, rate: (text.length - start) * 2, measured: false };
  const elapsed = now - arrival.at;
  if (!reveal.finalEvent && text.length > arrival.length && elapsed >= 50) {
    const rate = (text.length - arrival.length) * 1000 / elapsed;
    const weight = Math.max(rate < arrival.rate ? 0.5 : 0, 1 - Math.exp(-elapsed / 1000));
    arrival.rate = arrival.measured ? arrival.rate + weight * (rate - arrival.rate) : rate;
    arrival.interval = Math.max(elapsed, 0.75 * arrival.interval + 0.25 * elapsed);
    arrival.at = now;
    arrival.length = text.length;
    arrival.measured = true;
  }
  // Completion no longer needs a jitter buffer, but must not dump the remaining text.
  if (reveal.finalEvent) reveal.finishAt ??= now + Math.min(700, (text.length - reveal.target) * 1000 / arrival.rate);
  else delete reveal.finishAt;
  WAC.streamingReveals.set(live, reveal);
  if (!WAC.shouldRevealStreamingText(live)) WAC.clearStreamingReveals(live, true);
  else if (!WAC.streamingRevealFrame) WAC.streamingRevealFrame = window.requestAnimationFrame(WAC.stepStreamingReveals);
};

WAC.stepStreamingReveals = function (now) {
  WAC.streamingRevealFrame = 0;
  // Read visibility before changing the DOM, then scroll once for the whole frame.
  const ready = [];
  for (const [live, reveal] of WAC.streamingReveals) {
    if (!live.isConnected) { WAC.streamingReveals.delete(live); continue; }
    const animate = WAC.shouldRevealStreamingText(live);
    if (!animate || now - reveal.last >= 32) ready.push({ live, reveal, animate });
  }
  if (ready.length) {
    const scrollState = WAC.captureAutoscrollState();
    for (const { live, reveal, animate } of ready) {
      const start = live.__wangpStreamingMarkdown.source.length;
      const remaining = reveal.text.length - reveal.target;
      const arrival = live.__wangpStreamingArrival;
      const bufferSeconds = Math.max(250, arrival.interval) / 1000;
      // Keep about one arrival interval in reserve. Correct excess/deficit gradually instead
      // of consuming the entire backlog on every chunk and pausing until the next one.
      const speed = reveal.finishAt
        ? remaining * 1000 / Math.max(1, reveal.finishAt - reveal.last)
        : arrival.rate + (remaining - arrival.rate * bufferSeconds) / (2 * bufferSeconds);
      reveal.target = Math.min(reveal.text.length, reveal.target + speed * (now - reveal.last) / 1000);
      let end = animate ? Math.floor(reveal.target) : reveal.text.length;
      // Stay on the pacing clock; rounding forward to a whole word can consume a slow
      // model's entire next chunk. Only keep UTF-16 surrogate pairs together.
      if (end < reveal.text.length && /[\uDC00-\uDFFF]/u.test(reveal.text[end])) end -= 1;
      if (end <= start) { reveal.last = now; continue; }
      WAC.appendStreamingMarkdown(live, reveal.text.slice(start, end));
      reveal.last = now;
      if (end === reveal.text.length) {
        WAC.streamingReveals.delete(live);
        if (reveal.finalEvent) {
          // Finalization may regroup compact blocks; preserve the position from before this text grew.
          WAC.applyAutoscrollState(scrollState);
          WAC.finalizeBlock(reveal.finalEvent);
        }
      }
    }
    WAC.applyAutoscrollState(scrollState);
  }
  if (WAC.streamingReveals.size) WAC.streamingRevealFrame = window.requestAnimationFrame(WAC.stepStreamingReveals);
};

WAC.incrementalMessageState = function (messageId) {
  const key = String(messageId || '');
  if (!WAC.blockState[key]) WAC.blockState[key] = { order: [], blocks: {} };
  return WAC.blockState[key];
};

WAC.createBlockNode = function (html) {
  const tpl = document.createElement('template');
  tpl.innerHTML = String(html || '').trim();
  return tpl.content.firstElementChild;
};

WAC.positionBlockNode = function (messageNode, blockNode, blockIndex) {
  const body = messageNode && messageNode.querySelector ? messageNode.querySelector('.chat__body') : null;
  if (!body || !blockNode) return;
  const blocks = (WAC.compactChildren?.(body) || Array.from(body.children)).filter(node => node.dataset.blockId);
  const currentIndex = blocks.indexOf(blockNode);
  const index = Number.isFinite(Number(blockIndex)) ? Number(blockIndex) : currentIndex < 0 ? Math.max(-1, ...blocks.map((node, ordinal) => Number(node.dataset.blockIndex ?? ordinal))) + 1 : Number(blockNode.dataset.blockIndex ?? currentIndex);
  blockNode.dataset.blockIndex = String(index);
  const before = blocks.find((node, ordinal) => node !== blockNode && Number(node.dataset.blockIndex ?? ordinal) > index);
  if (before) {
    if (currentIndex < 0 || blocks[currentIndex + 1] !== before) (before._compactSlot || before).before(blockNode);
  }
  else if (!blocks.length) body.appendChild(blockNode);
  else if (blockNode !== blocks[blocks.length - 1]) {
    const last = blocks[blocks.length - 1];
    (last._compactSlot || last).after(blockNode);
  }
};

WAC.rememberBlock = function (event, node, text, finalized) {
  const messageState = WAC.incrementalMessageState(event.message_id);
  const blockId = String(event.block_id || '');
  if (!messageState.blocks[blockId]) messageState.order.push(blockId);
  messageState.blocks[blockId] = {
    id: blockId,
    type: String(event.block_type || (node && node.dataset.blockType) || ''),
    index: Number.isFinite(Number(event.block_index)) ? Number(event.block_index) : messageState.order.length - 1,
    html: node ? (WAC.canonicalBlockHTML?.(node) ?? node.outerHTML) : String(event.html || ''),
    text: String(typeof text === 'undefined' ? '' : text),
    finalized: !!finalized,
  };
  messageState.order.sort((left, right) => Number(messageState.blocks[left].index) - Number(messageState.blocks[right].index));
};

WAC.upsertBlock = function (event) {
  if (!event || !event.message_id || !event.block_id) return;
  if (!WAC.messageNode(event.message_id) && event.message) WAC.upsertMessage(event.message, true, event.message_index);
  const messageNode = WAC.messageNode(event.message_id);
  if (!messageNode) return WAC.markSyncRequired(event);
  WAC.captureDisclosureState(messageNode);
  const scrollState = WAC.captureAutoscrollState();
  const next = WAC.createBlockNode(event.html);
  if (!next) return WAC.markSyncRequired(event);
  const initialText = String(event.text || '');
  const nextLive = WAC.liveTextNode(next);
  if (nextLive) WAC.resetStreamingMarkdown(nextLive);
  const existing = WAC.blockNode(event.message_id, event.block_id);
  let node = next;
  if (existing) {
    const key = WAC.disclosureKey(existing.querySelector && existing.matches('.chat__disclosure') ? existing : existing.querySelector('.chat__disclosure'));
    if (key) WAC.disclosureState[key] = !!(existing.matches('.chat__disclosure') ? existing.open : existing.querySelector('.chat__disclosure').open);
    node = WAC.patchBlockNode(existing, next);
  }
  WAC.positionBlockNode(messageNode, node, event.block_index);
  WAC.rememberBlock(event, node, initialText, !event.streaming);
  WAC.hideEmpty();
  WAC.applyDisclosureState(messageNode);
  const live = WAC.liveTextNode(node);
  if (live) {
    WAC.resetStreamingMarkdown(live);
    if (existing) WAC.renderStreamingMarkdown(live, initialText);
    else WAC.queueStreamingReveal(live, initialText);
  }
  window.WanGPAppPlayer?.sync(node);  // Deepy changed the app being played: reload it
  WAC.applyAutoscrollState(scrollState);
};

WAC.currentBlockText = function (event, node) {
  const messageState = WAC.incrementalMessageState(event.message_id);
  const known = messageState.blocks[String(event.block_id || '')];
  if (known) return String(known.text || '');
  const live = WAC.liveTextNode(node);
  return live ? String(live.textContent || '') : '';
};

WAC.appendBlockText = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  const live = WAC.liveTextNode(node);
  if (!node || !live) return WAC.markSyncRequired(event);
  const current = WAC.currentBlockText(event, node);
  const start = Number(event.text_start);
  const end = Number(event.text_end);
  // Python publishes code-point offsets; JS length counts UTF-16 code units.
  let currentLength = 0;
  for (const character of current) currentLength += 1;
  if (Number.isFinite(end) && currentLength >= end) return;
  if (!Number.isFinite(start) || currentLength !== start) return WAC.markSyncRequired(event);
  const scrollState = WAC.captureAutoscrollState();
  const suffix = String(event.text || '');
  const known = WAC.incrementalMessageState(event.message_id).blocks[String(event.block_id)];
  if (known) { known.text = current + suffix; known.finalized = false; }
  else WAC.rememberBlock(event, node, current + suffix, false);
  if (!live.__wangpStreamingMarkdown) WAC.renderStreamingMarkdown(live, current);
  const reveal = WAC.streamingReveals.get(live);
  if (reveal) reveal.finalEvent = null;
  WAC.queueStreamingReveal(live, current + suffix);
  WAC.applyAutoscrollState(scrollState);
};

WAC.replaceBlockText = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  const live = WAC.liveTextNode(node);
  if (!node || !live) return WAC.markSyncRequired(event);
  const scrollState = WAC.captureAutoscrollState();
  WAC.clearStreamingReveals(live);
  WAC.renderStreamingMarkdown(live, String(event.text || ''));
  WAC.rememberBlock(event, node, String(event.text || ''), false);
  WAC.applyAutoscrollState(scrollState);
};

WAC.finalizeBlock = function (event) {
  const live = WAC.liveTextNode(WAC.blockNode(event.message_id, event.block_id));
  const reveal = WAC.streamingReveals.get(live);
  if (reveal && WAC.shouldRevealStreamingText(live) && String(event.text || '').startsWith(live.__wangpStreamingMarkdown.source)) {
    reveal.finalEvent = event;
    WAC.queueStreamingReveal(live, String(event.text || ''));
    const known = WAC.incrementalMessageState(event.message_id).blocks[String(event.block_id)];
    known.text = String(event.text || '');
    known.html = event.html;
    known.finalized = true;
    return;
  }
  if (live) WAC.clearStreamingReveals(live);
  if (!WAC.messageNode(event.message_id) && event.message) WAC.upsertMessage(event.message, true, event.message_index);
  const messageNode = WAC.messageNode(event.message_id);
  const current = WAC.blockNode(event.message_id, event.block_id);
  const next = WAC.createBlockNode(event.html);
  if (!messageNode || !next) return WAC.markSyncRequired(event);
  if (current) WAC.captureDisclosureState(current);
  const scrollState = WAC.captureAutoscrollState();
  const node = current ? WAC.patchBlockNode(current, next) : next;
  WAC.positionBlockNode(messageNode, node, event.block_index);
  WAC.rememberBlock(event, node, String(event.text || ''), true);
  WAC.applyDisclosureState(node);
  WAC.applyAutoscrollState(scrollState);
};

WAC.removeBlock = function (event) {
  const node = WAC.blockNode(event.message_id, event.block_id);
  if (node) WAC.clearStreamingReveals(node);
  const scrollState = WAC.captureAutoscrollState();
  WAC.patchCompactMedia?.(node, null);
  if (node) node.remove();
  const messageState = WAC.blockState[String(event.message_id || '')];
  if (messageState) {
    delete messageState.blocks[String(event.block_id || '')];
    messageState.order = messageState.order.filter((blockId) => blockId !== String(event.block_id || ''));
  }
  WAC.applyAutoscrollState(scrollState);
};

WAC.replayMessageBlocks = function (messageId, messageNode) {
  const messageState = WAC.blockState[String(messageId || '')];
  if (!messageState || !messageNode) return;
  for (const blockId of messageState.order) {
    const block = messageState.blocks[blockId];
    if (!block) continue;
    const next = WAC.createBlockNode(block.html);
    if (!next) continue;
    const live = WAC.liveTextNode(next);
    if (live && !block.finalized) WAC.renderStreamingMarkdown(live, String(block.text || ''));
    const existing = messageNode.querySelector(`[data-block-id="${CSS.escape(blockId)}"]`);
    const node = existing ? WAC.patchBlockNode(existing, next) : next;
    WAC.positionBlockNode(messageNode, node, block.index);
  }
};

WAC.syncAttributes = function (target, source) {
  if (!target || !source || !target.getAttributeNames || !source.getAttributeNames) return;
  const sourceNames = new Set(source.getAttributeNames());
  for (const name of target.getAttributeNames()) {
    if (!sourceNames.has(name)) target.removeAttribute(name);
  }
  for (const name of sourceNames) {
    const nextValue = source.getAttribute(name);
    if (target.getAttribute(name) !== nextValue) target.setAttribute(name, nextValue);
  }
};

WAC.patchDisclosureNode = function (current, next) {
  if (!current || !next) return;
  const wasOpen = !!current.open;
  WAC.syncAttributes(current, next);
  current.open = wasOpen;
  current.className = next.className;
  const currentSummary = current.querySelector(':scope > summary');
  const nextSummary = next.querySelector(':scope > summary');
  if (currentSummary && nextSummary && currentSummary.innerHTML !== nextSummary.innerHTML) currentSummary.innerHTML = nextSummary.innerHTML;
  const currentBody = current.querySelector(':scope > .chat__disclosure-body');
  const nextBody = next.querySelector(':scope > .chat__disclosure-body');
  if (currentBody && nextBody && WAC.patchLiveOutput(currentBody, nextBody)) return;
  if (currentBody && nextBody && currentBody.innerHTML !== nextBody.innerHTML) currentBody.innerHTML = nextBody.innerHTML;
};

// A running Python script updates only its live output (progress, printed lines): the code and the copy buttons stay as they are,
// and the output keeps following its last line unless the user scrolled up in it.
WAC.patchLiveOutput = function (currentBody, nextBody) {
  const key = currentBody.getAttribute('data-static-key');
  if (!key || key !== nextBody.getAttribute('data-static-key')) return false;
  const currentLive = currentBody.querySelector('[data-live-output]');
  const nextLive = nextBody.querySelector('[data-live-output]');
  if (!currentLive || !nextLive) return false;
  const currentMeta = currentLive.querySelector('.chat__tool-live-meta');
  const nextMeta = nextLive.querySelector('.chat__tool-live-meta');
  if (currentMeta && nextMeta && currentMeta.textContent !== nextMeta.textContent) currentMeta.textContent = nextMeta.textContent;
  const nextOut = nextLive.querySelector('.chat__tool-stdout');
  let currentOut = currentLive.querySelector('.chat__tool-stdout');
  if (nextOut && !currentOut) {
    currentOut = nextOut.cloneNode(false);
    currentLive.appendChild(currentOut);
  }
  if (currentOut && nextOut && currentOut.textContent !== nextOut.textContent) {
    const following = currentOut.scrollHeight - currentOut.scrollTop - currentOut.clientHeight < 8;
    currentOut.textContent = nextOut.textContent;
    if (following) currentOut.scrollTop = currentOut.scrollHeight;
  }
  return true;
};

WAC.patchBlockNode = function (current, next) {
  WAC.clearStreamingReveals(current);
  const blockIndex = current.dataset.blockIndex;
  WAC.patchCompactMedia?.(current, next);
  if (current.matches('.chat__disclosure')) WAC.patchDisclosureNode(current, next);
  else {
    WAC.syncAttributes(current, next);
    WAC.patchMessageBody(current, next);
  }
  if (blockIndex !== undefined) current.dataset.blockIndex = blockIndex;
  return current;
};

WAC.patchMessageBody = function (currentBody, nextBody) {
  if (!currentBody || !nextBody) return;
  WAC.flattenCompactActions?.(currentBody);
  const existingByKey = new Map();
  const nodeKey = node => node.nodeType === 1 ? node.dataset.blockId || WAC.disclosureKey(node) : '';
  currentBody.querySelectorAll(':scope > [data-block-id], :scope > .chat__disclosure').forEach((node) => {
    const key = nodeKey(node);
    if (key) existingByKey.set(key, node);
  });
  let cursor = currentBody.firstChild;
  for (const nextNode of Array.from(nextBody.childNodes)) {
    const key = nodeKey(nextNode);
    const reusable = key ? existingByKey.get(key) : null;
    if (reusable) {
      WAC.patchBlockNode(reusable, nextNode);
      existingByKey.delete(key);
      if (reusable === cursor) cursor = cursor.nextSibling;
      else currentBody.insertBefore(reusable, cursor);
      continue;
    }
    const cursorIsKeyed = cursor && nodeKey(cursor);
    if (cursor && !cursorIsKeyed) {
      const replaced = cursor;
      cursor = cursor.nextSibling;
      if (!replaced.isEqualNode(nextNode)) replaced.replaceWith(nextNode);
    } else {
      currentBody.insertBefore(nextNode, cursor);
    }
  }
  while (cursor) {
    const removed = cursor;
    cursor = cursor.nextSibling;
    removed.remove();
  }
};

WAC.patchMessageNode = function (current, next) {
  if (!current || !next) return;
  const preserveQueuedEdit = String(WAC.queuedEditMessageId || '') === String(current.getAttribute('data-message-id') || '') && !!next.querySelector('[data-message-action="edit"]');
  WAC.syncAttributes(current, next);
  current.className = next.className;
  if (preserveQueuedEdit) current.classList.add('is-editing');
  const currentAvatar = current.querySelector(':scope > .chat__avatar');
  const nextAvatar = next.querySelector(':scope > .chat__avatar');
  if (currentAvatar && nextAvatar) {
    WAC.syncAttributes(currentAvatar, nextAvatar);
    if (currentAvatar.innerHTML !== nextAvatar.innerHTML) currentAvatar.innerHTML = nextAvatar.innerHTML;
  }
  const currentCard = current.querySelector(':scope > .chat__message-card');
  const nextCard = next.querySelector(':scope > .chat__message-card');
  if (!currentCard || !nextCard) {
    current.replaceChildren(...Array.from(next.childNodes));
    return;
  }
  WAC.syncAttributes(currentCard, nextCard);
  currentCard.className = nextCard.className;
  const currentMeta = currentCard.querySelector(':scope > .chat__meta');
  const nextMeta = nextCard.querySelector(':scope > .chat__meta');
  if (currentMeta && nextMeta) {
    WAC.syncAttributes(currentMeta, nextMeta);
    currentMeta.className = nextMeta.className;
    if (currentMeta.innerHTML !== nextMeta.innerHTML) currentMeta.innerHTML = nextMeta.innerHTML;
  }
  const currentBody = currentCard.querySelector(':scope > .chat__body');
  const nextBody = nextCard.querySelector(':scope > .chat__body');
  if (currentBody && nextBody) {
    WAC.syncAttributes(currentBody, nextBody);
    currentBody.className = nextBody.className;
    WAC.patchMessageBody(currentBody, nextBody);
  }
  const currentEnd = currentCard.querySelector(':scope > .chat__message-end');
  const nextEnd = nextCard.querySelector(':scope > .chat__message-end');
  if (currentEnd && nextEnd) {
    WAC.syncAttributes(currentEnd, nextEnd);
    currentEnd.className = nextEnd.className;
    if (currentEnd.innerHTML !== nextEnd.innerHTML) currentEnd.innerHTML = nextEnd.innerHTML;
  } else if (currentEnd) {
    currentEnd.remove();
  } else if (nextEnd) {
    currentCard.appendChild(nextEnd);
  }
};

WAC.messageBodyText = function (node) {
  const body = node && node.querySelector ? node.querySelector('.chat__body') : null;
  return body ? WAC.normalizeText(body.innerText || body.textContent || '') : '';
};

WAC.upsertMessage = function (message, preserveIncrementalState, messageIndex) {
  if (!message || !message.id) return;
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (!transcript) return;
  WAC.captureDisclosureState(transcript);
  const scrollState = WAC.captureAutoscrollState();
  const node = WAC.createMessageNode(message);
  if (!node) return;
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(message.id))}"]`);
  const incomingId = String(message.id);
  if (!preserveIncrementalState) delete WAC.blockState[incomingId];
  const positioned = Number.isInteger(Number(messageIndex)) && Number(messageIndex) >= 0;
  if (positioned) {
    const nextOrder = WAC.state.order.filter((messageId) => String(messageId) !== incomingId);
    nextOrder.splice(Math.min(Number(messageIndex), nextOrder.length), 0, incomingId);
    WAC.state.order = nextOrder;
  } else if (!existing && !WAC.state.order.includes(incomingId)) {
    WAC.state.order.push(incomingId);
  }
  if (existing) {
    WAC.patchMessageNode(existing, node);
  } else {
    transcript.appendChild(node);
  }
  if (positioned) {
    const inserted = existing || node;
    const orderIndex = WAC.state.order.indexOf(incomingId);
    const beforeId = WAC.state.order[orderIndex + 1];
    const beforeNode = beforeId ? WAC.messageNode(beforeId) : null;
    if (inserted.nextElementSibling !== beforeNode) transcript.insertBefore(inserted, beforeNode);
  }
  WAC.state.messages[incomingId] = message;
  WAC.hideEmpty();
  WAC.applyDisclosureState(transcript);
  WAC.syncQueuedRequestEdit();
  window.WanGPAppPlayer?.sync(existing || node);
  WAC.applyAutoscrollState(scrollState);
};

WAC.removeMessage = function (messageId) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  const scrollState = WAC.captureAutoscrollState();
  const existing = transcript.querySelector(`[data-message-id="${CSS.escape(String(messageId))}"]`);
  if (existing) WAC.clearStreamingReveals(existing);
  if (existing) existing.remove();
  delete WAC.state.messages[String(messageId)];
  delete WAC.blockState[String(messageId)];
  WAC.state.order = WAC.state.order.filter(id => id !== String(messageId));
  WAC.syncQueuedRequestEdit();
  WAC.showEmptyIfNeeded();
  WAC.applyAutoscrollState(scrollState);
};

WAC.setRestoration = function (restoration) {
  WAC.restorationStatus = restoration ? {visible: true, kind: 'session_loading', text: restoration.text} : null;
  WAC.setStatus(WAC.state.status);
};

WAC.uploadMediaFiles = async function (files, request, notice, updateGallery, fromChat = false) {
  let lastId = null;
  for (const file of files) {
    notice('Uploading ' + file.name + '…');
    const form = new FormData(); form.append('file', file);
    if (fromChat) form.append('from_chat', 'true');
    const result = await request('media', form);
    updateGallery?.(result.gallery);
    lastId = result.id;
  }
  return lastId;
};

WAC.setPendingUploads = function (items) {
  WAC.pendingUploads = items;
  WAC.refreshChatDraftActions?.();
  const strip = document.getElementById('assistant_chat_pending_uploads');
  if (!strip) return;
  const html = items.map(item => item.html).join('');
  // Repeated events carry the same previews: keep their loaded images.
  if (WAC.pendingUploadsHtml !== html) {WAC.pendingUploadsHtml = html; strip.innerHTML = html;}
  strip.hidden = !items.length;
};

WAC.mountChatUpload = function (request, notice, updateGallery) {
  const controls = document.getElementById('assistant_chat_controls');
  if (!controls || document.getElementById('assistant_chat_upload_button')) return;
  const button = document.createElement('button');
  button.id = 'assistant_chat_upload_button'; button.type = 'button';
  button.title = 'Upload images, videos or audio'; button.setAttribute('aria-label', button.title);
  button.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>';
  const input = document.createElement('input');
  // Audio by extension: iOS Safari filters audio/* as video and greys out audio files.
  input.type = 'file'; input.accept = 'image/*,video/*,.wav,.mp3,.aac,.flac,.m4a,.ogg,.opus,.wma'; input.multiple = true; input.hidden = true;
  const status = document.createElement('span');
  status.id = 'assistant_chat_upload_status'; status.setAttribute('role', 'status'); status.hidden = true;
  const uploadNotice = text => { status.textContent = text; status.hidden = !text; };
  const strip = document.createElement('div');
  strip.id = 'assistant_chat_pending_uploads'; strip.hidden = true;
  WAC.pendingUploadsHtml = '';
  strip.onclick = async event => {
    const remove = event.target.closest('[data-unattach]');
    if (!remove) return;
    remove.disabled = true;
    try {WAC.consumePayload((await request('media/unattach', {chat_session_id: WAC.chatSessionId, media_id: remove.dataset.unattach})).event);}
    catch (error) {uploadNotice(error.message); remove.disabled = false;}
  };
  button.onclick = () => input.click();
  input.onchange = async () => {
    if (!input.files.length) return;
    button.disabled = true; button.setAttribute('aria-busy', 'true');
    const files = Array.from(input.files);
    const uploading = (async () => {
      const id = await WAC.uploadMediaFiles(files, request, uploadNotice, updateGallery, true);
      await request('media/' + encodeURIComponent(id) + '/select', {});
    })();
    WAC.pendingChatUpload = uploading;
    try {
      await uploading;
      uploadNotice('');
    } catch (error) { uploadNotice(error.message); }
    finally { if (WAC.pendingChatUpload === uploading) WAC.pendingChatUpload = null; button.disabled = false; button.removeAttribute('aria-busy'); input.value = ''; }
  };
  controls.append(button, input);
  if (document.body.hasAttribute('data-deepy-app')) {
    const clear = document.createElement('button'); clear.id = 'assistant_chat_clear_draft'; clear.type = 'button';
    clear.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></svg>';
    let clearing = false;
    WAC.refreshChatDraftActions = () => {
      const hasText = !!WAC.requestInput().value;
      clear.disabled = clearing || (!hasText && !WAC.pendingUploads.length);
      clear.title = hasText ? 'Clear text' : 'Remove last attachment'; clear.setAttribute('aria-label', clear.title);
    };
    clear.onclick = async () => {
      if (WAC.requestInput().value) {WAC.setRequestInputValue(''); return;}
      clearing = true; WAC.refreshChatDraftActions();
      try {WAC.consumePayload((await request('media/unattach', {chat_session_id: WAC.chatSessionId})).event);}
      catch (error) {uploadNotice(error.message);}
      finally {clearing = false; WAC.refreshChatDraftActions();}
    };
    WAC.requestInput().addEventListener('input', WAC.refreshChatDraftActions);
    controls.append(status, strip, clear);
  }
  else {controls.before(strip); controls.after(status);}
  WAC.setPendingUploads(WAC.pendingUploads || []);
};

WAC.waitForChatUpload = async function () {
  if (!WAC.pendingChatUpload) return;
  const status = document.getElementById('assistant_chat_upload_status');
  status.textContent = 'Message queued — waiting for media upload…'; status.hidden = false;
  await WAC.pendingChatUpload;
};

WAC.setStatus = function (status, restoreAnchor) {
  WAC.state.status = status || null;
  const effective = WAC.restorationStatus || status;
  const node = WAC.statusNode();
  const key = effective?.visible && effective.kind === 'tool' ? WAC.chatSessionId + ':' + effective.text : null;
  if (key && node?.classList.contains('is-visible') && node.querySelector('.chat__status-text')?.title !== String(effective.text)) {
    if (WAC.pendingToolStatus?.key === key) return;
    clearTimeout(WAC.pendingToolStatus?.timer);
    WAC.pendingToolStatus = {key, timer: setTimeout(() => {
      WAC.pendingToolStatus = null;
      WAC.renderStatus(WAC.state.status, restoreAnchor);
    }, 1000)};
    return;
  }
  clearTimeout(WAC.pendingToolStatus?.timer);
  WAC.pendingToolStatus = null;
  WAC.renderStatus(status, restoreAnchor);
};

WAC.syncStatusLayout = function (scrollState = WAC.captureAutoscrollState()) {
  const node = WAC.statusNode();
  if (node?.classList.contains('is-visible')) {
    const height = node.offsetHeight + 'px';
    if (node.parentElement.style.getPropertyValue('--chat-status-height') !== height) node.parentElement.style.setProperty('--chat-status-height', height);
  }
  WAC.applyAutoscrollState(scrollState);
};

WAC.renderStatus = function (status, restoreAnchor) {
  WAC.ensureShell();
  const scrollState = WAC.captureAutoscrollState();
  status = WAC.restorationStatus || status;
  const restoring = !!WAC.restorationStatus;
  document.querySelectorAll('button#assistant_chat_reset_button, #assistant_chat_reset_button button').forEach(button => { button.disabled = restoring; });
  const node = WAC.statusNode();
  if (!node) return;
  const textNode = node.querySelector('.chat__status-text');
  const pauseNode = node.querySelector('.chat__status-pause');
  const stopNode = node.querySelector('.chat__status-stop');
  if (!status || !status.visible || !status.text) {
    WAC.renderStatusProgress(node, null);
    node.classList.remove('is-visible');
    node.removeAttribute('data-kind');
    if (textNode) textNode.textContent = '';
    if (pauseNode) {
      pauseNode.textContent = 'Pause';
      pauseNode.setAttribute('aria-label', 'Pause Deepy');
      pauseNode.dataset.mode = 'pause';
      pauseNode.hidden = false;
      pauseNode.disabled = true;
    }
    if (stopNode) {
      stopNode.hidden = false;
      stopNode.disabled = true;
    }
    WAC.setBusyInputHelper(false);
    WAC.syncStatusLayout(scrollState);
    return;
  }
  if (textNode) { textNode.textContent = String(status.text).replace(/_/g, '_\u200b'); textNode.title = String(status.text); }
  WAC.renderStatusProgress(node, status);
  const kind = String(status.kind || 'status');
  node.dataset.kind = kind;
  if (pauseNode) {
    const isPaused = kind === 'paused';
    pauseNode.hidden = kind === 'session_loading';
    pauseNode.textContent = isPaused ? 'Resume' : kind === 'pause_pending' ? 'Pausing…' : kind === 'resuming' ? 'Resuming…' : 'Pause';
    pauseNode.setAttribute('aria-label', isPaused ? 'Resume Deepy' : 'Pause Deepy');
    pauseNode.dataset.mode = isPaused ? 'resume' : 'pause';
    pauseNode.disabled = kind === 'pause_pending' || kind === 'resuming' || kind === 'session_loading' || kind === 'stop_pending';
  }
  if (stopNode) {
    stopNode.hidden = kind === 'session_loading';
    stopNode.disabled = kind === 'session_loading';
  }
  node.classList.add('is-visible');
  WAC.setBusyInputHelper(kind !== 'session_loading');
  WAC.syncStatusLayout(scrollState);
};

// A detail line and a progress bar under the status text (a running Python script reports both), then the tool that script is calling.
WAC.renderStatusProgress = function (node, status) {
  if (!node.querySelector('.chat__status-main')) {
    const textNode = node.querySelector('.chat__status-text');
    if (!textNode) return;
    const main = document.createElement('div');
    main.className = 'chat__status-main';
    main.innerHTML = '<div class="chat__status-detail" hidden></div><div class="chat__status-progress" hidden><span></span></div>';
    textNode.replaceWith(main);
    main.prepend(textNode);
  }
  if (!node.querySelector('.chat__status-sub')) {
    const subNode = document.createElement('div');
    subNode.className = 'chat__status-sub';
    subNode.hidden = true;
    node.querySelector('.chat__status-main').appendChild(subNode);
  }
  const detailNode = node.querySelector('.chat__status-detail');
  const barNode = node.querySelector('.chat__status-progress');
  const detail = status && status.detail ? String(status.detail) : '';
  if (detailNode) {
    if (detailNode.textContent !== detail) { detailNode.textContent = detail; detailNode.title = detail; }
    detailNode.hidden = !detail;
  }
  const progress = status && status.progress && Number(status.progress.total) > 0 ? status.progress : null;
  if (barNode) {
    barNode.hidden = !progress;
    const fill = barNode.firstElementChild;
    if (progress && fill) fill.style.width = Math.max(0, Math.min(100, 100 * Number(progress.done || 0) / Number(progress.total))).toFixed(1) + '%';
  }
  const subNode = node.querySelector('.chat__status-sub');
  const sub = status && status.sub ? String(status.sub) : '';
  if (subNode) {
    if (subNode.textContent !== sub) { subNode.textContent = sub; subNode.title = sub; }
    subNode.hidden = !sub;
  }
};

WAC.setStats = function (stats) {
  WAC.ensureShell();
  WAC.state.stats = stats || null;
  const node = WAC.statsNode();
  if (!node) return;
  let textNode = node.querySelector('.chat__stats-text');
  if (!textNode) {
    textNode = document.createElement('span');
    textNode.className = 'chat__stats-text';
    node.appendChild(textNode);
  }
  if (!stats || stats.visible === false || !stats.text) {
    node.classList.remove('is-visible');
    textNode.textContent = '';
    textNode.removeAttribute('title');
    node.setAttribute('aria-hidden', node.classList.contains('has-input-helper') ? 'false' : 'true');
    return;
  }
  textNode.textContent = String(stats.text);
  textNode.title = String(stats.text);
  node.classList.add('is-visible');
  node.setAttribute('aria-hidden', 'false');
};

WAC.sync = function (messages, status, stats, acknowledgedSubmissionIds) {
  WAC.clearStreamingReveals();
  WAC.ensureShell();
  WAC.captureDisclosureState(WAC.transcript());
  const followSubmittedRequest = WAC.syncAcknowledgesFollowedSubmission(messages, acknowledgedSubmissionIds);
  const scrollState = followSubmittedRequest ? { atBottom: true, top: 0 } : WAC.captureAutoscrollState();
  WAC.blockState = {};
  WAC.replaceState(messages, status, stats);
  WAC.reconcileOptimisticSubmits(acknowledgedSubmissionIds);
  WAC.hydrate(scrollState);
  if (followSubmittedRequest) {
    WAC.followSubmissionId = '';
    WAC.scrollToBottomAfterLayout();
  }
};

WAC.reset = function () {
  WAC.clearStreamingReveals();
  WAC.setPendingUploads([]);
  WAC.compactScrollState = null;
  if (WAC.queuedEditMessageId) WAC.finishQueuedRequestEdit();
  WAC.state = { order: [], messages: {}, status: null, stats: null };
  WAC.blockState = {};
  WAC.optimisticSubmits = [];
  WAC.pendingSteeringId = '';
  WAC.followSubmissionId = '';
  WAC.chatSessionId = '';
  WAC.chatRevision = -1;
  WAC.chatSequence = -1;
  WAC.syncRequired = false;
  WAC.syncRecoveryPending = false;
  WAC.disclosureState = {};
  WAC.ensureShell();
  const transcript = WAC.transcript();
  if (transcript) transcript.innerHTML = '';
  WAC.showEmptyIfNeeded();
  WAC.setStatus(null);
  WAC.setStats(null);
};

WAC.hydrate = function (scrollState) {
  const transcript = WAC.transcript();
  if (!transcript) return;
  const existingById = new Map();
  transcript.querySelectorAll(':scope > [data-message-id]').forEach((node) => {
    const messageId = String(node.getAttribute('data-message-id') || '');
    if (messageId) existingById.set(messageId, node);
  });
  let cursor = transcript.firstElementChild;
  for (const messageId of WAC.state.order) {
    const message = WAC.state.messages[messageId];
    if (!message) continue;
    const node = WAC.createMessageNode(message);
    if (!node) continue;
    WAC.replayMessageBlocks(messageId, node);
    const existing = existingById.get(String(messageId));
    if (existing) {
      WAC.patchMessageNode(existing, node);
      existingById.delete(String(messageId));
      if (existing === cursor) cursor = cursor.nextElementSibling;
      else transcript.insertBefore(existing, cursor);
    } else {
      transcript.insertBefore(node, cursor);
    }
  }
  for (const obsolete of existingById.values()) obsolete.remove();
  WAC.applyDisclosureState(transcript);
  WAC.syncQueuedRequestEdit();
  WAC.showEmptyIfNeeded();
  WAC.setStatus(WAC.state.status, null);
  WAC.setStats(WAC.state.stats);
  WAC.applyAutoscrollState(scrollState);
};

WAC.applyEvent = function (payload) {
  return WAC.consumePayload(payload);
};

WAC.syncDisclosureBridge = function () {
  const transcript = WAC.transcript();
  if (!transcript || transcript === WAC.disclosureNode) return;
  if (WAC.disclosureNode) {
    WAC.disclosureNode.removeEventListener('toggle', WAC.handleDisclosureToggle, true);
  }
  WAC.disclosureNode = transcript;
  WAC.disclosureNode.addEventListener('toggle', WAC.handleDisclosureToggle, true);
};

WAC.handleThumbnailError = function (event) {
  const image = event.target;
  const icon = image.nextElementSibling;
  if (image.tagName !== 'IMG' || !image.classList.contains('chat__attachment-thumb') || !icon || !icon.hasAttribute('data-attachment-icon')) return;
  image.hidden = true;  // a preview that could not be made: its typed icon instead of a broken image
  icon.hidden = false;
};

WAC.handleScroll = function () {
  if (WAC.scroll().clientHeight > 0) WAC.lastScrollState = WAC.captureAutoscrollState();
  // A pending resize must not restore an older position after the user has scrolled.
  if (WAC.composerResizeFrame) WAC.composerResizeScrollState = WAC.captureAutoscrollState();
  WAC.syncJumpToBottom();
};

WAC.syncScrollBridge = function () {
  const scroll = WAC.scroll();
  if (!scroll || scroll === WAC.scrollNode) {
    WAC.syncJumpToBottom();
    return;
  }
  if (WAC.scrollNode) WAC.scrollNode.removeEventListener('scroll', WAC.handleScroll, { passive: true });
  WAC.scrollNode = scroll;
  WAC.lastScrollState = WAC.captureAutoscrollState();
  WAC.scrollNode.addEventListener('scroll', WAC.handleScroll, { passive: true });
  // Media/layout changes can reach the bottom without changing scrollTop or firing scroll.
  WAC.jumpBottomResizeObserver?.disconnect();
  WAC.jumpBottomResizeObserver = new ResizeObserver(() => WAC.syncJumpToBottom());
  WAC.jumpBottomResizeObserver.observe(scroll);
  WAC.jumpBottomResizeObserver.observe(WAC.transcript());
  WAC.statusResizeObserver?.disconnect();
  const status = WAC.statusNode();
  if (status) {
    WAC.statusResizeObserver = new ResizeObserver(() => WAC.syncStatusLayout());
    WAC.statusResizeObserver.observe(status);
  }
  WAC.syncJumpToBottom();
};

WAC.installObserver = function () {
  if (WAC.observer) return;
  const target = document.querySelector('gradio-app') || document.body;
  if (!target) return;
  const scope = '[id^="assistant_chat_"], #deepy_type_value, #deepy_type_choice';
  const containsChat = node => node.nodeType === 1 && (node.matches(scope) || node.querySelector(scope));
  // Retain Gradio mount/replacement detection, but do not refresh Deepy for
  // unrelated galleries, playback clocks, progress bars, or form fields.
  WAC.observer = new MutationObserver((mutations) => {
      if (!mutations.some(mutation => mutation.target.closest?.(scope) ||
          Array.from(mutation.addedNodes).some(containsChat) ||
          Array.from(mutation.removedNodes).some(containsChat))) return;
      if (WAC.observerScheduled) return;
      WAC.observerScheduled = true;
      window.requestAnimationFrame(() => {
        WAC.observerScheduled = false;
        if (WAC.host()) WAC.ensureShell();
        WAC.syncScrollBridge();
        WAC.syncThemeState();
        WAC.syncDockLayout();
        WAC.syncSettingsDropdowns();
        WAC.syncDeepyTypePreview();
        WAC.setQueuedEditButtonLabels(!!WAC.queuedEditMessageId);
        WAC.handleEventNodeMutation();
        WAC.readEventSource();
        WAC.observeDockEnvironment();
      });
  });
  WAC.observer.observe(target, { childList: true, subtree: true });
  // Geometry and theme have explicit, narrow sources instead of observing
  // every class/style mutation on the page (including our own layout writes).
  WAC.dockResizeObserver = new ResizeObserver(() => WAC.syncDockLayout());
  WAC.themeObserver = new MutationObserver(() => WAC.syncThemeState());
  WAC.observeDockEnvironment = function () {
    const dock = WAC.dock();
    const parent = dock?.parentElement;
    if (WAC.observedDockParent !== parent) {
      WAC.dockResizeObserver.disconnect();
      WAC.observedDockParent = parent;
      const candidates = [parent, parent?.closest('.column'), parent?.parentElement?.closest('.column')];
      new Set(candidates.filter(Boolean)).forEach(node => WAC.dockResizeObserver.observe(node));
    }
    WAC.themeObserver.disconnect();
    [document.documentElement, document.body, document.querySelector('gradio-app'), document.querySelector('.gradio-container')]
      .filter(Boolean).forEach(node => WAC.themeObserver.observe(node, {attributes: true, attributeFilter: ['class', 'style', 'data-theme', 'theme']}));
  };
  WAC.observeDockEnvironment();
};


WAC.installDockBridge = function () {
  if (WAC.dockBridgeInstalled) return;
  WAC.dockBridgeInstalled = true;
  window.addEventListener('resize', WAC.syncSettingsDropdowns);
  document.addEventListener('scroll', (event) => {
    if (event.target.matches?.('.chat__settings-scroll')) WAC.syncSettingsDropdowns();
  }, true);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) WAC.clearStreamingReveals(null, true);
  });
  WAC.dockOpen = false;
  try { window.localStorage.removeItem('wangp-assistant-chat-open'); } catch (_error) {}
  document.addEventListener('beforeinput', (event) => {
    const input = WAC.requestInput();
    // A pending resize has already moved the layout; keep the intent captured before it.
    if (input && event.target === input && !WAC.composerResizeFrame) WAC.composerResizeScrollState = WAC.captureAutoscrollState();
  }, true);
  document.addEventListener('input', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) WAC.syncDeepyTypePreview();
    const input = WAC.requestInput();
    if (input && event.target === input) WAC.scheduleComposerLayout();
  }, true);
  document.addEventListener('focusin', (event) => {
    const input = WAC.requestInput();
    if (input && event.target === input) WAC.syncComposerLayout();
  }, true);
  document.addEventListener('change', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) WAC.syncDeepyTypePreview();
    const sessionPicker = event.target && event.target.closest ? event.target.closest('[data-wac-session-picker]') : null;
    if (sessionPicker) {
      const container = sessionPicker.closest('.chat__session-picker');
      const resumeButton = container ? container.querySelector('[data-wac-session-resume]') : null;
      if (resumeButton) resumeButton.disabled = !String(sessionPicker.value || '').trim();
    }
  }, true);
  document.addEventListener('click', (event) => {
    if (event.target && event.target.closest && event.target.closest('#deepy_type_choice')) window.setTimeout(WAC.syncDeepyTypePreview, 0);
  }, true);
  document.addEventListener('pointerdown', (event) => {
    WAC.disclosurePointer = null;
    WAC.attachmentTouch = null;
    if (WAC.handleCollapseButtonPointerDown(event)) return;
    if (WAC.handleDisclosurePointerDown(event)) return;
    if (event.target.matches('.chat__compact-section[open]') && WAC.startDisclosurePointer(event, event.target, true)) return;
    if (WAC.handleAttachmentPointerDown(event)) return;
  }, true);
  for (const type of ['pointermove', 'pointerup', 'pointercancel']) document.addEventListener(type, event => {
    WAC.trackAttachmentTouch(event);
    WAC.trackDisclosurePointer(event);
  }, {capture: true, passive: true});
  for (const type of ['contextmenu', 'dragstart']) document.addEventListener(type, event => {
    if (window.matchMedia('(pointer: coarse)').matches && event.target.closest('.chat__attachment')) {
      if (WAC.attachmentTouch) WAC.attachmentTouch.moved = true;
      event.preventDefault();
    }
  }, true);
  document.addEventListener('click', (event) => {
    // Opening a block can move another control under the finger before the synthetic click.
    if (event.detail !== 0 && WAC.disclosurePointer) {
      WAC.disclosurePointer = null;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (WAC.handleAttachmentClick(event)) return;
    if (WAC.handleCopyButtonClick(event)) return;
    if (WAC.handleCollapseButtonClick(event)) return;
    if (WAC.handleQueuedRequestClick(event)) return;
    const attachmentLink = event.target && event.target.closest ? event.target.closest('.chat__attachment, .chat__body a') : null;
    if (attachmentLink) return;
    const disclosureSummary = event.target && event.target.closest ? event.target.closest('summary') : null;
    if (disclosureSummary) {
      const disclosureNode = disclosureSummary.parentElement;
      if (disclosureNode && disclosureNode.matches('.chat__disclosure, .chat__compact-section')) {
        event.preventDefault();
        event.stopPropagation();
        WAC.toggleDisclosure(disclosureNode);
        WAC.disclosurePointer = null;
        return;
      }
    }
    const toggle = event.target && event.target.closest ? event.target.closest('#assistant_chat_toggle') : null;
    if (toggle) {
      event.preventDefault();
      WAC.toggleDock();
      return;
    }
    const settingsToggle = event.target && event.target.closest ? event.target.closest('#assistant_chat_settings_toggle') : null;
    if (settingsToggle) {
      event.preventDefault();
      WAC.toggleSettings();
      return;
    }
    const pauseButton = event.target && event.target.closest ? event.target.closest('.chat__status-pause') : null;
    if (pauseButton) {
      event.preventDefault();
      event.stopPropagation();
      const target = WAC.pauseBridgeTargets()[0];
      if (target && typeof target.click === 'function') target.click();
      return;
    }
    const stopButton = event.target && event.target.closest ? event.target.closest('.chat__status-stop') : null;
    if (stopButton) {
      event.preventDefault();
      event.stopPropagation();
      const target = WAC.stopBridgeTargets()[0];
      if (target && typeof target.click === 'function') target.click();
      return;
    }
    const jumpBottomButton = event.target && event.target.closest ? event.target.closest('.chat__jump-bottom') : null;
    if (jumpBottomButton) {
      event.preventDefault();
      WAC.scrollToBottom();
      return;
    }
    const resetButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_reset_button') : null;
    if (WAC.restorationStatus && resetButton) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (resetButton && WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      WAC.finishQueuedRequestEdit();
      return;
    }
    const askButton = event.target && event.target.closest ? event.target.closest('#assistant_chat_ask_button') : null;
    if (!askButton) return;
    if (WAC.finishVoiceBeforeSubmit?.()) { event.preventDefault(); event.stopPropagation(); return; }
    const input = WAC.requestInput();
    const text = input ? String(input.value || '').trim() : '';
    if (WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      if (!text) {
        if (input) input.focus({ preventScroll: true });
        return;
      }
      const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(WAC.queuedEditMessageId)}"]`);
      WAC.submitQueuedRequestAction(messageNode, 'edit', text);
      return;
    }
    if (!text) return;
    WAC.setDockOpen(true, !WAC.submitRequest);
    const busy = WAC.isAssistantBusy();
    const submissionId = WAC.pushOptimisticUserMessage(text, busy ? 'Queued' : '');
    WAC.setBusyInputHelper(true);
    WAC.setBridgeValue('#assistant_chat_submission_id textarea, #assistant_chat_submission_id input', submissionId);
    if (WAC.submitRequest) {
      event.preventDefault();
      event.stopPropagation();
      WAC.submitRequest(text, submissionId);
      return;
    }
    if (busy) {
      if (WAC.queueBusyRequest(text, submissionId)) {
        event.preventDefault();
        event.stopPropagation();
        WAC.clearRequestInput(text);
        return;
      }
    }
  }, true);
  document.addEventListener('keydown', (event) => {
    // Escape in a full-screen viewer only leaves full screen.
    if (event.target.closest?.('dialog[open]') || document.fullscreenElement) return;
    if (event.key === 'Escape' && WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      WAC.finishQueuedRequestEdit();
      return;
    }
    if (event.key !== 'Escape') return;
    if (WAC.settingsOpen) {
      WAC.setSettingsOpen(false);
      return;
    }
    if (!WAC.dockOpen) return;
    WAC.setDockOpen(false);
  }, true);
  document.addEventListener('keydown', (event) => {
    const input = WAC.requestInput();
    if (!input || event.target !== input || event.key !== 'Enter' || event.shiftKey || event.altKey) return;
    if (WAC.finishVoiceBeforeSubmit?.()) { event.preventDefault(); event.stopPropagation(); return; }
    const text = String(input.value || '').trim();
    if (WAC.queuedEditMessageId) {
      event.preventDefault();
      event.stopPropagation();
      if (!text) {
        input.focus({ preventScroll: true });
        return;
      }
      const messageNode = WAC.transcript() && WAC.transcript().querySelector(`[data-message-id="${CSS.escape(WAC.queuedEditMessageId)}"]`);
      WAC.submitQueuedRequestAction(messageNode, 'edit', text);
      return;
    }
    if (!text) return;
    event.preventDefault();
    event.stopPropagation();
    WAC.setDockOpen(true, !WAC.submitRequest);
    if (event.ctrlKey || event.metaKey) {
      const submissionId = WAC.pushOptimisticUserMessage(text, 'Steered');
      if (WAC.steerRequest(text, submissionId)) {
        WAC.pendingSteeringId = submissionId;
        WAC.setStatus({ visible: true, kind: 'queued', text: 'Steering requested. Waiting for the current thought/action boundary...' });
        WAC.setBusyInputHelper(true);
        window.setTimeout(() => { if (WAC.pendingSteeringId === submissionId) WAC.pendingSteeringId = ''; }, WAC.optimisticMaxAgeMs);
        window.setTimeout(() => { WAC.clearRequestInput(text); }, 0);
      } else {
        WAC.dropOptimisticSubmit(submissionId);
        WAC.removeMessage(submissionId);
      }
      return;
    }
    const askButton = document.querySelector('#assistant_chat_ask_button button, #assistant_chat_ask_button');
    if (askButton && typeof askButton.click === 'function') askButton.click();
  }, true);
  WAC.syncDockState();
  WAC.syncDockLayout();
  WAC.setQueuedEditButtonLabels(!!WAC.queuedEditMessageId);
};

if (!WAC.init) {
  document.addEventListener('error', WAC.handleThumbnailError, true);  // load errors do not bubble: caught on the way down, on every page that shows the chat
  WAC.installObserver();
  WAC.installEventBridge();
  WAC.installDockBridge();
  WAC.init = true;
}

setTimeout(() => { WAC.ensureShell(); WAC.syncDeepyTypePreview(); WAC.handleEventNodeMutation(); WAC.readEventSource(); WAC.syncDockState(); WAC.syncDockLayout(); WAC.syncComposerLayout(); }, 50);
if (window.__wangpAssistantChatPending.length > 0) {
  const pending = window.__wangpAssistantChatPending.slice();
  window.__wangpAssistantChatPending.length = 0;
  for (const payload of pending) WAC.consumePayload(payload);
}
window.applyAssistantChatEvent = function (payload) {
  return WAC.consumePayload(payload);
};
