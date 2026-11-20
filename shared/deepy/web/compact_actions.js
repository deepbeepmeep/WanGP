// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* Presentation only: canonical messages and blocks remain flat and unchanged. */
(() => {
  const WAC = window.__wangpAssistantChatNS;
  let compactByDefault = true;
  let presentation = {active_message_id: '', active_user_id: '', durations: {}};
  let scheduled = false;
  let refreshAll = false;
  const dirtyMessages = new Set();
  let clock = null;
  let clockTimer = null;

  function durationLabel(duration) {
    const seconds = Math.round(duration);
    return seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
  }

  function updateThinkingTime() {
    const heading = WAC.transcript()?.querySelector('.chat__compact-thinking .chat__compact-title');
    if (!heading) return;
    const elapsed = clock ? clock.elapsed + (clock.paused ? 0 : (performance.now() - clock.received) / 1000) : null;
    const title = elapsed === null ? 'Working…' : `Working… ${durationLabel(elapsed)}`;
    if (heading.textContent !== title) heading.textContent = title;
  }

  function syncClockTimer() {
    if (clock && !clock.paused && !document.hidden) {
      if (clockTimer === null) clockTimer = setInterval(updateThinkingTime, 1000);
    } else if (clockTimer !== null) {
      clearInterval(clockTimer);
      clockTimer = null;
    }
    if (!document.hidden) updateThinkingTime();
  }
  document.addEventListener('visibilitychange', syncClockTimer);

  function holdScroll(body) {
    if (body._compactSignature) WAC.compactScrollState ||= WAC.captureAutoscrollState();
  }

  // Read the canonical order without detaching the displayed sections. Reparenting the
  // entire turn interrupts native touch scrolling even when scrollTop stays unchanged.
  WAC.compactChildren = body => {
    if (!body.classList.contains('chat__body')) return null;
    holdScroll(body);
    const media = new Set((body._compactMedia || []).map(item => item.node));
    const children = parent => Array.from(parent.childNodes).flatMap(node => {
      if (node === body._compactStatement?._compactSlot) return [body._compactStatement];
      if (node.nodeType !== 1) return [];
      if (media.has(node) || node === body._compactMediaPreview) return [];
      return node.dataset.compactKey ? children(node.lastElementChild) : [node];
    });
    return children(body);
  };

  WAC.patchCompactMedia = (current, next) => {
    const body = current?.closest('.chat__body');
    if (!body?._compactSignature) return;
    holdScroll(body);
    body._compactDirty = true;
    if (!next && current._compactSlot) {
      current._compactSlot.remove();
      current._compactSlot = null;
      body._compactStatement = null;
    }
    const media = current._compactMedia;
    if (!media) return;
    const attachment = next?.querySelector(':scope > .chat__attachments');
    if (attachment) {
      WAC.syncAttributes(media.node, attachment);
      WAC.patchMessageBody(media.node, attachment);
      // Keep the media at its displayed position while patching the owning block.
      attachment.replaceWith(media.slot);
    } else {
      media.node.remove();
      body._compactMedia.splice(body._compactMedia.indexOf(media), 1);
      current._compactMedia = null;
    }
  };

  WAC.canonicalBlockHTML = node => {
    const media = node._compactMedia;
    if (!media) return node.outerHTML;
    const clone = node.cloneNode(true);
    clone.childNodes[Array.from(node.childNodes).indexOf(media.slot)].replaceWith(media.node.cloneNode(true));
    return clone.outerHTML;
  };

  function reconcileChildren(parent, nodes) {
    let cursor = parent.firstChild;
    for (const node of nodes) {
      if (node === cursor) cursor = cursor.nextSibling;
      else parent.insertBefore(node, cursor);
    }
    while (cursor) {
      const next = cursor.nextSibling;
      cursor.remove();
      cursor = next;
    }
  }

  function restoreStatement(body) {
    const statement = body._compactStatement;
    if (!statement) return;
    statement._compactSlot.replaceWith(statement);
    statement._compactSlot = null;
    body._compactStatement = null;
  }

  WAC.flattenCompactActions = body => {
    if (!body?.classList.contains('chat__body')) return;
    // Keep the pre-update position until regrouping finishes, without measuring the flat intermediate DOM.
    holdScroll(body);
    if (body.contains(document.activeElement) && document.activeElement !== body) body._compactFocus = document.activeElement;
    restoreStatement(body);
    for (const media of body._compactMedia || []) {
      media.slot.parentElement._compactMedia = null;
      media.slot.replaceWith(media.node);
    }
    body._compactMedia = [];
    body._compactMediaPreview?.remove();
    const unwrap = parent => {
      for (const child of Array.from(parent.children)) {
        if (!child.dataset.compactKey) continue;
        const content = child.lastElementChild;
        unwrap(content);
        child.replaceWith(...Array.from(content.childNodes));
      }
    };
    unwrap(body);
    body._compactSignature = '';
  };

  function section(body, id) {
    let node = body._compactTurn;
    if (!node) {
      node = document.createElement('details');
      node.className = 'chat__compact-section chat__compact-turn';
      node.dataset.compactKey = 'turn:' + id;
      node.open = !compactByDefault;
      node.append(document.createElement('summary'), document.createElement('div'));
      node.firstElementChild.innerHTML = '<span class="chat__compact-title"></span><span class="chat__compact-counts"></span>';
      node.lastElementChild.className = 'chat__compact-content';
      node.onclick = event => {
        if (event.target !== node || !node.open) return;
        WAC.closeDisclosure(node, event.detail ? event.clientY : undefined);
      };
      node._syncDisclosureLayout = () => placeTurnContent(body, node);
      node.ontoggle = () => {
        // User gestures and render updates already settled the layout synchronously.
        if (node.parentElement !== body || node._compactOpen === node.open) return;
        const scroll = WAC.captureAutoscrollState();
        node._syncDisclosureLayout();
        WAC.applyAutoscrollState(scroll);
      };
      body._compactTurn = node;
    }
    return node;
  }

  function placeTurnContent(body, turn) {
    const statement = turn.open || body._compactFinal ? null : body._compactLatest;
    if (body._compactStatement !== statement) {
      restoreStatement(body);
      if (statement) {
        const slot = document.createComment('compact statement');
        statement.replaceWith(slot);
        statement._compactSlot = slot;
        body._compactStatement = statement;
      }
    }
    const mediaSet = new Set(body._compactMedia.map(media => media.node));
    const preview = body._compactMediaPreview ||= document.createElement('div');
    preview.className = 'chat__compact-media-preview';
    const visible = turn.open ? [] : body._compactLayout.filter(node => mediaSet.has(node) || node === statement);
    reconcileChildren(turn.lastElementChild, body._compactLayout.filter(node => turn.open || !mediaSet.has(node)).map(node => node._compactSlot || node));
    reconcileChildren(preview, visible);
    reconcileChildren(body, [turn, ...(visible.length ? [preview] : []), ...(body._compactFinal ? [body._compactFinal] : [])]);
    turn._compactOpen = turn.open;
  }

  function updateMessage(message) {
    const body = message.querySelector('.chat__body');
    if (!body) return;
    const original = WAC.compactChildren(body);
    const nodes = original.filter(node => node.dataset.blockId);
    const id = message.dataset.messageId;
    const interrupted = /Interrupted|Stopped|Error/i.test(Array.from(message.querySelectorAll('.chat__message-end, .chat__badge'), node => node.textContent).join(' '));
    const waitingForFirstAction = presentation.active_user_id && !presentation.active_message_id && WAC.state.order.indexOf(id) > WAC.state.order.indexOf(presentation.active_user_id);
    const completed = id !== presentation.active_message_id && !waitingForFirstAction;
    const last = nodes[nodes.length - 1];
    const final = completed && !interrupted && last?.dataset.blockType === 'markdown' ? last : null;
    const duration = presentation.durations[id];
    const signature = JSON.stringify([nodes.map(node => [node.dataset.blockId, node.dataset.blockType]), completed, !!final, interrupted, duration]);
    if (signature === body._compactSignature && !body._compactDirty) return;
    body._compactMedia = [];
    const layout = [];
    for (const node of original) {
      layout.push(node);
      let media = node._compactMedia;
      if (node === final) {
        if (media) {
          media.slot.replaceWith(media.node);
          node._compactMedia = null;
        }
      }
      else {
        const attachments = media?.node || node.querySelector(':scope > .chat__attachments');
        if (attachments?.querySelector('[data-media-kind="image"], [data-media-kind="video"], [data-media-kind="audio"]')) {
          if (!media) {
            const slot = document.createComment('compact media');
            attachments.replaceWith(slot);
            media = {node: attachments, slot};
            node._compactMedia = media;
          }
          body._compactMedia.push(media);
          layout.push(attachments);
        } else if (media) {
          media.slot.replaceWith(media.node);
          node._compactMedia = null;
        }
      }
    }
    if (layout.length > (final ? 1 : 0)) {
      const active = !completed && !interrupted;
      let title = active ? 'Working…' : 'Processed Query';
      if (!active && Number.isFinite(duration)) {
        title += ` in ${durationLabel(duration)}`;
      }
      if (interrupted) title += ' — Interrupted';
      const turn = section(body, id);
      const titleNode = turn.firstElementChild.firstElementChild;
      if (titleNode.textContent !== title) titleNode.textContent = title;
      const thoughts = nodes.filter(node => ['reasoning', 'context_thinking'].includes(node.dataset.blockType)).length;
      const tools = nodes.filter(node => node.dataset.blockType === 'tool').length;
      const counts = ` · ${thoughts} ${thoughts === 1 ? 'thought' : 'thoughts'} · ${tools} ${tools === 1 ? 'tool call' : 'tool calls'}`;
      const countsNode = turn.firstElementChild.lastElementChild;
      if (countsNode.textContent !== counts) countsNode.textContent = counts;
      turn.firstElementChild.classList.toggle('chat__compact-thinking', active);
      const finalIndex = final ? layout.indexOf(final) : layout.length;
      body._compactLayout = layout.slice(0, finalIndex);
      body._compactFinal = final;
      body._compactLatest = nodes.findLast(node => node.dataset.blockType === 'markdown') || null;
      placeTurnContent(body, turn);
    } else {
      restoreStatement(body);
      reconcileChildren(body, layout);
    }
    if (body._compactFocus?.isConnected && !body._compactFocus.closest('details:not([open])')) body._compactFocus.focus({preventScroll: true});
    body._compactFocus = null;
    body._compactSignature = signature;
    body._compactDirty = false;
    updateThinkingTime();
  }

  function refresh(ids = null) {
    const scroll = WAC.captureAutoscrollState();
    const transcript = WAC.transcript();
    const messages = ids ? Array.from(ids, id => transcript?.querySelector(`[data-message-id="${CSS.escape(id)}"]`)).filter(node => node?.matches('.chat__message--assistant')) : transcript?.querySelectorAll('.chat__message--assistant');
    WAC.compactScrollState = scroll;
    try { messages?.forEach(updateMessage); }
    finally { WAC.compactScrollState = null; }
    WAC.applyAutoscrollState(scroll);
  }
  function schedule(event, previousActive) {
    if (!event) return;
    if (event.type === 'sync' || event.type === 'reset') refreshAll = true;
    if (!['append_block_text', 'replace_block_text'].includes(event.type)) {
      const id = event.message_id || event.message?.id;
      if (id) dirtyMessages.add(id);
    }
    if (previousActive !== presentation.active_message_id) {
      if (previousActive) dirtyMessages.add(previousActive);
      if (presentation.active_message_id) dirtyMessages.add(presentation.active_message_id);
    }
    if (!scheduled && (refreshAll || dirtyMessages.size)) {
      scheduled = true;
      queueMicrotask(() => {
        scheduled = false;
        refresh(refreshAll ? null : dirtyMessages);
        dirtyMessages.clear(); refreshAll = false;
      });
    }
  }
  const finalize = WAC.finalizeBlock;
  WAC.finalizeBlock = event => {
    finalize(event);
    // Streaming can finalize after its transport event has already been presented.
    schedule(event, presentation.active_message_id);
  };
  WAC.setCompactActions = value => {
    compactByDefault = !!value;
    document.querySelectorAll('[data-deepy-compact-actions]').forEach(input => input.checked = compactByDefault);
    WAC.transcript()?.querySelectorAll('.chat__compact-turn').forEach(turn => {
      turn.open = !compactByDefault;
      turn.closest('.chat__body')._compactDirty = true;
    });
    refresh();
    syncClockTimer();
  };
  WAC.applyDisplaySettings = values => {
    if (compactByDefault !== values.compact_actions) WAC.setCompactActions(values.compact_actions);
    WAC.galleryMediaViewer = values.gallery_media_viewer;
  };
  const consume = WAC.consumePayload;
  WAC.consumePayload = function (payload) {
    const envelope = typeof payload === 'string' ? (() => {try {return JSON.parse(payload);} catch {return null;}})() : payload;
    const event = envelope?.event || envelope;
    const previousActive = presentation.active_message_id;
    const previousSession = WAC.chatSessionId;
    const result = consume.call(this, envelope);
    // The shared consumer has already passed every replay item through this wrapper.
    if (Array.isArray(envelope?.batch)) return result;
    if (event?.type === 'reset' || previousSession !== WAC.chatSessionId) {
      presentation = {active_message_id: '', active_user_id: '', durations: {}};
      clock = null;
      syncClockTimer();
    }
    const accepted = event && (event.revision === undefined || event.revision >= WAC.chatRevision) && (event.sequence === undefined || event.sequence >= WAC.chatSequence);
    if (accepted && event.presentation) {
      for (const [id, duration] of Object.entries(event.presentation.durations || {})) {
        if (presentation.durations[id] !== duration) dirtyMessages.add(id);
      }
      Object.assign(presentation.durations, event.presentation.durations);
      presentation.active_message_id = event.presentation.active_message_id;
      presentation.active_user_id = event.presentation.active_user_id;
      clock = Number.isFinite(event.presentation.elapsed_seconds) ? {elapsed: event.presentation.elapsed_seconds, paused: event.presentation.paused, received: performance.now()} : null;
      syncClockTimer();
    } else if (accepted && ['upsert_block', 'append_block_text', 'replace_block_text', 'finalize_block'].includes(event.type) && presentation.active_user_id) {
      presentation.active_message_id = event.message_id;
    }
    schedule(event, previousActive);
    return result;
  };
  const input = document.querySelector('[data-deepy-compact-actions]');
  if (input) {
    const status = input.parentElement.querySelector('[role=alert]');
    input.checked = compactByDefault;
    input.onchange = async () => {
      const previous = compactByDefault;
      status.textContent = '';
      WAC.setCompactActions(input.checked);
      input.disabled = true;
      try { await WAC.saveDisplaySettings({compact_actions: input.checked}); }
      catch (error) { WAC.setCompactActions(previous); status.textContent = error.message; }
      finally { input.disabled = false; }
    };
  }
})();
