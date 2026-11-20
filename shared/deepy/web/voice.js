// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
  const WAC = window.__wangpAssistantChatNS;
  const smartphone = navigator.userAgentData?.mobile || /iPhone|iPod|Android.*Mobile/i.test(navigator.userAgent);
  if (document.body.hasAttribute('data-deepy-app') && smartphone) return;
  if (WAC.voiceInstalled) return;
  WAC.voiceInstalled = true;
  let active = null, noticeTimer = null, mode = 'disabled', mountPending = false;
  const icon = '<svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg>';
  function status(button, text, recording = false) {
    button.title = text; button.setAttribute('aria-label', text); button.setAttribute('aria-pressed', String(recording));
    button.classList.toggle('is-recording', recording);
  }
  function release(recording) {
    clearTimeout(recording.timer);
    recording.stream?.getTracks().forEach(track => track.stop());
    recording.stream = null;
  }
  function finish(recording) {
    clearTimeout(recording.statusTimer);
    release(recording);
    // An error dialog stays open after active is cleared; Cancel must still close it.
    if (active === recording || !active) recording.dialog?.close();
    if (active !== recording) return;
    recording.button.disabled = false; recording.button.classList.remove('is-transcribing');
    status(recording.button, recording.chat ? 'Record a voice message' : 'Dictate prompt');
    recording.input.removeAttribute('aria-busy');
    recording.dialog?.close();
    active = null;
  }
  function cancel(recording) {
    recording.cancelled = true; recording.abort.abort();
    cancelPreparation(recording);
    if (recording.recorder?.state === 'recording') recording.recorder.stop();
    finish(recording);
    if (recording.chat) WAC.voiceNotice('');
  }
  function cancelPreparation(recording) {
    recording.prepareAbort?.abort();
    if (recording.id) fetch(`deepy_api/voice/${recording.id}/cancel`, {method: 'POST', keepalive: true}).catch(() => {});
  }
  function prepare(recording) {
    recording.id = crypto.randomUUID();
    recording.prepareAbort = new AbortController();
    recording.preparation = fetch(`deepy_api/voice/${recording.id}/prepare`, {method: 'POST', signal: recording.prepareAbort.signal})
      .then(async response => {
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Unable to prepare Whisper.');
        if (recording.cancelled) cancelPreparation(recording);
      }).catch(error => { recording.preparationError = error; });
  }
  async function trackTranscription(recording) {
    try {
      const response = await fetch(`deepy_api/voice/${recording.id}`, {signal: recording.abort.signal});
      if (!response.ok) return;
      const result = await response.json();
      if (active !== recording || recording.cancelled) return;
      if (result.status === 'waiting_gpu') message(recording, 'Whisper is waiting for the GPU…');
      else if (result.status === 'transcribing') message(recording, 'Transcribing…');
      recording.statusTimer = setTimeout(() => trackTranscription(recording), 500);
    } catch (_) {}
  }
  function message(recording, text) {
    if (recording.message === text) return;
    recording.message = text;
    if (recording.chat) WAC.voiceNotice(text, {persistent: true, onCancel: () => cancel(recording)});
    else recording.dialog.querySelector('[role=status]').textContent = text;
  }
  function promptDialog(recording) {
    let dialog = document.getElementById('wangp-voice-dialog');
    if (!dialog) {
      dialog = document.createElement('dialog'); dialog.id = 'wangp-voice-dialog'; dialog.setAttribute('aria-labelledby', 'wangp-voice-heading');
      dialog.innerHTML = '<h3 id="wangp-voice-heading">Dictate prompt</h3><p role="status">Starting microphone…</p><div><button type="button" data-voice-cancel>Cancel</button><button type="button" data-voice-stop>Stop</button></div>';
      document.body.append(dialog);
    }
    recording.dialog = dialog;
    dialog.querySelector('[role=status]').textContent = 'Starting microphone…';
    const stop = dialog.querySelector('[data-voice-stop]'); stop.disabled = true;
    stop.onclick = () => recording.recorder?.stop();
    dialog.querySelector('[data-voice-cancel]').onclick = () => cancel(recording);
    dialog.oncancel = event => { event.preventDefault(); cancel(recording); };
    dialog.showModal();
  }
  async function transcribe(recording) {
    release(recording);
    if (recording.cancelled) return;
    recording.button.disabled = true; recording.button.classList.add('is-transcribing');
    status(recording.button, 'Transcribing…'); recording.input.setAttribute('aria-busy', 'true');
    if (recording.dialog) recording.dialog.querySelector('[data-voice-stop]').disabled = true;
    message(recording, 'Preparing transcription…');
    let sent = false;
    try {
      const voiceStatus = await WAC.refreshVoiceConfig();
      if (recording.cancelled) return;
      if (voiceStatus.mode === 'disabled') throw new Error('Microphone transcription is disabled in Configuration.');
      if (voiceStatus.download_required) message(recording, 'Preparing transcription… downloading Whisper for first use.');
      await recording.preparation;
      if (recording.cancelled) return;
      if (recording.preparationError) throw recording.preparationError;
      recording.statusTimer = setTimeout(() => trackTranscription(recording), 500);
      const type = recording.recorder.mimeType;
      const suffix = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogg' : 'webm';
      const form = new FormData(); form.append('file', new Blob(recording.chunks, {type}), `voice.${suffix}`);
      const response = await fetch(`deepy_api/voice/${recording.id}/transcribe`, {method: 'POST', body: form, signal: recording.abort.signal});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Transcription failed.');
      if (recording.cancelled) return;
      if (!recording.input.isConnected) throw new Error('The prompt form changed. Please record again in the current prompt.');
      const text = [recording.input.value.trim(), result.text.trim()].filter(Boolean).join('\n');
      if (recording.chat) { WAC.setRequestInputValue(text); WAC.scheduleComposerLayout(); }
      else {
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(recording.input, text);
        recording.input.dispatchEvent(new Event('input', {bubbles: true}));
      }
      sent = recording.submit && !!text;
      if (recording.chat) WAC.voiceNotice(result.text.trim() ? '' : 'No speech detected.');
    } catch (error) {
      if (!recording.cancelled) {
        if (recording.chat) WAC.voiceNotice(error.message);
        else { message(recording, error.message); recording.keepDialog = true; }
      }
    } finally {
      cancelPreparation(recording);
      const dialog = recording.dialog;
      finish(recording);
      if (recording.keepDialog) dialog.showModal();
      if (sent) document.querySelector('#assistant_chat_ask_button button, #assistant_chat_ask_button')?.click();
    }
  }
  async function toggle(button, input, chat) {
    if (active) {
      if (active.button === button && active.recorder?.state === 'recording') active.recorder.stop();
      return;
    }
    const recording = {button, input, chat, chunks: [], abort: new AbortController()}; active = recording;
    if (!chat) promptDialog(recording);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access is unavailable in this browser.');
      if (!window.MediaRecorder) throw new Error('Audio recording is unavailable in this browser.');
      recording.stream = await navigator.mediaDevices.getUserMedia({audio: true});
      if (recording.cancelled) { release(recording); return; }
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus'].find(type => MediaRecorder.isTypeSupported(type));
      recording.recorder = mimeType ? new MediaRecorder(recording.stream, {mimeType}) : new MediaRecorder(recording.stream);
      recording.recorder.ondataavailable = event => { if (event.data.size) recording.chunks.push(event.data); };
      recording.recorder.onerror = () => { message(recording, 'Audio recording failed.'); cancel(recording); };
      recording.recorder.onstop = () => transcribe(recording);
      recording.recorder.start(); status(button, 'Stop recording and transcribe', true);
      prepare(recording);
      message(recording, chat ? 'Recording… tap the microphone or Ask to finish.' : 'Recording…');
      if (recording.dialog) recording.dialog.querySelector('[data-voice-stop]').disabled = false;
      recording.timer = setTimeout(() => { if (recording.recorder.state === 'recording') recording.recorder.stop(); }, 180000);
      if (recording.submit) recording.recorder.stop();
    } catch (error) {
      if (!recording.cancelled) { message(recording, error.message); const dialog = recording.dialog; finish(recording); if (dialog) dialog.showModal(); }
    }
  }
  WAC.finishVoiceBeforeSubmit = () => {
    if (!active?.chat) return false;
    active.submit = true;
    if (active.recorder?.state === 'recording') active.recorder.stop();
    return true;
  };
  WAC.voiceNotice = (text, {persistent = false, httpsUrl = null, onCancel = null} = {}) => {
    const request = document.getElementById('assistant_chat_request');
    if (!request) return;
    let notice = document.getElementById('assistant_chat_voice_notice');
    if (!notice) {
      notice = document.createElement('div'); notice.id = 'assistant_chat_voice_notice'; notice.setAttribute('role', 'status');
      const controls = document.getElementById('assistant_chat_controls');
      if (document.body.hasAttribute('data-deepy-app')) controls.append(notice);
      else controls.after(notice);
    }
    notice.textContent = text;
    notice.classList.toggle('has-cancel', !!onCancel);
    notice.hidden = !text;
    clearTimeout(noticeTimer);
    if (text) {
      if (httpsUrl) { const link = document.createElement('a'); link.href = httpsUrl; link.textContent = 'Open HTTPS'; notice.append(link); }
      const close = document.createElement('button'); close.type = 'button'; close.textContent = onCancel ? 'Cancel' : '\u00d7'; close.setAttribute('aria-label', onCancel ? 'Cancel voice transcription' : 'Dismiss microphone notice'); close.onclick = onCancel || (() => WAC.voiceNotice('')); notice.append(close);
      if (!persistent) noticeTimer = setTimeout(() => WAC.voiceNotice(''), 6000);
    }
  };
  function mount() {
    mountPending = false;
    if (mode === 'disabled') {
      document.querySelectorAll('#assistant_chat_microphone, .wangp-prompt-microphone').forEach(button => {button.hidden = true;});
      return;
    }
    const ask = document.getElementById('assistant_chat_ask_button');
    if (ask && !document.getElementById('assistant_chat_microphone')) {
      const button = document.createElement('button'); button.type = 'button'; button.id = 'assistant_chat_microphone'; button.innerHTML = icon;
      status(button, 'Record a voice message'); button.onclick = () => toggle(button, WAC.requestInput(), true); ask.before(button);
    }
    document.querySelectorAll('.wangp-voice-prompt').forEach(host => {
      const input = host.querySelector('textarea'); if (!input) return;
      const label = host.querySelector('[data-testid="block-info"]'); if (!label) return;
      let button = host.querySelector('.wangp-prompt-microphone');
      if (!button) {
        button = document.createElement('button'); button.type = 'button'; button.className = 'wangp-prompt-microphone'; button.innerHTML = icon;
        status(button, 'Dictate prompt'); button.onclick = () => toggle(button, input, false);
      }
      const help = label.querySelector('.wangp-context-tool-info, .wangp-field-help-inline');
      if (help) { if (help.nextElementSibling !== button) help.after(button); }
      else if (button.parentElement !== label) label.append(button);
    });
    document.querySelectorAll('#assistant_chat_microphone, .wangp-prompt-microphone').forEach(button => {button.hidden = false;});
  }
  WAC.refreshVoiceConfig = async () => {
    const response = await fetch('deepy_api/voice');
    if (!response.ok) throw new Error('Unable to check microphone configuration.');
    const config = await response.json(); setMode(config.mode); return config;
  };
  const mountSelector = '.wangp-voice-prompt, #assistant_chat_ask_button';
  const observer = new MutationObserver(mutations => {
    if (mountPending || !mutations.some(mutation => mutation.target.closest?.(mountSelector) ||
        Array.from(mutation.addedNodes).some(node => node.nodeType === 1 && (node.matches(mountSelector) || node.querySelector(mountSelector))))) return;
    mountPending = true;
    requestAnimationFrame(mount);
  });
  function setMode(value) {
    if (value === mode) return;
    mode = value;
    // The standalone Web composer is static. Gradio can remount prompt
    // fields and their labels, but only enabled microphones need reattaching.
    if (mode === 'disabled') observer.disconnect();
    else if (!document.body.hasAttribute('data-deepy-app')) observer.observe(document.querySelector('gradio-app') || document.body, {childList: true, subtree: true});
    mount();
  }
  WAC.setVoiceMode = value => {
    if (value === 'disabled' && active) cancel(active);
    setMode(value);
  };
  window.addEventListener('focus', () => WAC.refreshVoiceConfig().catch(() => {}));
  window.addEventListener('pagehide', () => { if (active) cancel(active); });
  WAC.refreshVoiceConfig().catch(() => {});
})();
