// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* WanGP app runtime, injected into every app page ahead of its modules (shared/apps/server.py). An app runs with an opaque origin, where the
   browser's own storage throws, so localStorage and sessionStorage are replaced: localStorage keeps the app's saved data on WanGP's side (high
   scores, saves, settings). Console lines, uncaught errors, failed loads, the frame rate and the WebGL renderer are reported to WanGP (the
   player's console and Deepy's app_logs). The player pauses and resumes the app (animation frames, audio, media); audio contexts resume on the
   first click or key press, as browsers require. window.wangp exposes the saved data; apps may define window.wangpState() for checks. */
(() => {
  'use strict';
  // Own scheduled callbacks, not an extra loop checking whether the app is visible.
  // IDs and remaining timer delays survive suspension, as does the app's clock.
  // This function is also installed before code in dedicated workers.
  function installScheduler(scope, initiallyPaused, baseURL) {
    const realNow = scope.performance.now.bind(scope.performance), realDate = scope.Date.now.bind(scope.Date);
    const native = {};
    for (const name of ['setTimeout', 'clearTimeout', 'requestAnimationFrame', 'cancelAnimationFrame', 'requestIdleCallback', 'cancelIdleCallback']) {
      if (scope[name]) native[name] = scope[name].bind(scope);
    }
    let paused = initiallyPaused, stoppedAt = realNow(), elapsed = 0, nextId = 0;
    const jobs = new Map(), workers = new Set();
    const now = () => (paused ? stoppedAt : realNow()) - elapsed;
    Object.defineProperty(scope.performance, 'now', {value: now, configurable: true});
    scope.Date.now = () => Math.round(realDate() - elapsed - (paused ? realNow() - stoppedAt : 0));
    const cancelNative = job => {
      if (job.native == null) return;
      native[job.kind === 'frame' ? 'cancelAnimationFrame' : job.kind === 'idle' ? 'cancelIdleCallback' : 'clearTimeout'](job.native);
      job.native = null;
    };
    const arm = job => {
      if (paused) return;
      const run = argument => {
        job.native = null;
        if (paused || !jobs.has(job.id)) return;
        if (job.kind !== 'interval') jobs.delete(job.id);
        try {
          if (job.kind === 'frame') { const time = argument - elapsed; scheduler.onFrame?.(time); job.callback.call(scope, time); }
          else if (job.kind === 'idle') job.callback.call(scope, argument);
          else job.callback.apply(scope, job.args);
        } finally {
          if (job.kind === 'interval' && jobs.has(job.id)) { job.remaining = job.delay; arm(job); }
        }
      };
      job.due = now() + job.remaining;
      job.native = job.kind === 'frame' ? native.requestAnimationFrame(run)
        : job.kind === 'idle' ? native.requestIdleCallback(run, job.options)
        : native.setTimeout(run, job.remaining);
    };
    const add = (kind, callback, delay = 0, args = [], options) => {
      if (typeof callback !== 'function') callback = new Function(String(callback));
      delay = Math.max(0, Number(delay) || 0);
      const job = {id: ++nextId, kind, callback, delay, remaining: delay, args, options, native: null};
      jobs.set(job.id, job); arm(job); return job.id;
    };
    const cancel = id => { const job = jobs.get(Number(id)); if (job) { cancelNative(job); jobs.delete(job.id); } };
    scope.setTimeout = (callback, delay, ...args) => add('timer', callback, delay, args);
    scope.setInterval = (callback, delay, ...args) => add('interval', callback, delay, args);
    scope.clearTimeout = scope.clearInterval = cancel;
    if (native.requestAnimationFrame) {
      scope.requestAnimationFrame = callback => add('frame', callback);
      scope.cancelAnimationFrame = cancel;
    }
    if (native.requestIdleCallback) {
      scope.requestIdleCallback = (callback, options) => add('idle', callback, 0, [], options);
      scope.cancelIdleCallback = cancel;
    }
    const scheduler = {now, get paused() { return paused; }, setPaused(value) {
      if (paused === value) return;
      if (value) {
        stoppedAt = realNow(); paused = true;
        for (const job of jobs.values()) {
          job.remaining = Math.max(0, job.due - now()); cancelNative(job);
        }
      } else {
        elapsed += realNow() - stoppedAt; paused = false;
        for (const job of jobs.values()) arm(job);
      }
      for (const worker of workers) worker.suspend(value);
    }};
    if (scope.Worker) {
      const NativeWorker = scope.Worker, createURL = URL.createObjectURL.bind(URL), revokeURL = URL.revokeObjectURL.bind(URL), blobs = new Map();
      // Libraries commonly revoke their worker blob immediately after construction.
      URL.createObjectURL = blob => { const url = createURL(blob); blobs.set(url, blob); return url; };
      URL.revokeObjectURL = url => { blobs.delete(String(url)); revokeURL(url); };
      scope.Worker = class extends NativeWorker {
        constructor(url, options = {}) {
          const original = new URL(url, baseURL).href, retained = blobs.has(original) ? createURL(blobs.get(original)) : null;
          const target = retained || original, key = 'wangp-worker-' + Math.random().toString(36).slice(2);
          const source = `(()=>{const scheduler=(${installScheduler.toString()})(self,${paused},${JSON.stringify(target)});
            const pending=[];let loading=true;
            addEventListener('message',event=>{
              if(event.data?.[${JSON.stringify(key)}]!==undefined){
                event.stopImmediatePropagation();scheduler.setPaused(event.data[${JSON.stringify(key)}]);
                if(!scheduler.paused&&!loading)drain();
              }else if(scheduler.paused||loading){event.stopImmediatePropagation();pending.push(event);}
            },true);
            function drain(){for(const e of pending.splice(0))dispatchEvent(new MessageEvent('message',{data:e.data,ports:e.ports}));}
            function ready(){loading=false;postMessage({[${JSON.stringify(key)}]:'ready'});if(!scheduler.paused)drain();}
            ${options.type === 'module' ? `import(${JSON.stringify(target)}).then(ready);` : `importScripts(${JSON.stringify(target)});ready();`}})();`;
          const bootstrap = createURL(new Blob([source], {type: 'text/javascript'}));
          // A classic blob bootstrap also works in the app's opaque origin.
          // Module entry points still execute as modules, through import().
          super(bootstrap, {...options, type: 'classic'});
          const pending = [], worker = this;
          const release = () => { revokeURL(bootstrap); if (retained) revokeURL(retained); };
          const state = {suspend(value) {
            NativeWorker.prototype.postMessage.call(worker, {[key]: value});
            if (!value) for (const event of pending.splice(0)) worker.dispatchEvent(new MessageEvent('message', {data: event.data, ports: event.ports}));
          }};
          workers.add(state);
          this.addEventListener('message', event => {
            if (event.data?.[key] === 'ready') { event.stopImmediatePropagation(); release(); }
            else if (paused) { event.stopImmediatePropagation(); pending.push(event); }
          });
          this.addEventListener('error', release, {once: true});
          this.terminate = () => { workers.delete(state); pending.length = 0; release(); NativeWorker.prototype.terminate.call(this); };
        }
      };
    }
    return scheduler;
  }
  // Managed frames wait for the player's initial lifecycle message. A standalone
  // app still starts normally; a hidden tab starts suspended.
  const scheduler = installScheduler(window, document.hidden || window.name === 'wangp-managed-app', location.href);
  const config = window.__wangpApp || {store: {}, persist: false};
  const base = (location.pathname.match(/^\/apps\/r\/[^/]+\//) || ['/'])[0];
  const api = base + '__wangp__/';
  const send = (path, payload) => fetch(api + path, {method: 'POST', body: JSON.stringify(payload), keepalive: true}).catch(() => {});  // text/plain: no preflight
  const parentWindow = window.parent;
  const notify = message => { try { parentWindow.postMessage({wangp: message.type, ...message}, '*'); } catch (error) { /* no player */ } };

  // ---- saved data
  const data = new Map(Object.entries(config.store || {}));
  let change = {set: {}, remove: []}, flushTimer = 0;
  const flush = () => {
    clearTimeout(flushTimer); flushTimer = 0;
    if (!config.persist || (!Object.keys(change.set).length && !change.remove.length)) return;
    const payload = change; change = {set: {}, remove: []};
    send('store', payload);
  };
  const methods = (map, persist) => ({
    get length() { return map.size; },
    key: index => [...map.keys()][index] ?? null,
    getItem: key => map.has(String(key)) ? map.get(String(key)) : null,
    setItem: (key, value) => {
      key = String(key); value = String(value); map.set(key, value);
      if (persist) { change.set[key] = value; change.remove = change.remove.filter(item => item !== key); flushTimer = flushTimer || setTimeout(flush, 400); }
    },
    removeItem: key => {
      key = String(key); map.delete(key);
      if (persist) { delete change.set[key]; change.remove.push(key); flushTimer = flushTimer || setTimeout(flush, 400); }
    },
    clear: () => {
      if (persist) { change = {set: {}, remove: [...map.keys()]}; flushTimer = flushTimer || setTimeout(flush, 400); }
      map.clear();
    },
  });
  const storage = (map, persist) => {
    const target = methods(map, persist);
    return new Proxy(target, {  // localStorage.best = 3 and localStorage['best'] work as on a real Storage
      get: (object, key) => key in object ? object[key] : typeof key === 'string' && map.has(key) ? map.get(key) : undefined,
      set: (object, key, value) => { object.setItem(key, value); return true; },
      deleteProperty: (object, key) => { object.removeItem(key); return true; },
      has: (object, key) => key in object || map.has(key),
      ownKeys: () => [...map.keys()],
      getOwnPropertyDescriptor: (object, key) => map.has(key) ? {value: map.get(key), enumerable: true, configurable: true, writable: true} : undefined,
    });
  };
  for (const [name, value] of [['localStorage', storage(data, true)], ['sessionStorage', storage(new Map(), false)]]) {
    try { Object.defineProperty(window, name, {value, configurable: true}); } catch (error) { /* left as the browser has it */ }
  }
  addEventListener('pagehide', flush);

  // ---- reports
  const queue = [], stats = {};
  let reportTimer = 0;
  const report = () => {
    reportTimer = 0;
    if (!queue.length && !stats.dirty) return;
    const payload = {entries: queue.splice(0, 100)};
    if (stats.dirty) { payload.stats = {fps: stats.fps, renderer: stats.renderer, size: [innerWidth, innerHeight]}; stats.dirty = false; }
    send('log', payload);
    if (queue.length) reportTimer = setTimeout(report, 1000);
  };
  const format = value => {
    if (value instanceof Error) return value.stack || String(value);
    if (value && typeof value === 'object') { try { return JSON.stringify(value).slice(0, 500); } catch (error) { return String(value); } }
    return String(value);
  };
  const record = (level, values) => {
    const text = values.map(format).join(' ').slice(0, 2000);
    queue.push({level, text});
    if (queue.length > 300) queue.splice(0, queue.length - 300);
    reportTimer = reportTimer || setTimeout(report, 1000);
    notify({type: 'log', level, text});
  };
  for (const level of ['log', 'info', 'warn', 'error']) {
    const original = console[level].bind(console);
    console[level] = (...values) => { original(...values); record(level, values); };
  }
  addEventListener('error', event => {
    const target = event.target, attribute = target && target !== window && (target.src ? 'src' : target.href ? 'href' : '');
    if (!attribute) return;
    const element = `<${target.tagName.toLowerCase()}${target.id ? ` id="${target.id}"` : ''}>`;
    record('error', [target.getAttribute(attribute) === '' ? `${element} has an empty ${attribute}="", which loads the page itself: leave ${attribute} out until there is a file to show` : `Failed to load ${target[attribute]} (${element})`]);
  }, true);
  addEventListener('error', event => record('exception', [event.error || `${event.message} (${String(event.filename || '').split('/').pop()}:${event.lineno})`]));
  addEventListener('unhandledrejection', event => record('exception', ['Unhandled promise rejection:', event.reason]));

  // Measure frames the app actually requests; static apps need no reporting loop.
  let paused = scheduler.paused, frames = 0, since = performance.now(), lastFrame = -1;
  scheduler.onFrame = now => {
    if (now === lastFrame) return;
    lastFrame = now;
    frames += 1;
    if (now - since >= 2000) {
      stats.fps = Math.round(frames * 10000 / (now - since)) / 10; stats.dirty = true; frames = 0; since = now;
      notify({type: 'fps', fps: stats.fps});
      reportTimer = reportTimer || setTimeout(report, 1000);
    }
  };
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
    const context = getContext.call(this, type, ...rest);
    if (context && /webgl/i.test(String(type)) && !this.__wangpWatched) {
      this.__wangpWatched = true;
      this.addEventListener('webglcontextlost', () => record('error', ['WebGL context lost']));
      if (!stats.renderer) {
        try {
          const debug = context.getExtension('WEBGL_debug_renderer_info');
          stats.renderer = String(debug ? context.getParameter(debug.UNMASKED_RENDERER_WEBGL) : context.getParameter(context.RENDERER)); stats.dirty = true;
        } catch (error) { /* hidden by the browser */ }
      }
    }
    return context;
  };

  // ---- audio: resumed on the first gesture, suspended while paused
  const contexts = new Set(), stoppedContexts = new Set(), stoppedMedia = new Set(), stoppedAnimations = new Set();
  for (const name of ['AudioContext', 'webkitAudioContext']) {
    const Native = window[name];
    if (typeof Native !== 'function') continue;
    window[name] = class extends Native {
      constructor(...args) {
        super(...args); contexts.add(this);
        this.addEventListener('statechange', () => {
          if (paused && this.state === 'running') { stoppedContexts.add(this); this.suspend().catch(() => {}); }
          if (this.state === 'closed') { contexts.delete(this); stoppedContexts.delete(this); }
        });
        if (paused) { stoppedContexts.add(this); this.suspend().catch(() => {}); }
      }
      resume() { if (paused) { stoppedContexts.add(this); return Promise.resolve(); } return super.resume(); }
    };
  }
  const wake = () => { if (!paused) for (const context of contexts) if (context.state === 'suspended') context.resume().catch(() => {}); };
  addEventListener('pointerdown', wake, true);
  addEventListener('keydown', wake, true);
  addEventListener('pointerdown', () => window.focus(), true);

  // Pause CSS and Web Animations without observing DOM changes. The stylesheet
  // also covers animations added by an in-flight resource completion while hidden.
  const pausedStyle = document.createElement('style');
  pausedStyle.textContent = 'html[data-wangp-paused] *, html[data-wangp-paused] *::before, html[data-wangp-paused] *::after { animation-play-state: paused !important; }';
  document.head.append(pausedStyle);
  document.addEventListener('play', event => {
    if (paused && event.target instanceof HTMLMediaElement) { stoppedMedia.add(event.target); event.target.pause(); }
  }, true);
  const animate = Element.prototype.animate;
  Element.prototype.animate = function (...args) {
    const animation = animate.apply(this, args);
    if (paused) { animation.pause(); stoppedAnimations.add(animation); }
    return animation;
  };

  const pause = () => {
    if (paused) return;
    paused = true;
    scheduler.setPaused(true);
    for (const context of contexts) if (context.state === 'running') { stoppedContexts.add(context); context.suspend().catch(() => {}); }
    for (const media of document.querySelectorAll('audio, video')) if (!media.paused) { media.pause(); stoppedMedia.add(media); }
    for (const animation of document.getAnimations()) if (animation.playState === 'running') { animation.pause(); stoppedAnimations.add(animation); }
    document.documentElement.dataset.wangpPaused = '';
    dispatchEvent(new CustomEvent('wangp-pause'));
  };
  const resume = () => {
    if (!paused) return;
    paused = false; scheduler.setPaused(false); since = performance.now(); frames = 0;
    delete document.documentElement.dataset.wangpPaused;
    for (const animation of stoppedAnimations) if (animation.playState === 'paused') animation.play();
    stoppedAnimations.clear();
    for (const context of stoppedContexts) if (context.state === 'suspended') context.resume().catch(() => {});
    stoppedContexts.clear();
    for (const media of stoppedMedia) media.play().catch(() => {});
    stoppedMedia.clear();
    dispatchEvent(new CustomEvent('wangp-resume'));
  };
  let playerPaused = paused;
  const applyPause = () => playerPaused || document.hidden ? pause() : resume();
  addEventListener('message', event => {
    if (event.source !== parentWindow || !event.data || typeof event.data.wangp !== 'string') return;
    if (event.data.wangp === 'pause' || event.data.wangp === 'resume') { playerPaused = event.data.wangp === 'pause'; applyPause(); }
    else if (event.data.wangp === 'state') {
      let state = null;
      try { state = typeof window.wangpState === 'function' ? JSON.parse(JSON.stringify(window.wangpState())) : null; } catch (error) { state = {error: String(error)}; }
      notify({type: 'state', state});
    }
  });
  document.addEventListener('visibilitychange', applyPause);
  addEventListener('pagehide', pause);
  addEventListener('pageshow', applyPause);
  if (paused) document.documentElement.dataset.wangpPaused = '';

  window.wangp = Object.freeze({
    name: config.name,
    store: Object.freeze({
      get: key => { const value = data.get(String(key)); if (value === undefined) return null; try { return JSON.parse(value); } catch (error) { return value; } },
      set: (key, value) => window.localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value)),
      remove: key => window.localStorage.removeItem(key),
      keys: () => [...data.keys()],
    }),
    paused: () => paused,
  });
  notify({type: 'ready'});
  addEventListener('load', () => notify({type: 'ready'}));
})();
