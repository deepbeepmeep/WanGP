# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Runtime monkey patch that keeps WanGP's main tab responsive across tab switches and backgrounding."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH = True
GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE = False
FOCUS_QUEUE_SERVER_CONFIG_KEY = "process_queues_when_browser_unfocused"
BACKGROUND_SCHEDULER_DEFAULT_ENABLED = True
OFFTAB_KEEPALIVE_DEFAULT_ENABLED = True

_PATCH_SENTINEL = "window.__gradioFocusQueuePatch"
_TARGET_TEMPLATES = {"frontend/index.html", "frontend/share.html"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_javascript() -> str:
    if not ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH:
        return ""
    verbose = "true" if GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE else "false"
    background_default_enabled = "true" if BACKGROUND_SCHEDULER_DEFAULT_ENABLED else "false"
    offtab_default_enabled = "true" if OFFTAB_KEEPALIVE_DEFAULT_ENABLED else "false"
    return f"""
(function () {{
  if (typeof window === "undefined" || window.__gradioFocusQueuePatch) {{
    return;
  }}

  const MAIN_TAB_ID = "media_gen";
  const OFFTAB_DELAY_MS = 25;
  const INTERACTION_KEEPALIVE_MS = 1500;
  const KEEPALIVE_CLASS = "wangp-main-tab-keepalive";
  const KEEPALIVE_STYLE_ID = "wangp-main-tab-keepalive-style";
  const nativeRequestAnimationFrame = window.requestAnimationFrame.bind(window);
  const nativeCancelAnimationFrame = window.cancelAnimationFrame.bind(window);
  const nativeAddEventListener = EventTarget.prototype.addEventListener;
  const nativeRemoveEventListener = EventTarget.prototype.removeEventListener;
  const channel = typeof MessageChannel === "function" ? new MessageChannel() : null;
  const wrappedListeners = new WeakMap();
  const wrappedHandleEventListeners = new WeakMap();
  const syntheticJobs = new Map();
  const backgroundQueue = [];
  let nextSyntheticId = 1;
  let activeOfftabContext = null;

  function isElementVisible(element) {{
    if (!element) {{
      return false;
    }}
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }}

  function getMainTab() {{
    if (patch.mainTab && patch.mainTab.isConnected) {{
      return patch.mainTab;
    }}
    patch.mainTab = document.querySelector(`[role="tab"][data-tab-id="${{MAIN_TAB_ID}}"]`);
    return patch.mainTab;
  }}

  function getMainTabPanel() {{
    const tab = getMainTab();
    const panelId = tab ? tab.getAttribute("aria-controls") : null;
    if (!panelId) {{
      patch.mainTabPanel = null;
      patch.mainTabPanelId = null;
      return null;
    }}
    if (patch.mainTabPanel && patch.mainTabPanelId === panelId && patch.mainTabPanel.isConnected) {{
      return patch.mainTabPanel;
    }}
    patch.mainTabPanelId = panelId;
    patch.mainTabPanel = document.getElementById(panelId);
    return patch.mainTabPanel;
  }}

  function isMainTabSelected() {{
    const tab = getMainTab();
    return !!tab && tab.getAttribute("aria-selected") === "true";
  }}

  function isMainTabActive() {{
    return isElementVisible(getMainTabPanel());
  }}

  function ensureKeepAliveStyle() {{
    if (document.getElementById(KEEPALIVE_STYLE_ID)) {{
      return;
    }}
    const style = document.createElement("style");
    style.id = KEEPALIVE_STYLE_ID;
    style.textContent = `
      .${{KEEPALIVE_CLASS}} {{
        display: flex !important;
        position: absolute !important;
        left: -200vw !important;
        top: 0 !important;
        width: 1px !important;
        height: 1px !important;
        min-height: 0 !important;
        max-height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        z-index: -1 !important;
        contain: layout style paint !important;
      }}
    `;
    document.head.appendChild(style);
  }}

  function syncMainTabKeepAlive() {{
    const panel = getMainTabPanel();
    if (!panel) {{
      return false;
    }}
    panel.classList.toggle(KEEPALIVE_CLASS, patch.shouldKeepMainTabAlive());
    patch.mainTabKeepAliveActive = panel.classList.contains(KEEPALIVE_CLASS);
    return patch.mainTabKeepAliveActive;
  }}

  function primeMainTabKeepAlive() {{
    const panel = getMainTabPanel();
    if (!panel) {{
      return false;
    }}
    panel.classList.add(KEEPALIVE_CLASS);
    patch.mainTabKeepAliveActive = true;
    return true;
  }}

  function primeMainTabInteraction() {{
    patch.interactionKeepAliveUntil = Date.now() + INTERACTION_KEEPALIVE_MS;
  }}

  function isSameTopLevelTabList(tab) {{
    const mainTab = getMainTab();
    if (!tab || !mainTab || tab === mainTab) {{
      return false;
    }}
    return tab.closest('[role="tablist"]') === mainTab.closest('[role="tablist"]');
  }}

  function installTopLevelTabPrime() {{
    if (patch.topLevelTabPrimeInstalled) {{
      return;
    }}
    patch.topLevelTabPrimeInstalled = true;
    const handler = (event) => {{
      const tab = event && event.target && event.target.closest ? event.target.closest('[role="tab"]') : null;
      if (!patch.enableOfftabKeepAlive || !isMainTabSelected() || !isSameTopLevelTabList(tab)) {{
        return;
      }}
      primeMainTabKeepAlive();
    }};
    document.addEventListener("pointerdown", handler, true);
    document.addEventListener("click", handler, true);
  }}

  function installMainTabInteractionPrime() {{
    if (patch.mainTabInteractionPrimeInstalled) {{
      return;
    }}
    patch.mainTabInteractionPrimeInstalled = true;
    const handler = (event) => {{
      if (!patch.enableOfftabKeepAlive) {{
        return;
      }}
      const panel = getMainTabPanel();
      const target = event && event.target ? event.target : null;
      if (!panel || !target || !panel.contains(target)) {{
        return;
      }}
      primeMainTabInteraction();
    }};
    document.addEventListener("pointerdown", handler, true);
    document.addEventListener("click", handler, true);
  }}

  function scheduleMainTabKeepAliveSync() {{
    if (patch.keepAliveSyncQueued) {{
      return;
    }}
    patch.keepAliveSyncQueued = true;
    Promise.resolve().then(() => {{
      patch.keepAliveSyncQueued = false;
      patch.mainTabKeepAliveActive = syncMainTabKeepAlive();
    }});
  }}

  function installMainTabKeepAliveObserver() {{
    if (patch.keepAliveObserverInstalled) {{
      return;
    }}
    patch.keepAliveObserverInstalled = true;
    let observedTab = null, observedPanel = null;
    function invalidateComponentIds() {{
      patch.targetComponentIds = null;
      patch.targetComponentIdsPanelId = null;
    }}
    function observeMainTab() {{
      const tab = getMainTab(), panel = getMainTabPanel();
      if (tab === observedTab && panel === observedPanel) return false;
      observer.disconnect();
      observedTab = tab;
      observedPanel = panel;
      invalidateComponentIds();
      if (tab && panel) {{
        observer.observe(tab, {{attributes: true, attributeFilter: ["aria-selected", "aria-controls"]}});
        observer.observe(panel, {{childList: true, subtree: true}});
        // Detect replacement of the tab or panel without watching other panels.
        observer.observe(tab.parentElement, {{childList: true}});
        observer.observe(panel.parentElement, {{childList: true}});
        scheduleMainTabKeepAliveSync();
      }} else {{
        observer.observe(document.body || document.documentElement, {{childList: true, subtree: true}});
      }}
      return true;
    }}
    const hasComponent = node => node.nodeType === 1 && (node.matches('[id^="component-"]') || node.querySelector('[id^="component-"]'));
    const observer = new MutationObserver(records => {{
      if (observeMainTab() || !observedPanel) return;
      if (records.some(record => record.type === "attributes" && record.target === observedTab)) scheduleMainTabKeepAliveSync();
      if (patch.targetComponentIds !== null && records.some(record => record.type === "childList" && observedPanel.contains(record.target) &&
          [...record.addedNodes, ...record.removedNodes].some(hasComponent))) invalidateComponentIds();
    }});
    patch.keepAliveObserver = observer;
    // The bootstrap watch ends as soon as Gradio mounts the main tab.
    observer.observe(document.body || document.documentElement, {{childList: true, subtree: true}});
    observeMainTab();
  }}

  function getTargetComponentIds() {{
    if (patch.targetComponentIds !== null && patch.targetComponentIdsPanelId === patch.getMainTabPanelId()) {{
      return patch.targetComponentIds;
    }}
    const panel = getMainTabPanel();
    if (!panel) {{
      return null;
    }}
    const componentNodes = [panel, ...panel.querySelectorAll('[id^="component-"]')];
    const componentIds = new Set();
    for (const node of componentNodes) {{
      const match = /^component-(\\d+)$/.exec(String(node && node.id || ""));
      if (match) {{
        componentIds.add(Number(match[1]));
      }}
    }}
    patch.targetComponentIds = componentIds;
    patch.targetComponentIdsPanelId = patch.getMainTabPanelId();
    return patch.targetComponentIds;
  }}

  function isTargetGradioEvent(detail) {{
    if (!detail || !Number.isInteger(detail.id)) {{
      return false;
    }}
    const targetComponentIds = getTargetComponentIds();
    return !!targetComponentIds && targetComponentIds.has(detail.id);
  }}

  function shouldPatchBackgroundAnimationFrame(stack, callback) {{
    if (!patch.shouldPatchBackground()) {{
      return false;
    }}
    const stackText = String(stack || "");
    const source = String(callback || "");
    const isBlocksDispatch = stackText.includes("/assets/Blocks-") && source.includes("Jt(");
    const isCoreFlush = stackText.includes("/assets/index-") && source.includes("l.update(") && source.includes("ge.length") && source.includes("f.props[v.prop]=j");
    return isBlocksDispatch || isCoreFlush;
  }}

  function scheduleBackgroundAnimationFrame(callback) {{
    if (!channel) {{
      return nativeRequestAnimationFrame(callback);
    }}
    const syntheticId = -nextSyntheticId++;
    const job = {{ id: syntheticId, kind: "background", callback, canceled: false }};
    syntheticJobs.set(syntheticId, job);
    backgroundQueue.push(job);
    channel.port2.postMessage(syntheticId);
    patch.backgroundDispatchCount += 1;
    patch.lastBackgroundDispatch = {{ timestamp: Date.now() }};
    if (patch.verbose) {{
      console.debug("[Gradio] background synthetic animation frame", syntheticId);
    }}
    return syntheticId;
  }}

  function dispatchBackgroundJob(job) {{
    if (!job || job.canceled) {{
      return;
    }}
    syntheticJobs.delete(job.id);
    try {{
      job.callback(window.performance.now());
    }} catch (error) {{
      window.setTimeout(() => {{
        throw error;
      }}, 0);
    }}
  }}

  function shouldUseOfftabScheduler(detail) {{
    return patch.shouldPatchOfftab() && isTargetGradioEvent(detail);
  }}

  function shouldPatchOfftabDeferredCallback(callback) {{
    if (!patch.shouldPatchOfftab() || Date.now() > patch.syntheticWindowUntil) {{
      return false;
    }}
    const source = String(callback || "");
    const isBlocksDispatch = source.includes("Jt(");
    const isCoreFlush = source.includes("l.update(") && source.includes("ge.length") && source.includes("f.props[v.prop]=j");
    return isBlocksDispatch || isCoreFlush;
  }}

  function scheduleOfftabAnimationFrame(callback, context) {{
    const syntheticId = -nextSyntheticId++;
    const timeoutId = window.setTimeout(() => {{
      syntheticJobs.delete(syntheticId);
      try {{
        callback(window.performance.now());
      }} catch (error) {{
        window.setTimeout(() => {{
          throw error;
        }}, 0);
      }}
    }}, context.delay);
    syntheticJobs.set(syntheticId, {{ kind: "offtab", timeoutId }});
    patch.offtabDispatchCount += 1;
    patch.lastOfftabDispatch = {{
      componentId: context.componentId,
      eventName: context.eventName,
      delay: context.delay,
      timestamp: Date.now(),
    }};
    return syntheticId;
  }}

  function cancelSyntheticAnimationFrame(syntheticId) {{
    const job = syntheticJobs.get(syntheticId);
    if (typeof job === "undefined") {{
      return false;
    }}
    syntheticJobs.delete(syntheticId);
    if (job.kind === "offtab") {{
      window.clearTimeout(job.timeoutId);
    }} else {{
      job.canceled = true;
    }}
    return true;
  }}

  function withOfftabScheduler(detail, invoke) {{
    const previousContext = activeOfftabContext;
    const context = {{
      componentId: detail && Number.isInteger(detail.id) ? detail.id : null,
      eventName: detail && typeof detail.event === "string" ? detail.event : "",
      delay: OFFTAB_DELAY_MS,
    }};
    patch.syntheticWindowUntil = Date.now() + Math.max(200, context.delay * 4);
    activeOfftabContext = context;
    try {{
      return invoke();
    }} finally {{
      activeOfftabContext = previousContext;
    }}
  }}

  function wrapGradioListener(listener) {{
    if (wrappedListeners.has(listener)) {{
      return wrappedListeners.get(listener);
    }}
    const wrapped = function (event) {{
      const detail = event && event.detail ? event.detail : null;
      if (!shouldUseOfftabScheduler(detail)) {{
        return listener.apply(this, arguments);
      }}
      patch.lastTargetedEvent = {{
        componentId: detail.id,
        eventName: typeof detail.event === "string" ? detail.event : "",
        timestamp: Date.now(),
      }};
      return withOfftabScheduler(detail, () => listener.apply(this, arguments));
    }};
    wrappedListeners.set(listener, wrapped);
    return wrapped;
  }}

  function wrapGradioHandleEventListener(listener) {{
    if (!listener || typeof listener.handleEvent !== "function") {{
      return listener;
    }}
    if (wrappedHandleEventListeners.has(listener)) {{
      return wrappedHandleEventListeners.get(listener);
    }}
    const wrapped = {{
      handleEvent(event) {{
        const detail = event && event.detail ? event.detail : null;
        if (!shouldUseOfftabScheduler(detail)) {{
          return listener.handleEvent.call(listener, event);
        }}
        patch.lastTargetedEvent = {{
          componentId: detail.id,
          eventName: typeof detail.event === "string" ? detail.event : "",
          timestamp: Date.now(),
        }};
        return withOfftabScheduler(detail, () => listener.handleEvent.call(listener, event));
      }},
    }};
    wrappedHandleEventListeners.set(listener, wrapped);
    return wrapped;
  }}

  const patch = {{
    enableBackgroundScheduler: {background_default_enabled},
    enableOfftabKeepAlive: {offtab_default_enabled},
    forceBackground: false,
    verbose: {verbose},
    targetComponentIds: null,
    targetComponentIdsPanelId: null,
    syntheticWindowUntil: 0,
    lastTargetedEvent: null,
    mainTab: null,
    mainTabPanel: null,
    mainTabPanelId: null,
    keepAliveObserver: null,
    keepAliveObserverInstalled: false,
    keepAliveSyncQueued: false,
    topLevelTabPrimeInstalled: false,
    mainTabInteractionPrimeInstalled: false,
    mainTabKeepAliveActive: false,
    interactionKeepAliveUntil: 0,
    backgroundDispatchCount: 0,
    offtabDispatchCount: 0,
    lastBackgroundDispatch: null,
    lastOfftabDispatch: null,
    isBackground() {{
      if (this.forceBackground) {{
        return true;
      }}
      try {{
        return document.visibilityState !== "visible" || !document.hasFocus();
      }} catch (_error) {{
        return false;
      }}
    }},
    shouldKeepMainTabAlive() {{
      return this.enableOfftabKeepAlive && !isMainTabSelected();
    }},
    shouldPatchBackground() {{
      return this.enableBackgroundScheduler && this.isBackground() && isMainTabActive();
    }},
    shouldPatchOfftab() {{
      return this.enableOfftabKeepAlive && !this.isBackground() && (this.mainTabKeepAliveActive || Date.now() < this.interactionKeepAliveUntil);
    }},
    getMainTabPanelId() {{
      const panel = getMainTabPanel();
      return panel ? panel.id : null;
    }},
    getTargetComponentIds,
    syncMainTabKeepAlive,
    scheduleMainTabKeepAliveSync,
    primeMainTabKeepAlive,
    primeMainTabInteraction,
  }};

  if (channel) {{
    channel.port1.onmessage = () => {{
      dispatchBackgroundJob(backgroundQueue.shift());
    }};
  }}

  window.__gradioFocusQueuePatch = patch;
  ensureKeepAliveStyle();
  installMainTabKeepAliveObserver();
  installTopLevelTabPrime();
  installMainTabInteractionPrime();

  window.requestAnimationFrame = function (callback) {{
    if (typeof callback !== "function") {{
      return nativeRequestAnimationFrame(callback);
    }}
    if (activeOfftabContext) {{
      return scheduleOfftabAnimationFrame(callback, activeOfftabContext);
    }}
    const backgroundActive = patch.shouldPatchBackground();
    const offtabActive = patch.shouldPatchOfftab();
    if (!backgroundActive && !offtabActive) {{
      return nativeRequestAnimationFrame(callback);
    }}
    const stack = new Error().stack || "";
    if (backgroundActive && shouldPatchBackgroundAnimationFrame(stack, callback)) {{
      return scheduleBackgroundAnimationFrame(callback);
    }}
    if (offtabActive && shouldPatchOfftabDeferredCallback(callback)) {{
      return scheduleOfftabAnimationFrame(callback, {{
        componentId: patch.lastTargetedEvent ? patch.lastTargetedEvent.componentId : null,
        eventName: patch.lastTargetedEvent ? patch.lastTargetedEvent.eventName : "",
        delay: OFFTAB_DELAY_MS,
      }});
    }}
    return nativeRequestAnimationFrame(callback);
  }};

  window.cancelAnimationFrame = function (animationFrameId) {{
    if (cancelSyntheticAnimationFrame(animationFrameId)) {{
      return;
    }}
    return nativeCancelAnimationFrame(animationFrameId);
  }};

  EventTarget.prototype.addEventListener = function (type, listener, options) {{
    if (type === "gradio") {{
      if (typeof listener === "function") {{
        return nativeAddEventListener.call(this, type, wrapGradioListener(listener), options);
      }}
      if (listener && typeof listener.handleEvent === "function") {{
        return nativeAddEventListener.call(this, type, wrapGradioHandleEventListener(listener), options);
      }}
    }}
    return nativeAddEventListener.call(this, type, listener, options);
  }};

  EventTarget.prototype.removeEventListener = function (type, listener, options) {{
    if (type === "gradio") {{
      if (typeof listener === "function" && wrappedListeners.has(listener)) {{
        return nativeRemoveEventListener.call(this, type, wrappedListeners.get(listener), options);
      }}
      if (listener && typeof listener.handleEvent === "function" && wrappedHandleEventListeners.has(listener)) {{
        return nativeRemoveEventListener.call(this, type, wrappedHandleEventListeners.get(listener), options);
      }}
    }}
    return nativeRemoveEventListener.call(this, type, listener, options);
  }};

  console.info("[Gradio] main tab keepalive patch installed");
}})();
"""


def _inject_script(template_source: str) -> str:
    if _PATCH_SENTINEL in template_source:
        return template_source
    script_tag = f"\n\t\t<script>\n{get_javascript()}\n\t\t</script>\n"
    module_tag = '<script type="module"'
    insert_at = template_source.find(module_tag)
    if insert_at != -1:
        return template_source[:insert_at] + script_tag + template_source[insert_at:]
    head_close = template_source.find("</head>")
    if head_close != -1:
        return template_source[:head_close] + script_tag + template_source[head_close:]
    return template_source + script_tag


def install() -> bool:
    if not ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH:
        return False
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv and sys.argv[0] else ""
    cwd = Path.cwd().resolve()
    if cwd != _PROJECT_ROOT and _PROJECT_ROOT not in cwd.parents and argv0 != "wgp.py":
        return False
    import gradio.routes as gradio_routes

    templates = getattr(gradio_routes, "templates", None)
    loader = getattr(getattr(templates, "env", None), "loader", None)
    if loader is None:
        return False
    if getattr(loader, "_focus_queue_patch_installed", False):
        return True

    original_get_source: Callable = loader.get_source

    def patched_get_source(environment, template):
        source, filename, uptodate = original_get_source(environment, template)
        if template in _TARGET_TEMPLATES:
            source = _inject_script(source)
        return source, filename, uptodate

    loader.get_source = patched_get_source
    loader._focus_queue_patch_installed = True
    loader._focus_queue_patch_original_get_source = original_get_source
    templates.env.cache.clear()
    return True


__all__ = [
    "BACKGROUND_SCHEDULER_DEFAULT_ENABLED",
    "ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH",
    "FOCUS_QUEUE_SERVER_CONFIG_KEY",
    "GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE",
    "OFFTAB_KEEPALIVE_DEFAULT_ENABLED",
    "get_javascript",
    "install",
]
