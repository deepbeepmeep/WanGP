// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Deepy panel: Deepy's own window on the production's conversation, read only: its thoughts and every step, as they are written.
import { state, deepyWait } from './app.js';
import { api } from './api.js';

const WINDOW = '../deepy/window?contour=0&input=0&avatars=0&stop=0&actions=0&wide=1&font=0.84';

export function mountConversation(root) {
  let frame = null, view = null, after = 0, open = false, loading = false, gap = 0;

  // What Deepy saved since the last look: all of it the first time, and what the window missed while it was out of sight.
  async function load() {
    if (loading || !view) return;
    loading = true;
    try {
      const pid = state.pid, first = !after, data = await api.conversation(pid, after);
      if (pid !== state.pid) return;
      if (data.restart) view.reset();
      after = data.next;
      if (first || data.restart) view.replay(data.events);
      else if (data.events.length) view.update({ batch: data.events });
    } catch { /* the next look tries again */ } finally { loading = false; }
  }

  // What a window opening in the middle of a step has not heard yet.
  function seed() {
    if (!state.live.running) return;
    if (state.live.stats) view.update({ type: 'stats', stats: { ...state.live.stats, visible: true } });
    if (state.live.status) view.update({ type: 'status', status: { ...state.live.status, visible: true } });
    if (deepyWait(state.live)) view.update({ type: 'status', status: { visible: true, kind: 'loading', text: deepyWait(state.live), progress: state.live.status?.progress } });
  }

  return {
    visible(now) {
      if (now === open) return;
      open = now;
      if (!open) return;
      if (view) { view.shown(); view.rejoin(); load().then(seed); return; }
      if (frame) return;  // still loading
      frame = document.createElement('iframe');
      frame.title = 'Deepy';
      frame.src = `${WINDOW}&theme=${document.documentElement.dataset.theme}`;
      frame.addEventListener('load', () => {
        view = frame.contentWindow.DeepyWindow;
        view.onGap = () => { clearTimeout(gap); gap = setTimeout(load, 1500); };  // an update it could not place: what Deepy saved fills the hole
        load().then(seed);
      });
      root.append(frame);
    },
    live(update) { if (open) view?.update(update); },
    catchUp() { if (open) load(); },
    finished() {  // its last status never comes: the job's updates end with the task
      view?.update({ type: 'status', status: null });
      if (open) load();
    },
    theme() { view?.theme(document.documentElement.dataset.theme); },
  };
}
