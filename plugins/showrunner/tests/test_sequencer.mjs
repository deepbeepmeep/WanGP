// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Run: node --experimental-vm-modules --test plugins/showrunner/tests/test_sequencer.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm';
import { afterEach, beforeEach, test } from 'node:test';
import { Sequencer } from '../web/js/sequencer.js';

let audios, contexts, frames, clock, restore;
class Media {
  constructor(src = '') { this.src = src; this.paused = true; this.currentTime = 0; this.duration = 20; this.volume = 1; this.muted = false; this.style = {}; this.dataset = {}; }
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
  removeAttribute(name) { if (name === 'src') this.src = ''; }
  load() { this.loaded = true; }
}
class Node {
  constructor() { this.connections = []; this.gain = { value: 1 }; }
  connect(node) { this.connections.push(node); return node; }
  disconnect() { this.connections = []; }
}
class Context {
  constructor() { this.destination = {}; this.sources = []; this.state = 'suspended'; contexts.push(this); }
  resume() { this.state = 'running'; return Promise.resolve(); }
  close() { this.state = 'closed'; return Promise.resolve(); }
  createMediaElementSource(media) { const node = new Node(); node.media = media; this.sources.push(node); return node; }
  createGain() { return new Node(); }
}
function layer() {
  const children = { video: new Media(), img: { style: {} }, '.placeholder-frame': { style: {} } };
  return { style: {}, querySelector: name => children[name], remove() {} };
}
beforeEach(() => {
  audios = []; contexts = []; frames = new Map(); clock = 0;
  restore = new Map(['Audio', 'AudioContext', 'document', 'requestAnimationFrame', 'cancelAnimationFrame', 'performance'].map(key => [key, globalThis[key]]));
  globalThis.Audio = class extends Media { constructor(src) { super(src); audios.push(this); } };
  globalThis.AudioContext = Context;
  globalThis.document = { createElement: layer };
  globalThis.performance = { now: () => clock };
  globalThis.requestAnimationFrame = callback => { const id = frames.size + 1; frames.set(id, callback); return id; };
  globalThis.cancelAnimationFrame = id => frames.delete(id);
});
afterEach(() => { for (const [key, value] of restore) { if (value === undefined) delete globalThis[key]; else globalThis[key] = value; } });

function player(options = {}) {
  const sequencer = new Sequencer({ append() {}, style: {} }, { fileUrl: path => `/media/${path}`, songMode: 'mix', ...options });
  sequencer.setSegments([{ start: 0, duration: 12, srcIn: 0, shot: {}, take: { kind: 'video', path: 'dialogue.mp4' } }]);
  return sequencer;
}
const track = (path, start_s = 0, gain_db = 0) => ({ path, start_s, gain_db });
const audio = path => { const found = audios.find(item => item.src === `/media/${path}`); assert.ok(found, `${path} was not registered for playback`); return found; };

test('Mix starts added music alongside the take dialogue', () => {
  const p = player({ audioTracks: [track('music.wav')] });
  p.play(); p.render();
  assert.equal(audio('music.wav').paused, false);
  assert.equal(p.layers[0].video.paused, false);
  assert.equal(p.layers[0].video.muted, false);
  assert.equal(contexts[0].state, 'running');
  p.destroy();
});

test('delayed SFX follow seeks, stop at their end, and restart after seeking back', () => {
  const p = player({ audioTracks: [track('sfx.wav', 3)] });
  p.play();
  const effect = audio('sfx.wav'); effect.duration = 2;
  assert.equal(effect.paused, true);
  p.seek(4); assert.equal(effect.paused, false); assert.equal(effect.currentTime, 1);
  p.seek(5.5); assert.equal(effect.paused, true);
  p.seek(2); assert.equal(effect.paused, true); assert.equal(effect.currentTime, 0);
  p.seek(3.1); assert.equal(effect.paused, false); assert.ok(Math.abs(effect.currentTime - .1) < 1e-9);
  p.destroy();
});

for (const mode of ['song', 'clips', 'mix']) test(`${mode} keeps added tracks and applies the selected take/song policy`, () => {
  const p = player({ songMode: mode, songUrl: '/media/song.wav', songOffset: 2, audioTracks: [track('music.wav'), track('sfx.wav')] });
  p.play(); p.render();
  assert.equal(audio('music.wav').paused, false); assert.equal(audio('sfx.wav').paused, false);
  const song = audio('song.wav');
  assert.equal(song.paused, mode === 'clips');
  assert.equal(p.layers[0].video.muted, mode === 'song');
  if (mode !== 'clips') { assert.equal(song.currentTime, 2); assert.equal(song.volume, mode === 'mix' ? .5 : 1); }
  p.destroy();
});

test('track dB gains reach the output, including positive gain', () => {
  const p = player({ audioTracks: [track('quiet.wav', 0, -6), track('loud.wav', 0, 6)] });
  p.play();
  for (const [path, expected] of [['quiet.wav', .5011872336272722], ['loud.wav', 1.9952623149688795]]) {
    const source = contexts[0].sources.find(node => node.media === audio(path));
    assert.ok(source, `${path} has no output source`);
    const gain = source.connections[0];
    assert.ok(Math.abs(gain.gain.value - expected) < 1e-9);
    assert.deepEqual(gain.connections, [contexts[0].destination]);
  }
  p.destroy();
});

test('editing tracks during playback replaces their players without resetting the playhead', () => {
  const p = player({ audioTracks: [track('old.wav')] });
  p.play(); p.seek(4);
  const old = audio('old.wav');
  p.setAudioTracks([track('new.wav', 1, -5)]);
  assert.equal(old.paused, true); assert.equal(old.src, '');
  assert.equal(p.playing, true); assert.equal(p.time, 4);
  assert.equal(audio('new.wav').paused, false); assert.equal(audio('new.wav').currentTime, 3);
  const count = audios.length; p.setAudioTracks([track('new.wav', 1, -5)]);
  assert.equal(audios.length, count);
  p.destroy();
});

test('pause, end, and destroy stop and release all added audio', () => {
  const p = player({ audioTracks: [track('music.wav'), track('sfx.wav')] });
  p.play(); p.seek(4); p.pause();
  assert.ok(audios.every(item => item.paused));
  p.play(); assert.equal(audio('music.wav').currentTime, 4);
  clock = 13000;
  const tick = [...frames.values()].at(-1); tick(clock);
  assert.equal(p.playing, false); assert.ok(audios.every(item => item.paused));
  p.destroy();
  assert.ok(audios.every(item => !item.src && item.loaded));
  assert.equal(contexts[0].state, 'closed');
});

test('Timeline transport actually plays its saved added tracks', async () => {
  // Run the real Timeline view and Sequencer; stub only its browser shell and unrelated views.
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, { style: {}, dataset: {}, clientWidth: 1000, scrollLeft: 0,
      classList: { toggle() {} }, addEventListener() {}, append() {}, querySelectorAll: () => [],
      querySelector: child => element(selector + child), getBoundingClientRect: () => ({ left: 0, width: 1000 }) });
    return elements.get(selector);
  };
  const take = { kind: 'video', path: 'dialogue.mp4', duration_s: 12 };
  const doc = { brief: {}, audio: {}, deliverables: [], edit: { audio_mode: 'mix', audio_tracks: [track('music.wav')], overlays: [] },
    shots: [{ id: 's1', title: 'Dialogue', included: true, duration_s: 12, trim: [0, 12], status: 'approved' }] };
  const noop = () => {};
  const ui = Object.fromEntries(['icon','art','h','sectionColor','drawWave','toast','menu','confirmDialog','pickFiles'].map(name => [name, noop]));
  Object.assign(ui, { esc: String, formatTime: String, formatDuration: String, AUDIO_ACCEPT: '' });
  const dependencies = {
    './app.js': { state: { doc, selection: new Set() }, actions: { take: () => take, file: path => `/media/${path}` } },
    './api.js': { api: {} }, './ui.js': ui, './sequencer.js': { Sequencer, segments: (await import('../web/js/sequencer.js')).segments },
    './storyboard.js': { shotMenu: noop },
  };
  const context = createContext({ Audio: globalThis.Audio, localStorage: { getItem: () => null }, window: { addEventListener: noop, removeEventListener: noop } });
  const url = process.env.TIMELINE_TEST_SOURCE || new URL('../web/js/timeline.js', import.meta.url);
  const module = new SourceTextModule(await readFile(url, 'utf8'), { context });
  await module.link(name => {
    const exports = dependencies[name]; assert.ok(exports, `Unexpected Timeline dependency ${name}`);
    return new SyntheticModule(Object.keys(exports), function() { for (const [key,value] of Object.entries(exports)) this.setExport(key,value); }, { context });
  });
  await module.evaluate();
  const view = module.namespace.mountTimeline({ querySelector: element, querySelectorAll: () => [], addEventListener: noop }, doc);
  try {
    element('#play').onclick();
    assert.equal(audio('music.wav').paused, false);
    const original = audio('music.wav');
    doc.edit.audio_tracks = [track('replacement.wav', 0, -5)]; view.update();
    assert.equal(original.paused, true); assert.equal(audio('replacement.wav').paused, false);
  } finally { view.destroy(); }
});
