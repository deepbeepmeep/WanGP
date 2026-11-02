// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Live preview of the edit in the browser: chosen takes back to back with their trims and transitions, the song as the clock,
// and planned shots shown as animatic cards in their slots, so pacing can be judged before (and without) rendering.
import { esc, art } from './ui.js';

export function segments(doc, take) {
  const result = [];
  const songDriven = !!doc.brief.song && doc.edit.audio_mode !== 'clips';  // the song is the clock: transitions keep every shot on its part
  let position = 0;
  for (const shot of doc.shots) {
    if (!shot.included) continue;
    const chosen = take(shot);
    const length = chosen?.duration_s || shot.duration_s || 4;
    const start = +(shot.trim?.[0] || 0);
    const end = shot.trim?.[1] != null ? +shot.trim[1] : Math.min(length, shot.duration_s || length);  // a take ends at the planned length (as in the assembled edit)
    const duration = Math.max(0.1, (chosen ? end - start : shot.duration_s || 4));
    let transition = result.length && shot.transition?.type && shot.transition.type !== 'cut' ? { type: shot.transition.type, duration: Math.min(+shot.transition.duration_s || 0.5, duration / 2) } : null;
    if (transition && songDriven) {  // as the assembled edit: the outgoing shot runs into it (its spare frames, then its last frame held)
      transition = { ...transition, duration: Math.min(transition.duration, 1) };
      result.at(-1).duration += transition.duration;
    } else if (transition) position -= transition.duration;  // transitions overlap the two shots, as in the assembled edit
    result.push({ shot, take: chosen, start: position, duration, srcIn: chosen ? start : 0, transition });
    position += duration;
  }
  return result;
}

export class Sequencer {
  constructor(stage, { songUrl, songOffset = 0, songMode = 'song', fileUrl, onTime, onSegment }) {
    this.stage = stage;
    this.fileUrl = fileUrl;
    this.onTime = onTime;
    this.onSegment = onSegment;
    this.songMode = songMode;
    this.songOffset = songOffset;
    this.layers = [this.layer(), this.layer()];
    this.audio = songUrl ? Object.assign(new Audio(songUrl), { preload: 'auto' }) : null;
    this.time = 0;
    this.playing = false;
    this.segs = [];
    this.current = -1;
    this.frame = null;
  }

  layer() {
    const element = document.createElement('div');
    element.style.cssText = 'position:absolute;inset:0;opacity:0;transition:none';
    element.innerHTML = '<video playsinline preload="auto" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain"></video><img alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none"><div class="placeholder-frame" style="display:none"><div class="ph-art"></div><span></span></div>';
    this.stage.append(element);
    return { element, video: element.querySelector('video'), image: element.querySelector('img'), card: element.querySelector('.placeholder-frame'), segment: -1 };
  }

  setSegments(segs) {
    this.segs = segs;
    this.total = segs.length ? Math.max(...segs.map(seg => seg.start + seg.duration)) : 0;
    this.current = -1;
    this.layers.forEach(layer => layer.segment = -1);
    this.seek(Math.min(this.time, this.total));
  }

  indexAt(time) {
    for (let index = this.segs.length - 1; index >= 0; index--) if (time >= this.segs[index].start) return index;
    return 0;
  }

  load(layer, index) {
    if (layer.segment === index) return;
    layer.segment = index;
    const seg = this.segs[index];
    if (!seg) return;
    const kind = seg.take?.kind || 'planned';
    layer.video.style.display = kind === 'video' ? '' : 'none';
    layer.image.style.display = kind === 'image' ? '' : 'none';
    layer.card.style.display = kind === 'planned' ? '' : 'none';
    if (kind === 'video') {
      const url = this.fileUrl(seg.take.path);
      if (layer.video.dataset.src !== url) { layer.video.src = url; layer.video.dataset.src = url; }
      layer.video.muted = this.songMode === 'song' || !this.playing;
      try { layer.video.currentTime = seg.srcIn; } catch { /* metadata not loaded yet */ }
    } else if (kind === 'image') {
      layer.image.src = this.fileUrl(seg.take.path);
    } else {
      layer.card.querySelector('.ph-art').setAttribute('style', art(seg.shot.id + seg.shot.section, seg.shot.intent));
      layer.card.querySelector('span').innerHTML = `<b>${esc(seg.shot.title || seg.shot.id)}</b><br><span style="opacity:.8">${esc(seg.shot.intent || seg.shot.generation?.settings?.prompt || 'Not shot yet')}</span>`;
    }
  }

  render() {
    const time = this.time;
    const index = this.indexAt(time);
    const seg = this.segs[index];
    if (!seg) return;
    const next = this.segs[index + 1];
    const [front, back] = this.layers[0].segment === index ? this.layers : this.layers[1].segment === index ? [this.layers[1], this.layers[0]] : this.layers;
    this.load(front, index);
    front.element.style.opacity = '1';
    front.element.style.zIndex = '1';
    let backOpacity = 0;
    if (next) {
      this.load(back, index + 1);
      const into = time - next.start;
      if (next.transition && into > -0.001) {
        const progress = Math.min(1, into / next.transition.duration);
        backOpacity = progress;
        if (next.transition.type !== 'crossfade') {
          const color = next.transition.type === 'fade_white' ? '#fff' : '#000';
          this.stage.style.background = color;
          front.element.style.opacity = String(Math.max(0, 1 - progress * 2));
          backOpacity = Math.max(0, progress * 2 - 1);
        }
        if (back.video.style.display !== 'none' && this.playing && back.video.paused) back.video.play().catch(() => {});
      }
    }
    back.element.style.opacity = String(backOpacity);
    back.element.style.zIndex = '2';
    this.sync(front, seg, time);
    if (next && backOpacity > 0) this.sync(back, next, time);
    if (index !== this.current) {
      this.current = index;
      this.onSegment?.(seg, index);
    }
  }

  sync(layer, seg, time) {
    if (layer.video.style.display === 'none') return;
    const target = seg.srcIn + Math.max(0, time - seg.start);
    if (!this.playing) {
      if (Math.abs(layer.video.currentTime - target) > 0.04) { try { layer.video.currentTime = target; } catch { /* not ready */ } }
      if (!layer.video.paused) layer.video.pause();
      return;
    }
    layer.video.muted = this.songMode === 'song';
    if (layer.video.paused) layer.video.play().catch(() => {});
    if (Math.abs(layer.video.currentTime - target) > 0.25) { try { layer.video.currentTime = target; } catch { /* not ready */ } }
  }

  seek(time) {
    this.time = Math.max(0, Math.min(time, this.total || 0));
    if (this.audio) { try { this.audio.currentTime = this.songOffset + this.time; } catch { /* not ready */ } }
    this.stage.style.background = '#000';
    this.render();
    this.onTime?.(this.time);
  }

  play() {
    if (!this.segs.length) return;
    if (this.time >= this.total - 0.05) this.seek(0);
    this.playing = true;
    if (this.audio && this.songMode !== 'clips') { this.audio.currentTime = this.songOffset + this.time; this.audio.play().catch(() => {}); }
    this.last = performance.now();
    const tick = now => {
      if (!this.playing) return;
      if (this.audio && this.songMode !== 'clips' && !this.audio.paused) this.time = this.audio.currentTime - this.songOffset;
      else this.time += (now - this.last) / 1000;
      this.last = now;
      if (this.time >= this.total) { this.pause(); this.seek(this.total); return; }
      this.stage.style.background = '#000';
      this.render();
      this.onTime?.(this.time);
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  pause() {
    this.playing = false;
    cancelAnimationFrame(this.frame);
    this.audio?.pause();
    this.layers.forEach(layer => layer.video.pause());
  }

  toggle() { this.playing ? this.pause() : this.play(); return this.playing; }

  destroy() {
    this.pause();
    this.layers.forEach(layer => { layer.video.removeAttribute('src'); layer.video.load(); layer.element.remove(); });
    if (this.audio) { this.audio.removeAttribute('src'); this.audio.load(); }
  }
}
