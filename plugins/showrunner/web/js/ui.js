// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// DOM helpers, icons, toasts, dialogs, menus, formatting and the generative art of shots that have no picture yet.

const ICONS = {
  logo: '<path d="M4 7h16a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z"/><path d="M10 10.5v5l4.5-2.5z" fill="currentColor"/><path d="M5 3.5l3 3.2M19 3.5l-3 3.2"/>',
  home: '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  timeline: '<path d="M3 6h18M3 12h18M3 18h18"/><rect x="5" y="4" width="6" height="4" rx="1" fill="currentColor"/><rect x="11" y="10" width="8" height="4" rx="1" fill="currentColor"/><rect x="7" y="16" width="5" height="4" rx="1" fill="currentColor"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.6 3.3-5.5 6.5-5.5s5.7 1.9 6.5 5.5"/><circle cx="17.5" cy="9" r="2.6"/><path d="M16.5 14.6c2.6.2 4.4 1.9 5 5.4"/>',
  brief: '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 12h7M9 16h5"/>',
  film: '<rect x="3" y="3" width="18" height="18" rx="2.5"/><path d="M7 3v18M17 3v18M3 8h4M3 12h4M3 16h4M17 8h4M17 12h4M17 16h4"/>',
  music: '<path d="M9 18V5l11-2v13"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
  megaphone: '<path d="M3 11v3l11 5V6L3 11z"/><path d="M14 8.5c2.5 0 5 1.5 5 4s-2.5 4-5 4M6 14.5V19"/>',
  clapper: '<path d="M3 10h18v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M3 10l1.5-5.5 16 3.5L21 10"/><path d="M8 5.4l2 4.6M13 6.4l2 3.6"/>',
  book: '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>',
  sparkles: '<path d="M12 3l1.7 4.6L18 9.3l-4.3 1.7L12 15.6l-1.7-4.6L6 9.3l4.3-1.7z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  wand: '<path d="M4 20L16 8"/><path d="M15 3v3M18.5 4.5l-2 2M21 9h-3M13 4.5l1 1"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 18v1a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1"/>',
  settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  play: '<path d="M7 4.5v15l13-7.5z" fill="currentColor"/>',
  pause: '<rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor"/><rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7.5a4 4 0 0 1 8 0V11"/>',
  unlock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7.5a4 4 0 0 1 7.6-1.7"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  archive: '<rect x="3" y="4" width="18" height="5" rx="1"/><path d="M5 9v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9M10 13h4"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.6-4.5L4 8M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14.6 4.5L20 16M20 20v-4h-4"/>',
  send: '<path d="M5 12l15-8-6 16-2.5-6.5z"/><path d="M11.5 13.5L20 4"/>',
  expand: '<path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  chevronDown: '<path d="M6 9l6 6 6-6"/>',
  more: '<circle cx="5" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="19" cy="12" r="1.6" fill="currentColor"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.6 9.6 0 0 0 4.2-1"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5L5 20"/>',
  video: '<rect x="2.5" y="6" width="13" height="12" rx="2"/><path d="M15.5 10.5l6-3.5v10l-6-3.5z"/>',
  audio: '<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>',
  scissors: '<circle cx="6" cy="7" r="2.6"/><circle cx="6" cy="17" r="2.6"/><path d="M8.2 8.4L20 18M8.2 15.6L20 6"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  compare: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M12 3v18"/>',
  director: '<path d="M4 13.5c0-4.7 3.6-8.5 8-8.5s8 3.8 8 8.5"/><path d="M2.5 13.5h19"/><path d="M8 17.5c1 1.6 2.4 2.5 4 2.5s3-.9 4-2.5"/><circle cx="9" cy="10.5" r="1" fill="currentColor"/><circle cx="15" cy="10.5" r="1" fill="currentColor"/>',
  chat: '<path d="M4 5h16v11H9l-5 4z"/>',
  bin: '<path d="M3 7l2-3h14l2 3M3 7h18v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M9 11h6"/>',
  magnet: '<path d="M6 3v8a6 6 0 0 0 12 0V3"/><path d="M6 7h4M14 7h4"/>',
  zoomIn: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M8 11h6M11 8v6"/>',
  zoomOut: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M8 11h6"/>',
  transition: '<path d="M4 6h7v12H4zM13 6h7v12h-7z"/><path d="M9 12h6"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.6 2.3c-.7.4-1.2.9-1.2 1.7v.4M12 16.8v.4"/>',
  note: '<path d="M4 4h16v12l-4 4H4z"/><path d="M16 20v-4h4"/>',
  captions: '<rect x="2.5" y="5" width="19" height="14" rx="2.5"/><path d="M10.5 10.3a2.3 2.3 0 1 0 0 3.4M17 10.3a2.3 2.3 0 1 0 0 3.4"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/>',
  loader: '<path d="M12 3a9 9 0 1 0 9 9"/>',
  ratio: '<rect x="3" y="6" width="18" height="12" rx="2"/>',
  palette: '<path d="M12 3a9 9 0 0 0 0 18c1.3 0 2-.8 2-1.8 0-1.4-1.3-1.6-1.3-2.9 0-1 .9-1.8 2-1.8H17a4 4 0 0 0 4-4C21 6.6 17 3 12 3z"/><circle cx="7.5" cy="11" r="1.2" fill="currentColor"/><circle cx="10.5" cy="7" r="1.2" fill="currentColor"/><circle cx="15" cy="7.5" r="1.2" fill="currentColor"/>',
  gauge: '<path d="M4 16a8 8 0 1 1 16 0"/><path d="M12 16l4-5"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 10h1M10 10h1M14 10h1M18 10h0M7 14h10"/>',
};

export function icon(name, cls = '') {
  return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`;
}

export function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

// The showrunner names shots and cast members by their ids (s12, c3); the user knows shots by their place on the board ("shot 4", the
// title on hover) and people by their names. shotRefs returns escaped HTML, shotText plain text for titles and toasts.
export function shotRefs(text, shots, cast = [], versions = []) {
  return esc(versionRefs(text, versions)).replace(/\b([sc])(\d+)\b/g, match => {
    if (match[0] === 'c') return esc(cast.find(member => member.id === match)?.name || match);
    const index = shots.findIndex(shot => shot.id === match);
    return index < 0 ? match : `<span class="shot-ref" title="${esc(shots[index].title || '')} (${match})">shot ${index + 1}</span>`;
  });
}
// "Mia, the lead singer": what the Add dialogs ask for, a name and a role.
export function personFields(text) {
  const [name, ...role] = text.trim().split(',');
  return role.length ? { name: name.trim(), role: role.join(',').trim() } : { name: name.trim() };
}

// Deepy writes Markdown: the panel shows it as Deepy's own chat does (tables, lists, headings, bold, italics, code, links), raw HTML as text.
// words(text) turns the plain words between the marks into HTML (shot and cast names); media(id) shows a Gallery media id (visual:…, audio:…).
export function markdownHtml(text, words, media) {
  const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n'), out = [], item = /^\s*([-*+]|\d+[.)])\s+/;
  const cells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => mdSpan(cell.trim(), words, media));
  let paragraph = [];
  const flush = () => { if (paragraph.length) out.push(`<p>${paragraph.map(line => mdSpan(line, words, media)).join('<br>')}</p>`); paragraph = []; };
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (/^\s*```/.test(line)) {  // code, as it is
      flush();
      const code = [];
      while (++index < lines.length && !/^\s*```/.test(lines[index])) code.push(lines[index]);
      out.push(`<pre>${esc(code.join('\n'))}</pre>`);
    } else if (line.includes('|') && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[index + 1] || '')) {  // a table: its head, the rule under it, its rows
      flush();
      const head = cells(line), rows = [];
      for (index += 2; index < lines.length && lines[index].includes('|'); index++) rows.push(cells(lines[index]));
      index--;
      out.push(`<div class="md-table"><table><thead><tr>${head.map(cell => `<th>${cell}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    } else if (/^\s*#{1,6}\s+/.test(line)) {
      flush();
      out.push(`<p class="md-h">${mdSpan(line.replace(/^\s*#{1,6}\s+/, '').replace(/\s+#+\s*$/, ''), words, media)}</p>`);
    } else if (item.test(line)) {  // a list, numbered or not
      flush();
      const numbered = /^\s*\d/.test(line), tag = numbered ? 'ol' : 'ul', items = [];
      for (; index < lines.length && item.test(lines[index]) && /^\s*\d/.test(lines[index]) === numbered; index++) items.push(`<li>${mdSpan(lines[index].replace(item, ''), words, media)}</li>`);
      index--;
      out.push(`<${tag}>${items.join('')}</${tag}>`);
    } else if (!line.trim() || /^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) flush();
    else paragraph.push(line);
  }
  flush();
  return out.join('');
}
const MEDIA_ID = /^(?:.*\/)?((?:visual|audio):[0-9a-f]{12})$/;  // an id, or a link to it (/wangp_api/gallery/media/visual:…)
function mdSpan(text, words, media) {  // the marks of a line: code, pictures, links and media ids are set aside while its words become HTML
  const kept = [], keep = html => `\u0000${kept.push(html) - 1}\u0000`, idOf = url => { try { return decodeURIComponent(url).match(MEDIA_ID)?.[1]; } catch { return null; } };
  text = text.replace(/`([^`\n]+)`/g, (_, code) => keep(`<code>${esc(code)}</code>`))
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, url) => keep(idOf(url) ? media(idOf(url)) : words(alt)))
    .replace(/\[([^\]]*)\]\(([^)\s]+)\)/g, (_, label, url) => keep(idOf(url) ? media(idOf(url)) : /^https?:\/\//.test(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>` : words(label)))
    .replace(/\b(?:visual|audio):[0-9a-f]{12}\b/g, id => keep(media(id)));
  return words(text).replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>').replace(/(^|[^*\w])\*([^*\s][^*\n]*)\*(?!\w)/g, '$1<i>$2</i>').replace(/\u0000(\d+)\u0000/g, (_, number) => kept[number]);
}

export function shotText(text, shots, cast = [], versions = []) {
  return versionRefs(text, versions).replace(/\b([sc])(\d+)\b/g, match => {
    if (match[0] === 'c') return cast.find(member => member.id === match)?.name || match;
    const index = shots.findIndex(shot => shot.id === match);
    return index < 0 ? match : `shot ${index + 1}`;
  });
}
// Versions as the Deliver page numbers them: an export's file (with its folder), its revision ("r55") or its id read "version 3".
const versionNames = new WeakMap();  // a document's deliverables → their patterns, built once per document
function versionRefs(text, versions) {
  if (!versions.length) return String(text ?? '');
  let patterns = versionNames.get(versions);
  if (!patterns) versionNames.set(versions, patterns = versions.map((item, index) => {
    const file = item.path.split('/').pop().replace(/\.\w+$/, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return [new RegExp(`(?:\\b[Vv]ersion )?(?:(?:\\S*/)?${file}\\.\\w+|\\b${item.id}\\b${item.revision != null ? `|\\b(?:r|revision )${item.revision}\\b` : ''})`, 'g'), `version ${index + 1}`];
  }));  // a word "version" already in front is part of the name ("Version r55" reads "Version 1", not "Version version 1")
  return patterns.reduce((result, [pattern, name]) => result.replace(pattern, match => match[0] === 'V' ? `V${name.slice(1)}` : name), String(text ?? ''));
}

export function h(html) {
  const template = document.createElement('template');
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

export function $(selector, root = document) { return root.querySelector(selector); }
export function $$(selector, root = document) { return [...root.querySelectorAll(selector)]; }

export function on(root, event, selector, handler, options) {
  root.addEventListener(event, eventObject => {
    const target = eventObject.target.closest?.(selector);
    if (target && root.contains(target)) handler(eventObject, target);
  }, options);
}

export function formatTime(seconds, precise = false) {
  if (seconds == null || isNaN(seconds)) return '--:--';
  const sign = seconds < 0 ? '-' : '';
  seconds = Math.abs(seconds);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return sign + `${minutes}:${(precise ? rest.toFixed(2) : Math.floor(rest).toString()).padStart(precise ? 5 : 2, '0')}`;
}

export function formatDuration(seconds) {
  if (!seconds) return '0s';
  if (seconds < 60) return `${+seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

// The bar of a comparison player: it moves both videos together, to any moment of the longer one.
export const COMPARE_SEEK = '<div class="compare-bar"><span class="mono" id="cmp-time">0:00</span><input type="range" id="cmp-seek" min="0" max="1000" value="0" aria-label="Move through both videos" title="Move through both videos"></div>';
export function bindCompareSeek(root) {
  const a = root.querySelector('#cmp-a'), b = root.querySelector('#cmp-b'), seek = root.querySelector('#cmp-seek'), time = root.querySelector('#cmp-time');
  const length = () => Math.max(a.duration || 0, b.duration || 0);
  let moving = false;
  const show = () => {
    if (!length()) return;
    if (!moving) seek.value = Math.round(a.currentTime / length() * 1000);
    time.textContent = `${formatTime(a.currentTime)} / ${formatTime(length())}`;
  };
  const jump = seconds => { a.currentTime = seconds; b.currentTime = seconds; show(); };
  for (const video of [a, b]) video.addEventListener('loadedmetadata', show);
  a.addEventListener('timeupdate', show);
  seek.addEventListener('input', () => { moving = true; jump(seek.value / 1000 * length()); });
  seek.addEventListener('change', () => { moving = false; });
  return jump;
}

export function count(number, word) { return `${number} ${word}${number === 1 ? '' : 's'}`; }  // "1 shot", "3 shots"

// Roughly how long generating these shots takes, from the measured times of their models (api rates: seconds per second of video, or per
// picture). Drawn shots take no generation; shots on a model not timed yet are counted apart.
export function shootEstimate(shots, rates) {
  let seconds = 0, untimed = 0, timed = 0;
  for (const shot of shots) {
    const template = shot.generation?.template || 'gen_video', rate = rates[template];
    if (template === 'draw_svg') continue;
    if (!rate) { untimed++; continue; }
    timed++;
    seconds += rate.per_picture ?? rate.per_second * (+shot.duration_s || 0);
  }
  if (!timed) return untimed ? 'Time estimate after the first take' : '';
  const minutes = Math.max(1, Math.round(seconds / 60));
  return `about ${minutes < 90 ? `${minutes} min` : `${(seconds / 3600).toFixed(1)} h`}${untimed ? `, plus ${count(untimed, 'shot')} not timed yet` : ''}`;
}

export function ago(timestamp) {
  const seconds = Date.now() / 1000 - timestamp;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(timestamp * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

export function hash(text) {
  let value = 2166136261;
  for (const char of String(text)) value = Math.imul(value ^ char.charCodeAt(0), 16777619);
  return value >>> 0;
}

// A small, deterministic gradient "painting" standing in for a picture that does not exist yet.
export function art(seed, mood = '') {
  const base = hash(seed + mood);
  const hue = (base % 360);
  const hue2 = (hue + 40 + (base >> 9) % 90) % 360;
  const hue3 = (hue + 180 + (base >> 15) % 60) % 360;
  const x1 = 15 + (base >> 3) % 50, y1 = 10 + (base >> 7) % 40, x2 = 50 + (base >> 11) % 45, y2 = 45 + (base >> 13) % 50;
  return `background: radial-gradient(60% 70% at ${x1}% ${y1}%, hsla(${hue},85%,60%,.85), transparent 70%), radial-gradient(55% 60% at ${x2}% ${y2}%, hsla(${hue2},80%,55%,.75), transparent 70%), linear-gradient(135deg, hsl(${hue3},45%,14%), hsl(${hue},40%,8%));`;
}

export function sectionColor(name = '') {
  const key = String(name).toLowerCase();
  if (key.startsWith('intro')) return 'var(--sec-intro)';
  if (key.startsWith('verse')) return 'var(--sec-verse)';
  if (key.startsWith('chorus') || key.startsWith('hook') || key.startsWith('drop')) return 'var(--sec-chorus)';
  if (key.startsWith('bridge')) return 'var(--sec-bridge)';
  if (key.startsWith('outro') || key.startsWith('end')) return 'var(--sec-outro)';
  return `hsl(${hash(key) % 360} 60% 55%)`;
}

export function aspectValue(aspect) {
  const [width, height] = String(aspect || '16:9').split(':').map(Number);
  return width && height ? `${width}/${height}` : '16/9';
}

export function toast(message, kind = 'info', timeout = 3800) {
  const element = h(`<div class="toast ${kind}">${icon(kind === 'error' ? 'alert' : kind === 'ok' ? 'check' : 'info')}<span>${esc(message)}</span></div>`);
  document.getElementById('toasts').append(element);
  setTimeout(() => { element.style.opacity = '0'; element.style.transition = 'opacity .3s'; setTimeout(() => element.remove(), 320); }, timeout);
}

export function confirmDialog({ title, text, ok = 'Confirm', danger = false }) {
  return new Promise(resolve => {
    const back = h(`<div class="modal-back"><div class="modal" role="dialog" aria-modal="true"><h3>${esc(title)}</h3><p>${esc(text)}</p>
      <div class="actions"><button class="btn ghost" data-r="0">Cancel</button><button class="btn ${danger ? 'danger' : 'primary'}" data-r="1">${esc(ok)}</button></div></div></div>`);
    const close = value => { back.remove(); document.removeEventListener('keydown', keys); resolve(value); };
    const keys = event => { if (['Escape', 'Enter'].includes(event.key)) { event.stopPropagation(); close(event.key === 'Enter'); } };  // the page behind does not see the key
    back.addEventListener('click', event => { if (event.target === back) close(false); const button = event.target.closest('[data-r]'); if (button) close(button.dataset.r === '1'); });
    document.addEventListener('keydown', keys);
    document.getElementById('overlay-root').append(back);
    back.querySelector('[data-r="1"]').focus();
  });
}

// A picture, video or sound shown over the page (the showrunner's conversation opens its media here): Esc or a click beside it closes it.
export function mediaPopup({ kind, url, name = '' }) {
  const media = kind === 'image' ? `<img src="${esc(url)}" alt="${esc(name)}">` : `<${kind} src="${esc(url)}" controls autoplay></${kind}>`;
  const back = h(`<div class="modal-back"><div class="media-popup" role="dialog" aria-modal="true" aria-label="${esc(name)}">${media}${name ? `<div class="media-popup-name">${esc(name)}</div>` : ''}</div></div>`);
  const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
  const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
  back.addEventListener('click', event => { if (event.target === back) close(); });
  document.addEventListener('keydown', keys);
  document.getElementById('overlay-root').append(back);
}

export function promptDialog({ title, text = '', value = '', placeholder = '', ok = 'Save', multiline = false, rows = 4, maxLength = 0 }) {
  return new Promise(resolve => {
    const limit = maxLength ? ` maxlength="${maxLength}"` : '';
    const field = multiline ? `<textarea class="textarea" rows="${rows}"${limit} placeholder="${esc(placeholder)}">${esc(value)}</textarea>` : `<input class="input"${limit} value="${esc(value)}" placeholder="${esc(placeholder)}">`;
    const back = h(`<div class="modal-back"><div class="modal" role="dialog" aria-modal="true"><h3>${esc(title)}</h3>${text ? `<p>${esc(text)}</p>` : ''}${field}
      <div class="actions"><button class="btn ghost" data-r="0">Cancel</button><button class="btn primary" data-r="1">${esc(ok)}</button></div></div></div>`);
    const input = back.querySelector('input,textarea');
    const close = result => { back.remove(); resolve(result); };
    back.addEventListener('click', event => { if (event.target === back) close(null); const button = event.target.closest('[data-r]'); if (button) close(button.dataset.r === '1' ? input.value : null); });
    input.addEventListener('keydown', event => { if (event.key === 'Escape') { event.stopPropagation(); close(null); } if (event.key === 'Enter' && (!multiline || event.ctrlKey || event.metaKey)) close(input.value); });
    document.getElementById('overlay-root').append(back);
    input.focus();
    input.select?.();
  });
}

let openMenu = null;
// A button's attributes: greyed out with the reason it cannot be used now as its tooltip, or usable with its own tooltip.
// What a local Deepy's settings cost a production: until closed, as long as it says the same.
let closedAdvice = null;
export function adviceBar(info) {
  try { closedAdvice ??= localStorage.getItem('sr.advice'); } catch { /* a convenience */ }
  return info?.advice && info.advice !== closedAdvice ? `<div class="next-step advice">${icon('alert')}<span class="grow">${esc(info.advice)}</span><button class="btn ghost icon" id="advice-close" title="Close" aria-label="Close">${icon('x')}</button></div>` : '';
}
export function bindAdvice(root, info) {
  root.querySelector('#advice-close')?.addEventListener('click', () => {
    closedAdvice = info.advice;
    try { localStorage.setItem('sr.advice', info.advice); } catch { /* a convenience */ }
    root.querySelector('.advice')?.remove();
  });
}

export function availability(reason, title = '') {
  return reason ? `disabled title="${esc(reason)}"` : title ? `title="${esc(title)}"` : '';
}

export function menu(anchor, items) {
  closeMenu();
  const element = h(`<div class="menu" role="menu">${items.map((item, index) => item === '-' ? '<hr>' : `<button data-i="${index}" class="${item.danger ? 'danger' : ''}" ${availability(item.unavailable)}>${icon(item.icon || 'chevron')}<span>${esc(item.label)}</span></button>`).join('')}</div>`);
  document.body.append(element);
  const rect = anchor instanceof Element ? anchor.getBoundingClientRect() : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
  const width = element.offsetWidth, height = element.offsetHeight;
  element.style.left = `${Math.min(window.innerWidth - width - 8, rect.left)}px`;
  element.style.top = `${rect.bottom + height + 8 > window.innerHeight ? rect.top - height - 6 : rect.bottom + 6}px`;
  element.addEventListener('click', event => { const button = event.target.closest('[data-i]'); if (button) { closeMenu(); items[+button.dataset.i].action(); } });
  openMenu = element;
  setTimeout(() => document.addEventListener('pointerdown', outside, { once: true }), 0);
  function outside(event) { if (!element.contains(event.target)) closeMenu(); else document.addEventListener('pointerdown', outside, { once: true }); }
}
export function closeMenu() { openMenu?.remove(); openMenu = null; }

export function pickFiles(accept, multiple = true) {
  return new Promise(resolve => {
    const input = Object.assign(document.createElement('input'), { type: 'file', accept, multiple });
    input.addEventListener('change', () => resolve([...input.files]));
    input.click();
  });
}

const AUDIO_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus'];
// File pickers name audio by extension: iOS Safari filters audio/* as video and greys out audio files.
export const AUDIO_ACCEPT = AUDIO_EXTENSIONS.map(extension => '.' + extension).join(',');

export function kindOfFile(name) {
  const extension = String(name).toLowerCase().split('.').pop();
  if (['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(extension)) return 'image';
  if (['mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v'].includes(extension)) return 'video';
  if (AUDIO_EXTENSIONS.includes(extension)) return 'audio';
  return null;
}

// Waveform drawing shared by the brief, the timeline and the cast's voices.
export function drawWave(canvas, peaks, { color = '#ff3d81', color2 = '#8b5cf6', progress = 0, from = 0, to = 1, bars = true } = {}) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const peak = peaks?.peak || [];
  if (!peak.length) return;
  const gradient = context.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, color);
  gradient.addColorStop(1, color2);
  const start = Math.floor(from * peak.length), end = Math.max(start + 1, Math.ceil(to * peak.length));
  const step = bars ? 3 : 1;
  const columns = Math.floor(width / step);
  const middle = height / 2;
  for (let column = 0; column < columns; column++) {
    const index = start + Math.floor((column / columns) * (end - start));
    const value = peak[index] || 0;
    const rms = (peaks.rms || [])[index] || value * .6;
    const x = column * step;
    context.globalAlpha = x / width < progress ? 1 : .55;
    context.fillStyle = gradient;
    const barHeight = Math.max(1.5, value * (height - 6));
    context.fillRect(x, middle - barHeight / 2, Math.max(1, step - 1), barHeight);
    context.globalAlpha = x / width < progress ? .9 : .35;
    context.fillStyle = '#ffffff';
    const rmsHeight = Math.max(1, rms * (height - 6) * .55);
    context.fillRect(x, middle - rmsHeight / 2, Math.max(1, step - 1), rmsHeight);
  }
  context.globalAlpha = 1;
}

// Saves a form's edits after a pause in typing, merging every field edited meanwhile for the same target (a shot, a cast member, the
// brief): typing in one field and then quickly in the next keeps both. flush() sends at once (switching shots or views, leaving the page).
export function fieldSaver(send, wait = 450) {
  const pending = new Map();
  let timer = null;
  const flush = () => { clearTimeout(timer); timer = null; const items = [...pending]; pending.clear(); items.forEach(([key, fields]) => send(key, fields)); };
  const save = (key, fields) => { pending.set(key, mergeFields(pending.get(key) || {}, fields)); clearTimeout(timer); timer = setTimeout(flush, wait); };
  save.flush = flush;
  save.pending = key => pending.get(key);
  window.addEventListener('pagehide', flush);
  return save;
}

function mergeFields(base, extra) {
  const plain = value => value && typeof value === 'object' && !Array.isArray(value);
  const merged = { ...base };
  for (const [key, value] of Object.entries(extra)) merged[key] = plain(value) && plain(base[key]) ? mergeFields(base[key], value) : value;
  return merged;
}

// At most one call per wait: the first at once, the last one asked for when the wait ends (live events come many times a second).
export function throttle(fn, wait = 100) {
  let timer = null, asked = false;
  const run = () => { timer = null; if (asked) call(); };
  const call = () => { asked = false; fn(); timer = setTimeout(run, wait); };
  return () => { if (timer) asked = true; else call(); };
}

export function debounce(fn, wait = 250) {
  let timer = null;
  const wrapped = (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
  wrapped.flush = (...args) => { clearTimeout(timer); fn(...args); };
  return wrapped;
}
