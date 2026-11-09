// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Point at what should change: the shots concerned, a new shot between two others, and what to do. The showrunner does that and keeps the rest.
import { state, actions } from './app.js';
import { esc, h, icon, formatTime, count } from './ui.js';
import { segments } from './sequencer.js';
import { shotPoster } from './storyboard.js';

export function shotAt(doc, time) {
  return segments(doc, actions.take).findLast(item => item.start <= time)?.shot.id;
}

// A new shot joins the part of the shot it follows, with its template: a drawn card is no model for a shot.
function newShot(doc, position) {
  const drawn = shot => shot?.generation?.template === 'draw_svg';
  const before = doc.shots.slice(0, position).reverse().find(shot => !drawn(shot)) || doc.shots.find(shot => !drawn(shot));
  return { title: 'New shot', section: doc.shots[position - 1]?.section || '', duration_s: 4, generation: { template: before?.generation?.template || 'gen_video', settings: { prompt: '' } } };
}

// Make a new version when done: the choice stays for the next fixes of the page, for a user who fixes shot after shot before watching.
let makeVersion = true;

export function askForFix(ids, time = null, insertAfter, text = '') {
  const doc = state.doc, picked = new Set(ids), added = insertAfter === undefined ? [] : [insertAfter];  // added: the shot each new shot follows, null for the start
  let anchor = ids.at(-1);
  const back = h(`<div class="modal-back"><div class="modal fix-dialog" role="dialog" aria-modal="true"><h3>${time === null ? 'What should change?' : `What should change around ${esc(formatTime(time))}?`}</h3>
    <div class="fix-shots"></div>
    <textarea class="textarea" rows="3" placeholder="Make it 6 s and show her tasting the sauce. The lips do not follow the song. Her coat changes colour…"></textarea>
    <label class="fix-keep"><input type="checkbox" checked> Keep everything else as it is</label>
    <label class="fix-version" title="Off: the showrunner stops once the new takes are judged, and you press Assemble when you want to watch the edit"><input type="checkbox" ${makeVersion ? 'checked' : ''}> Make a new version when done</label>
    <p class="fix-hint"></p>
    <div class="actions"><button class="btn ghost" data-cancel>Cancel</button><button class="btn primary" data-send></button></div></div></div>`);
  back.querySelector('textarea').value = text;
  const strip = back.querySelector('.fix-shots'), note = back.querySelector('textarea'), send = back.querySelector('[data-send]'), keep = back.querySelector('.fix-keep input');
  const version = back.querySelector('.fix-version'), review = doc.brief.review_before_edit;  // with the takes reviewed before the edit, a fix ends on that review
  const gap = after => `<button class="fix-gap" data-after="${after ?? ''}" title="Insert a new shot here">${icon('plus')}</button>`;
  const fresh = after => added.map((item, index) => item === after ? `<button class="fix-shot on fix-new" data-new="${index}" title="Click to take it out again"><div class="fix-empty">${icon('plus')}</div><span>New shot</span></button>` : '').join('');
  const cards = () => gap(null) + fresh(null) + doc.shots.map((shot, index) => shot.included ? `<button class="fix-shot ${picked.has(shot.id) ? 'on' : ''}" data-shot="${shot.id}" ${shot.locked ? 'disabled title="Locked: the showrunner keeps it. Unlock it to have it redone."' : ''}>
    ${shotPoster(shot) ? `<img src="${shotPoster(shot)}" alt="">` : '<div class="fix-empty"></div>'}<span>${index + 1}. ${esc(shot.title || 'Untitled')}</span></button>${fresh(shot.id)}${gap(shot.id)}` : '').join('');
  const refresh = () => {
    const scroll = strip.scrollLeft;
    strip.innerHTML = cards();
    strip.scrollLeft = scroll;
    const redo = picked.size, more = added.length, scoped = (redo || more) && keep.checked;
    send.textContent = redo && more ? `Fix ${redo}, add ${more}` : more ? `Add ${count(more, 'shot')}` : redo ? `Fix ${count(redo, 'shot')}` : 'Tell the showrunner';
    back.querySelector('.fix-keep').hidden = !redo && !more;
    version.hidden = (!redo && !more) || review;
    send.disabled = (!scoped || more > 0) && !note.value.trim();
    const ending = review ? 'asks you to review the new takes' : makeVersion ? 'makes a new version' : 'stops once the new takes are judged: press Assemble when you want a version';
    back.querySelector('.fix-hint').textContent = more && !note.value.trim() ? 'Say what the new shot shows and how long it lasts.'
      : scoped ? `The other shots, the song and the cast stay as they are. The showrunner does what you ask, changes nothing else, and ${ending}. If it needs more, it tells you first.`
      : redo || more ? `The showrunner may also change other shots, the song or the cast if that is what it takes, and tells you what it renders again.${review || makeVersion ? '' : ' It makes no new version: press Assemble when you want one.'}`
      : 'Click the shots concerned (Shift+click for a run of them), press + to insert a new shot, or just say what is wrong: the showrunner finds where it is.';
  };
  const close = () => back.remove();
  const submit = async () => {
    if (send.disabled) return;
    const text = note.value.trim(), where = time === null ? '' : `Around ${formatTime(time)} of the video: `, told = [];
    close();
    for (const after of [...added].sort((a, b) => doc.shots.findIndex(shot => shot.id === b) - doc.shots.findIndex(shot => shot.id === a))) {  // from the end: the positions before it hold
      const shots = state.doc.shots, position = shots.findIndex(shot => shot.id === after) + 1;
      const id = (await actions.ops([{ op: 'add_shots', position, shots: [newShot(state.doc, position)] }]))?.[0]?.shots?.[0];
      if (!id) return;
      picked.add(id);
      told.unshift(`${id} is a new shot the user inserted ${after ? `after ${after}` : 'at the start'}${shots[position] ? ` and before ${shots[position].id}` : ''}: write it from this note and shoot it.`);
    }
    const named = state.doc.shots.filter(shot => picked.has(shot.id)).map(shot => shot.id), message = [...told, text && where + text].filter(Boolean).join(' ');
    const options = { show: true, newVersion: version.hidden || makeVersion };
    if (named.length && keep.checked) actions.direct('reshoot', message, named, options);
    else actions.direct('message', `${named.length ? `${named.join(', ')}: ` : ''}${message}`, [], options);
  };
  back.addEventListener('click', event => {
    if (event.target === back || event.target.closest('[data-cancel]')) return close();
    if (event.target.closest('[data-send]')) return submit();
    const insert = event.target.closest('.fix-gap'), inserted = event.target.closest('.fix-new'), card = event.target.closest('.fix-shot');
    if (insert) added.push(insert.dataset.after || null);
    else if (inserted) added.splice(+inserted.dataset.new, 1);
    else if (card) {
      const id = card.dataset.shot, order = [...strip.querySelectorAll('.fix-shot[data-shot]:not(:disabled)')].map(item => item.dataset.shot);
      if (event.shiftKey && order.includes(anchor)) {
        const [first, last] = [order.indexOf(anchor), order.indexOf(id)].sort((a, b) => a - b);
        order.slice(first, last + 1).forEach(item => picked.add(item));
      } else if (picked.has(id)) picked.delete(id);
      else picked.add(id);
      anchor = id;
    } else return;
    refresh();
  });
  back.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.stopPropagation(); close(); }
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) submit();
  });
  note.addEventListener('input', refresh);
  keep.addEventListener('change', refresh);
  version.querySelector('input').addEventListener('change', event => { makeVersion = event.target.checked; refresh(); });
  refresh();
  document.getElementById('overlay-root').append(back);
  strip.querySelector('.fix-shot.on')?.scrollIntoView({ inline: 'center', block: 'nearest' });
  note.focus();
}
