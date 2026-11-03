// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// The Guide: the pages of docs/SHOWRUNNER.md, imported and fetched when it first opens (never part of the page), shown at the part about
// the page the user is on. A page title written in bold in another page links to it.
import { api } from './api.js';
import { icon, esc, h, toast } from './ui.js';

const ICONS = { Overview: 'sparkles', 'Quick Start': 'play', 'Starting a Production': 'brief', 'The Cast': 'users', 'Long Productions': 'layers',
  'Reviewing and Changing the Plan': 'grid', 'Takes and Verdicts': 'check', 'The Timeline': 'timeline', Models: 'bolt', 'Talking to the Showrunner': 'chat',
  Delivering: 'download', Versions: 'compare', 'Keyboard Shortcuts': 'keyboard', 'Where Productions Are Kept': 'bin' };
let pages = null;

export async function openGuide(title) {
  if (document.querySelector('.guide-back')) return;
  if (!pages) { pages = api.guide().then(result => result.pages); pages.catch(() => { pages = null; }); }  // a failed fetch is tried again next time
  let list;
  try { list = await pages; } catch (error) { toast(error.message, 'error'); return; }
  const back = h(`<div class="modal-back guide-back"><div class="modal guide" role="dialog" aria-modal="true" aria-label="Showrunner Guide">
    <div class="guide-head"><span class="brand-mark">${icon('logo')}</span><h3>Showrunner Guide</h3><span class="grow"></span><button class="btn ghost icon" data-close title="Close (Escape)">${icon('x')}</button></div>
    <div class="guide-body"><nav class="guide-nav">${list.map((page, index) => `<button data-page="${index}">${icon(ICONS[page.title] || 'book')}<span>${esc(page.title)}</span></button>`).join('')}</nav>
      <article class="guide-page"></article></div></div></div>`);
  const article = back.querySelector('.guide-page');
  const show = index => {
    back.querySelectorAll('nav [data-page]').forEach(button => button.classList.toggle('on', +button.dataset.page === index));
    article.innerHTML = `<h2>${esc(list[index].title)}</h2>${list[index].html}`;
    article.querySelectorAll('strong').forEach(strong => {
      const target = list.findIndex(page => page.title === strong.textContent);
      if (target >= 0 && target !== index) strong.outerHTML = `<a href="#" data-page="${target}">${strong.innerHTML}</a>`;
    });
    article.scrollTop = 0;
  };
  const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
  const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };  // the board behind keeps its selection
  back.addEventListener('click', event => {
    const page = event.target.closest('[data-page]');
    if (page) { event.preventDefault(); show(+page.dataset.page); }
    else if (event.target === back || event.target.closest('[data-close]')) close();
  });
  document.addEventListener('keydown', keys);
  document.getElementById('overlay-root').append(back);
  show(Math.max(0, list.findIndex(page => page.title === title)));
  back.querySelector('nav .on').focus();
}
