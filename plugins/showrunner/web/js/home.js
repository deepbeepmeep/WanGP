// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Home: start a production from its kind, or open one.
import { api } from './api.js';
import { icon, esc, art, ago, formatDuration, menu, confirmDialog, promptDialog, toast } from './ui.js';

export const KINDS = {
  music_video: { label: 'Music video', icon: 'music', text: 'Cut to the beat of your song', hue: 'music' },
  short_film: { label: 'Short film', icon: 'clapper', text: 'Characters, scenes and dialogue', hue: 'film' },
  ad: { label: 'Ad', icon: 'megaphone', text: 'A product spot in 15 to 60 s', hue: 'ad' },
  trailer: { label: 'Trailer', icon: 'film', text: 'Tease a story in a minute', hue: 'trailer' },
  story: { label: 'Illustrated story', icon: 'book', text: 'Narrated picture-book pages', hue: 'story' },
  free: { label: 'Anything', icon: 'sparkles', text: 'Start from an idea', hue: 'free' },
};
const STAGE_LABELS = { brief: 'Brief', planning: 'Planning', review: 'In review', shooting: 'Shooting', editing: 'Editing', delivered: 'Delivered' };

export function kindTiles(selected = null) {
  return Object.entries(KINDS).map(([key, kind]) => `<button class="kind ${selected === key ? 'on' : ''}" data-kind="${key}">
    <span class="kind-art" style="${art(kind.hue, 'kind')}"></span><span class="kind-icon">${icon(kind.icon)}</span><b>${kind.label}</b><span>${kind.text}</span></button>`).join('');
}

export function renderHome(container, state, { open }) {
  const productions = state.catalog.filter(item => !item.archived), archived = state.catalog.filter(item => item.archived);
  const reason = state.directorInfo?.available ? '' : state.directorInfo?.reason;
  container.innerHTML = `<div class="home">
    <div class="hero">
      <div><h1>What are we <span class="gradient-text">making</span> today?</h1>
      <p>Pick a format. You write the brief, Deepy directs: it plans every shot, casts, shoots and edits, and you review and change anything on the storyboard and the timeline.</p></div>
      ${reason ? `<div class="banner" style="margin:0;max-width:420px">${icon('alert')}<span>${esc(reason)}</span></div>` : ''}
    </div>
    <div class="kinds">${kindTiles()}</div>
    <div class="section-title">${icon('layers')} Your productions <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">${productions.length || ''}</span></div>
    ${productions.length ? `<div class="productions">${productions.map(card).join('')}</div>` : `<div class="empty-state">${archived.length ? 'Every production is archived: open the list below to bring one back.' : 'No production yet. Choose a format above to start.'}</div>`}
    ${archived.length ? `<details class="archived-productions" ${state.archivedOpen ? 'open' : ''}><summary class="section-title">${icon('chevron', 'chev')} Archived <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">${archived.length}</span></summary>
      <div class="productions">${archived.map(card).join('')}</div></details>` : ''}
  </div>`;
  const folded = container.querySelector('.archived-productions');
  if (folded) folded.ontoggle = () => state.archivedOpen = folded.open;  // stays open when the list refreshes
  const archive = async (id, archived) => {
    try { await api.archive(id, archived); toast(archived ? 'Archived: it waits in the Archived list at the bottom.' : 'Back in your productions.', 'ok'); } catch (error) { toast(error.message, 'error'); }
  };
  const trash = async id => {
    if (await confirmDialog({ title: 'Delete this production?', text: 'Its folder moves to the productions trash, so it can still be restored by hand. To only hide it, archive it instead.', ok: 'Delete', danger: true })) {
      await api.trash(id);
      toast('Deleted (moved to the trash).');
    }
  };
  container.querySelectorAll('[data-kind]').forEach(button => button.onclick = async () => {
    const title = await promptDialog({ title: `New ${KINDS[button.dataset.kind].label.toLowerCase()}`, text: 'Its name (you can change it later, at the top of the page).',
      value: `New ${KINDS[button.dataset.kind].label.toLowerCase()}`, ok: 'Create' });
    if (title === null) return;  // a look at the formats creates nothing
    button.disabled = true;
    try {
      const doc = await api.create(title.trim() || `New ${KINDS[button.dataset.kind].label.toLowerCase()}`, button.dataset.kind);
      location.hash = `#/p/${doc.id}/brief`;
    } catch (error) { toast(error.message, 'error'); button.disabled = false; }
  });
  container.querySelectorAll('.prod-card').forEach(element => {
    const item = state.catalog.find(entry => entry.id === element.dataset.id);
    element.onclick = event => { if (!event.target.closest('.card-actions')) open(element.dataset.id); };
    element.querySelector('[data-archive]').onclick = () => archive(item.id, !item.archived);
    element.querySelector('[data-trash]').onclick = () => trash(item.id);
    element.querySelector('.menu-btn').onclick = event => {
      event.stopPropagation();
      menu(event.currentTarget, [
        { label: 'Open', icon: 'expand', action: () => open(element.dataset.id) },
        { label: 'Rename…', icon: 'note', action: async () => {
          const item = state.catalog.find(entry => entry.id === element.dataset.id);
          const title = await promptDialog({ title: 'Rename the production', value: item?.title || '', ok: 'Rename' });
          if (title?.trim()) await api.ops(element.dataset.id, [{ op: 'brief', fields: { title: title.trim() } }]);
        } },
        ...(state.catalog.find(entry => entry.id === element.dataset.id)?.latest ? [{ label: 'Download the latest version', icon: 'download', action: () => {
          const item = state.catalog.find(entry => entry.id === element.dataset.id);
          Object.assign(document.createElement('a'), { href: api.file(item.id, item.latest, true), download: item.latest.split('/').pop() }).click();
        } }] : []),
        { label: 'Duplicate', icon: 'copy', action: async () => { await api.duplicate(element.dataset.id); toast('Duplicated.', 'ok'); } },
        { label: item.archived ? 'Bring back from the archive' : 'Archive', icon: item.archived ? 'undo' : 'archive', action: () => archive(item.id, !item.archived) },
        '-',
        { label: 'Delete…', icon: 'trash', danger: true, action: () => trash(item.id) },
      ]);
    };
  });
}

function card(item) {
  const kind = KINDS[item.kind] || KINDS.free;
  const posters = item.posters || [];
  const cover = posters.length ? posters.slice(0, posters.length >= 4 ? 4 : 1).map(path => `<img loading="lazy" src="api/productions/${encodeURIComponent(item.id)}/file?path=${encodeURIComponent(path)}" alt="">`).join('')
    : `<div class="placeholder" style="${art(item.id, kind.hue)}">${icon(kind.icon)}</div>`;
  const progress = item.shots ? Math.round(100 * (item.approved || 0) / item.shots) : 0;
  return `<article class="prod-card" data-id="${esc(item.id)}">
    <div class="prod-cover ${posters.length >= 4 ? '' : 'single'}">${cover}<span class="tag stage-${item.changed_since_delivery ? 'editing' : item.stage} stage-tag">${item.changed_since_delivery ? 'Changed since delivery' : STAGE_LABELS[item.stage] || item.stage}</span>
      <div class="card-actions"><button class="btn small icon" data-archive title="${item.archived ? 'Bring back from the archive' : 'Archive: hide it from this page, keep everything'}">${icon(item.archived ? 'undo' : 'archive')}</button>
        <button class="btn small icon" data-trash title="Delete">${icon('trash')}</button><button class="btn small icon menu-btn" title="More">${icon('more')}</button></div></div>
    <div class="prod-body"><b>${esc(item.title)}</b>
      <div class="prod-meta">${icon(kind.icon)}<span>${kind.label}</span><span>·</span>${icon('clock')}<span>${formatDuration(item.duration_s)}</span><span>·</span><span>${item.shots} shots</span>${item.director === 'paused' ? `<span class="tag" style="margin-left:auto">${icon('pause')}Paused</span>` : item.director !== 'idle' ? `<span class="tag rendering" style="margin-left:auto">${icon('director')}Working</span>` : ''}</div>
      <div class="progress" title="${item.approved || 0} of ${item.shots} shots approved"><i style="width:${progress}%"></i></div>
      <div class="prod-meta"><span>${item.approved || 0}/${item.shots} approved</span><span style="margin-left:auto">${ago(item.updated)}</span></div>
    </div></article>`;
}
