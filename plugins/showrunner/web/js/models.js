// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// The model pickers: a row per Deepy template category, named as in Deepy's settings with what the showrunner makes with it, listing the
// category's templates; Auto is the one selected in Deepy's settings. The brief's rows set a production's models; Default models, what new
// productions start with.
import { api } from './api.js';
import { esc, h, toast } from './ui.js';

// A template's name in the lists, with what it takes when the other templates of its category do not all take it.
export function choiceLabel(role, choice) {
  const flags = [['refs', 'keeps faces'], ['audio', 'lip sync']].filter(([key]) => choice[key] && !role.choices.every(item => item[key])).map(([, text]) => text).join(', ');
  return `${choice.label}${flags ? ` (${flags})` : ''}`;
}

// A shot's template before the lists load, or one they no longer hold: its name, or what Deepy's selection of its category makes.
export function templateLabel(template) {
  if (template === 'draw_svg') return 'Drawn by the showrunner (card, diagram, map)';
  const [task, ...name] = String(template || 'gen_video').split('/');
  return name.join('/') || { gen_video: 'Deepy\'s video template', gen_video_with_refs: 'Deepy\'s template for scenes with the cast', gen_video_with_speech: 'Deepy\'s lip-sync template',
    gen_image: 'Deepy\'s image template', edit_image: 'Deepy\'s image editing template' }[task] || `${task.replaceAll('_', ' ')} (Deepy's template)`;
}

export function modelRows(roles) {
  return roles.map(role => `<label class="model-row"><span><b>${esc(role.label)}</b><span class="muted">${esc(role.hint)}</span></span>
    <select class="select" data-role="${role.id}" title="${esc(role.label)}: ${esc(role.hint)}"><option value="">Auto${role.auto ? ` · ${esc(role.choices.find(choice => choice.value === role.auto)?.label || role.auto)}` : ''}</option>
      ${role.choices.map(choice => `<option value="${esc(choice.value)}" ${choice.value === role.value ? 'selected' : ''} title="${esc(choice.value)}">${esc(choiceLabel(role, choice))}</option>`).join('')}</select></label>`).join('');
}

export async function openDefaultModels() {
  let roles;
  try { roles = (await api.defaultTemplates()).roles; } catch (error) { toast(error.message, 'error'); return; }
  const back = h(`<div class="modal-back"><div class="modal default-models" role="dialog" aria-modal="true" aria-label="Default models"><h3>Default models</h3>
    <p>New productions start with these templates; a production's brief can change them for that production. Each list holds the category's templates from Deepy's settings, and <b>Auto</b> is the one selected there. To add your own, select your saved WanGP settings in the generator, then press + next to the category in <b>Ask Deepy &gt; Settings</b>.</p>
    <div class="models">${modelRows(roles)}</div><div class="actions"><button class="btn primary" data-close>Done</button></div></div></div>`);
  const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
  const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
  back.addEventListener('click', event => { if (event.target === back || event.target.closest('[data-close]')) close(); });
  back.querySelectorAll('[data-role]').forEach(select => select.onchange = async () => {
    const role = roles.find(item => item.id === select.dataset.role);
    try {
      roles = (await api.chooseDefaultTemplate(role.id, select.value || null)).roles;
      toast(`New productions: ${role.label} with ${select.value ? select.selectedOptions[0].textContent : 'Auto'}.`, 'ok');
    } catch (error) { toast(error.message, 'error'); select.value = role.value || ''; }
  });
  document.addEventListener('keydown', keys);
  document.getElementById('overlay-root').append(back);
  back.querySelector('[data-close]').focus();
}
