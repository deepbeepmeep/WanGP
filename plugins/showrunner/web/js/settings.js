// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Showrunner's settings, for every production: the options most productions do without.
import { api } from './api.js';
import { h, toast } from './ui.js';

const OPTIONS = [
  ['notify_turns', 'Tell me when a task ends', 'When the showrunner finishes a task or stops on an error, plays WanGP\'s notification sound on the computer running WanGP and sends what comes next to your notification destinations (Config > Notification), if you have set any. For long shoots, when this page is out of sight.'],
  ['cast_loras', 'LoRAs on cast cards', 'Adds a LoRA choice to each cast card, for a person or a thing you trained one for. A LoRA only works with the family of models it was trained on: the card lists those of the models this production uses and says how many of its shots can use it; the other shots are made without it. The models themselves, with their own LoRAs, are chosen in Default models and the brief.'],
  ['show_all_errors', 'Show every error', 'Lists in the Showrunner tab every call it got wrong and corrected by itself, with the exact message, for finding out what went wrong. Off, the tab shows what really failed.'],
  ['any_shot_length', 'Let the showrunner pick any shot length', 'Off, a shot without words lasts a length its model really makes (4.46, 5.17, 5.88 s and so on with MiniMax H3), so that every action plays to its end; a speaking shot still ends after its line, because the model would invent words after it, and a music video cuts on its song. On, the showrunner can cut faster: a shot plays the start of its video, and the end of its action can be lost.'],
  ['show_deepy_tab', 'Show the Deepy tab', 'Adds a tab to the panel with Deepy\'s own window on the production: its thoughts and every step, as they are written. You read it there, and write in the Showrunner tab.'],
];

export async function openSettings(changed) {
  let settings;
  try { settings = await api.settings(); } catch (error) { toast(error.message, 'error'); return; }
  const back = h(`<div class="modal-back"><div class="modal settings-modal" role="dialog" aria-modal="true" aria-label="Settings"><h3>Settings</h3>
    <div class="col" style="gap:10px">${OPTIONS.map(([key, title, text]) => `<label class="setting-row" data-setting="${key}"><span><b>${title}</b><span class="muted">${text}</span></span>
      <span class="toggle ${settings[key] ? 'on' : ''}" role="switch" aria-checked="${settings[key]}" aria-label="${title}" tabindex="0"></span></label>`).join('')}</div>
    <div class="actions"><button class="btn primary" data-close>Done</button></div></div></div>`);
  const close = () => { back.remove(); document.removeEventListener('keydown', keys); };
  const keys = event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
  back.addEventListener('click', event => { if (event.target === back || event.target.closest('[data-close]')) close(); });
  back.querySelectorAll('[data-setting]').forEach(row => {
    const key = row.dataset.setting, toggle = row.querySelector('.toggle');
    const flip = async () => {
      try {
        settings = await api.saveSettings({ [key]: !settings[key] });
        toggle.classList.toggle('on', settings[key]);
        toggle.setAttribute('aria-checked', settings[key]);
        changed();
      } catch (error) { toast(error.message, 'error'); }
    };
    row.addEventListener('click', event => { event.preventDefault(); flip(); });
    toggle.addEventListener('keydown', event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); flip(); } });
  });
  document.addEventListener('keydown', keys);
  document.getElementById('overlay-root').append(back);
  back.querySelector('.toggle').focus();
}
