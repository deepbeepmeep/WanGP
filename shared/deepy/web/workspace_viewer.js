// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
/* Full workspace management is available only in the native Gradio interface. */
(() => {
  const icons = {search:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>', close:'<path d="m6 6 12 12M6 18 18 6"/>', import:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>', eject:'<path d="m12 3 8 11H4l8-11ZM4 20h16"/>', delete:'<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>', copy:'<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>', archive:'<path d="M3 3h18v5H3zM5 8v13h14V8M10 12h4"/>'};
  const svg = key => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg>`;
  icons.broom = '<path d="m20 3-8 10m-3-2 6 5-5 6-8-7 7-4Zm-3 5 4 4"/>';
  icons.lock = '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4M12 14v3"/>';
  icons.unlock = '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0M12 14v3"/>';
  icons.create = '<path d="M12 5v14M5 12h14"/>';
  icons.move = '<path d="M14 3H3v18h11M9 12h12m-5-5 5 5-5 5"/>';
  icons.first = '<path d="m11 19-9-7 9-7v14Zm10 0-9-7 9-7v14Z"/>';
  icons.last = '<path d="m13 5 9 7-9 7V5ZM3 5l9 7-9 7V5Z"/>';
  icons.fullscreen = '<path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>';
  icons.minimize = '<path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/>';
  class WorkspaceViewer {
    constructor(picker, transport) {
      this.picker = picker; this.transport = transport; this.source = 'video'; this.selections = {}; this.requestId = 0;
      const button = document.createElement('button'); button.type = 'button'; button.title = button.ariaLabel = 'Explore workspace'; button.innerHTML = svg('search');
      picker.node.append(button); button.onclick = event => {event.stopPropagation(); this.open();};
      this.dialog = document.createElement('dialog'); this.dialog.className = 'wangp-workspace-viewer'; this.dialog.ariaLabel = 'Workspace media viewer';
      this.dialog.innerHTML = `<header><div class="wv-heading"><div class="wv-workspace-picker"><select data-workspace aria-label="Workspace"></select><button data-create aria-label="Add workspace" title="Add workspace">${svg('create')}</button></div></div><div class="wv-header-actions"><button data-retention aria-label="Automatic workspace archiving" title="Automatic workspace archiving">${svg('broom')}</button><button data-close aria-label="Close workspace viewer" title="Close (Esc)">${svg('close')}</button></div><div class="wv-meta"><span data-summary></span><div class="wv-storage"><strong data-storage-label></strong><code data-folder></code><button data-folder-change hidden>Change Folder</button></div></div></header>
        <div class="wv-toolbar"><div role="tablist" data-sources aria-label="Workspace media"></div><span class="wv-spacer"></span><span data-activity></span><button data-protect aria-label="Protect workspace from automatic archiving" aria-pressed="false">${svg('unlock')}</button><button data-import>${svg('import')}Import</button><button data-refresh>Refresh</button><input data-files type="file" multiple hidden></div>
        <div class="wv-selection"><span data-count></span><button data-page-select>Select page</button><button data-clear>Clear selection</button><div data-actions hidden><button data-action="eject">${svg('eject')}Eject</button><button data-action="delete">${svg('delete')}Delete files</button><button data-action="copy">${svg('copy')}Copy to workspace</button><button data-action="move">${svg('move')}Move to workspace</button><button data-action="archive">${svg('archive')}ZIP</button><button data-action="first" title="Move selected media to the oldest end">${svg('first')}Move to start</button><button data-action="last" title="Move selected media to the newest end">${svg('last')}Move to end</button></div></div>
        <p class="wv-notice" role="status" hidden></p><div class="wv-content"><div class="wv-gallery"><section class="wv-browser" aria-label="Workspace media grid"><div class="wv-grid" role="listbox" aria-multiselectable="true" aria-label="Media"></div><div class="wv-empty" hidden>No media in this gallery. Import files to get started.</div><div class="wv-rubber" hidden></div></section>
        <footer><span>Oldest → newest · Ctrl/⌘ click or drag on empty space to select · Drag tiles to reorder</span><div class="wv-pagination"><button data-prev aria-label="Older page">←</button><label>Page <input data-page type="number" min="1" aria-label="Page number"></label><span data-pages></span><button data-next aria-label="Newer page">→</button></div></footer></div>
        <aside><h3>Media details</h3><div class="wv-detail-panes"><div class="wv-preview"></div><div class="wv-divider" role="separator" tabindex="0" aria-label="Resize media preview and settings summary" aria-orientation="horizontal" aria-valuemin="10" aria-valuemax="80" aria-valuenow="34" title="Drag to resize · Arrow keys to adjust · Double-click to reset"></div><iframe title="Generation properties" sandbox=""></iframe></div><button data-extract-settings disabled title="Load the selected media's generation settings and close the workspace manager">Extract Settings</button></aside></div>`;
      document.body.append(this.dialog);
      this.grid = this.dialog.querySelector('.wv-grid'); this.browser = this.dialog.querySelector('.wv-browser');
      this.info = this.dialog.querySelector('iframe'); this.preview = this.dialog.querySelector('.wv-preview');
      this.modal = document.createElement('dialog'); this.modal.className = 'wangp-workspace-dialog';
      this.modal.innerHTML = '<form><h2></h2><div class="wangp-workspace-body"><p></p><label data-name-label><span data-name-caption>Archive name</span><input maxlength="120" required></label><label data-target-label><span data-select-label>Target workspace</span><select required></select></label><div class="wangp-workspace-error" role="alert" hidden></div><div class="wangp-workspace-actions"><button type="button" data-cancel>Cancel</button><button type="submit">Confirm</button></div></div></form>';
      document.body.append(this.modal);
      this.modal.querySelector('[data-cancel]').onclick = () => this.modal.close();
      this.modal.addEventListener('close', () => document.dispatchEvent(new Event('wangp-viewer-scope')));
      this.modal.querySelector('form').onsubmit = event => {event.preventDefault(); this.confirm();};
      const storageButton = document.createElement('button'); storageButton.type = 'button'; storageButton.dataset.storage = ''; storageButton.textContent = 'Storage'; storageButton.title = 'Inspect or Convert Workspace Storage';
      this.dialog.querySelector('.wv-header-actions').prepend(storageButton);
      storageButton.onclick = () => this.storage();
      this.dialog.querySelector('[data-folder-change]').onclick = () => this.storage(true);
      this.dialog.querySelector('[data-close]').onclick = () => this.close();
      this.dialog.querySelector('[data-extract-settings]').onclick = () => {
        if (this.busy || this.loading || this.selected.size !== 1) return;
        window.__wangpAssistantChatNS.setBridgeValue('#wangp-workspace-extract-settings textarea', JSON.stringify({workspace: this.workspace, source: this.source, revision: this.state.revision, keys: [...this.selected], request: Date.now()}));
        this.close();
      };
      this.dialog.querySelector('[data-workspace]').onchange = event => this.changeWorkspace(event.target.value);
      this.dialog.querySelector('[data-create]').onclick = () => this.picker.open('create');
      this.picker.dialog.addEventListener('close', () => this.invalidate());
      this.dialog.querySelector('[data-protect]').onclick = () => this.protect();
      this.dialog.querySelector('[data-retention]').onclick = () => this.action('retention').catch(error => this.notice(error.message));
      this.dialog.onclose = () => {this.requestId++; this.detailRequest?.abort(); this.grid.replaceChildren(); this.hidePreview(); this.modal.close(); clearTimeout(this.refreshTimer); clearTimeout(this.pageHover); cancelAnimationFrame(this.rubberFrame);};
      this.dialog.addEventListener('cancel', () => {this.hidePreview();});
      this.dialog.addEventListener('close', () => document.dispatchEvent(new Event('wangp-viewer-scope')));
      this.dialog.querySelector('[data-import]').onclick = () => this.dialog.querySelector('[data-files]').click();
      this.dialog.querySelector('[data-files]').onchange = event => this.importFiles([...event.target.files]);
      this.dialog.querySelector('[data-refresh]').onclick = () => this.load(this.state?.page || 0);
      this.dialog.querySelector('[data-prev]').onclick = () => this.load(this.state.page - 1);
      this.dialog.querySelector('[data-next]').onclick = () => this.load(this.state.page + 1);
      this.dialog.querySelector('[data-page]').onchange = event => this.load(Number(event.target.value) - 1);
      this.dialog.querySelector('[data-page-select]').onclick = () => {this.state.items.forEach(item => this.selected.add(item.key)); this.selectionChanged();};
      this.dialog.querySelector('[data-clear]').onclick = () => {this.selected.clear(); this.selectionChanged();};
      this.dialog.querySelectorAll('[data-action]').forEach(button => button.onclick = () => this.action(button.dataset.action).catch(error => this.notice(error.message)));
      this.grid.onclick = event => {
        if (this.suppressClick) {this.suppressClick = false; return;}
        const tile = event.target.closest('[data-key]');
        if (tile) this.choose(tile.dataset.key, event);
      };
      this.grid.onkeydown = event => {
        if (event.target.closest('button')) return;
        if (event.key === ' ' || event.key === 'Enter') {const tile = event.target.closest('[data-key]'); if (tile) {event.preventDefault(); this.choose(tile.dataset.key, event);}}
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {event.preventDefault(); this.state.items.forEach(item => this.selected.add(item.key)); this.selectionChanged();}
      };
      this.installDrag(); this.installRubber(); this.installDivider();
      const suspend = () => {
        if (!this.dialog.open) return;
        const modal = document.querySelector('dialog[open]:focus-within') || [...document.querySelectorAll('dialog[open]')].at(-1);
        if (document.hidden || !document.hasFocus() || (modal && modal !== this.dialog)) this.stopPreview();
      };
      document.addEventListener('visibilitychange', suspend);
      window.addEventListener('blur', () => queueMicrotask(suspend));
      window.addEventListener('pagehide', () => this.stopPreview());
      document.addEventListener('click', suspend);
    }
    get selected() {return this.selections[this.source] ??= new Set();}
    // One tab per gallery of the workspace, in the order and with the import filters published by the server.
    renderSources(sources) {
      const list = this.dialog.querySelector('[data-sources]');
      if (list.dataset.ids === sources.map(source => source.id).join()) return;
      list.dataset.ids = sources.map(source => source.id).join();
      list.replaceChildren(...sources.map(source => {
        const tab = document.createElement('button'); tab.setAttribute('role', 'tab'); tab.dataset.source = source.id; tab.textContent = source.label;
        tab.onclick = () => {if (this.source !== source.id) {this.source = source.id; this.detailKey = null; this.clearPreview(); this.info.srcdoc = WanGPMediaView.documentHtml('Select media to view its properties.'); this.load(this.pages[this.source] ?? 0);}};
        return tab;
      }));
      this.dialog.querySelector('[data-files]').accept = sources.map(source => source.accept).join(',');
    }
    query(path, extra = {}) {return path + '?' + new URLSearchParams({workspace: this.workspace, source: this.source, ...extra});}
    notice(text) {const node = this.dialog.querySelector('.wv-notice'); node.textContent = text; node.hidden = !text;}
    navigation() {
      this.dialog.querySelector('[data-prev]').disabled = this.busy || this.loading || !this.state || this.state.page === 0;
      this.dialog.querySelector('[data-next]').disabled = this.busy || this.loading || !this.state || this.state.page + 1 === this.state.pages;
      this.dialog.querySelector('[data-page]').disabled = this.busy || this.loading || !this.state;
      this.dialog.querySelector('[data-protect]').disabled = this.busy || this.loading || !this.state;
      this.dialog.querySelector('[data-extract-settings]').disabled = this.busy || this.loading || !this.state || this.selected.size !== 1;
    }
    setBusy(value) {
      this.busy = value;
      this.dialog.querySelectorAll('button:not([data-close])').forEach(button => button.disabled = value);
      this.dialog.querySelector('[data-workspace]').disabled = value;
      if (value) this.dialog.setAttribute('aria-busy', 'true'); else this.dialog.removeAttribute('aria-busy');
      this.navigation();
    }
    open() {
      this.dialog.showModal();
      if (this.previewItem?.kind === 'app' && this.workspace === this.picker.state.selected && this.state) this.load(this.page);
      else this.resetWorkspace();
      document.dispatchEvent(new Event('wangp-viewer-scope'));
    }
    close() {
      this.hidePreview();
      this.dialog.close();
      document.dispatchEvent(new Event('wangp-viewer-scope'));
    }
    renderWorkspaces() {
      const select = this.dialog.querySelector('[data-workspace]'), choices = JSON.stringify(this.picker.state.items);
      if (this.choices !== choices) {select.replaceChildren(...this.picker.state.items.map(item => new Option(WanGPWorkspacePicker.label(item), item.id))); this.choices = choices;}
      select.value = this.picker.state.selected;
      select.title = select.selectedOptions[0].textContent;
    }
    renderActivity(state) {
      this.dialog.querySelector('[data-storage]').textContent = state.storage_mode === 'dedicated' ? 'Folder Storage' : 'Shared Storage';
      this.dialog.querySelector('[data-storage]').title = state.folder || 'Shared Output Folders';
      this.dialog.querySelector('[data-storage-label]').textContent = state.storage_mode === 'dedicated' ? 'Dedicated folder' : 'Shared references · Files can be in multiple folders';
      this.dialog.querySelector('[data-folder]').textContent = this.dialog.querySelector('[data-folder]').title = state.folder || '';
      this.dialog.querySelector('[data-folder]').hidden = !state.folder;
      this.dialog.querySelector('[data-folder-change]').hidden = !state.folder;
      const date = new Date(state.last_activity * 1000), activity = this.dialog.querySelector('[data-activity]');
      activity.textContent = 'Last Activity: ' + date.toLocaleString([], {dateStyle:'medium', timeStyle:'short'}); activity.title = date.toLocaleString();
      const button = this.dialog.querySelector('[data-protect]');
      button.innerHTML = svg(state.archive_protected ? 'lock' : 'unlock');
      button.setAttribute('aria-pressed', String(state.archive_protected));
      button.title = state.archive_protected ? 'Protected from automatic archiving — click to unlock' : 'Allow automatic archiving — click to protect';
    }
    resetWorkspace() {
      clearTimeout(this.refreshTimer); this.modal.close();
      this.workspace = this.picker.state.selected; this.selections = {}; this.state = null; this.detailKey = null;
      this.pages = {};
      this.grid.replaceChildren(); this.selectionChanged();
      this.renderWorkspaces(); this.dialog.querySelector('[data-activity]').textContent = '';
      this.dialog.querySelector('[data-summary]').textContent = '';
      this.dialog.querySelector('[data-prev]').disabled = this.dialog.querySelector('[data-next]').disabled = true;
      this.notice('Loading workspace…'); this.load(0, false, true);
    }
    async changeWorkspace(id) {
      this.renderWorkspaces(); this.setBusy(true);
      try {this.picker.render(await this.transport.request('workspaces/select', {id, context:'gallery'})); this.invalidate();}
      catch (error) {this.notice(error.message);}
      finally {this.setBusy(false);}
    }
    async protect() {
      this.setBusy(true);
      try {await this.transport.request('workspace_viewer/protection', {workspace:this.workspace, protected:!this.state.archive_protected}); await this.load(this.page, true);}
      catch (error) {this.notice(error.message);}
      finally {this.setBusy(false);}
    }
    async storage(relocate = false) {
      if (this.busy || !this.state) return;
      this.setBusy(true);
      try {
        const mode = relocate ? 'dedicated' : this.state.storage_mode === 'dedicated' ? 'shared' : 'dedicated';
        const preview = await this.transport.request('workspace_viewer/storage', {workspace:this.workspace, mode});
        this.modal.dataset.action = relocate ? 'folder' : 'storage'; this.modal.dataset.storageMode = mode; this.modal.dataset.workspace = this.workspace;
        delete this.modal.dataset.targetFolder;
        this.modal.querySelector('h2').textContent = relocate ? 'Change Workspace Folder' : mode === 'dedicated' ? 'Convert to Dedicated Folder' : 'Convert to Shared References';
        this.modal.querySelector('p').textContent = `${preview.media_count} media, ${preview.file_count} files, ${(preview.bytes / 1024 / 1024).toFixed(1)} MB. ${mode === 'dedicated' ? 'Copy these files into an independent workspace folder. Originals stay in place.' : 'Keep files in their current folder and store absolute references. Future outputs use the shared output folders.'}${this.state.folder ? '\nCurrent folder: ' + this.state.folder : ''}${preview.missing.length ? '\nRestore missing files before converting: ' + preview.missing.join(', ') : ''}`;
        this.modal.querySelector('[data-name-label]').hidden = !relocate; this.modal.querySelector('input').disabled = !relocate;
        this.modal.querySelector('[data-name-caption]').textContent = 'New folder on the server';
        this.modal.querySelector('input').maxLength = 32767;
        this.modal.querySelector('input').value = relocate ? this.state.folder : '';
        this.modal.querySelector('[data-target-label]').hidden = true; this.modal.querySelector('select').disabled = true;
        this.modal.querySelector('[role=alert]').hidden = true;
        this.modal.querySelector('[type=submit]').textContent = relocate ? 'Preview Copy' : 'Convert'; this.modal.querySelector('[type=submit]').disabled = !!preview.missing.length;
        this.stopPreview(); this.modal.showModal(); document.dispatchEvent(new Event('wangp-viewer-scope'));
      } catch (error) {this.notice(error.message);}
      finally {this.setBusy(false);}
    }
    invalidate() {
      if (!this.dialog.open) return;
      this.renderWorkspaces();
      if (this.workspace !== this.picker.state.selected) {this.resetWorkspace(); return;}
      if (!this.state) return;
      clearTimeout(this.refreshTimer); this.refreshTimer = setTimeout(() => this.load(this.page || 0, true), 150);
    }
    async load(page, keepScroll = false, initial = false) {
      clearTimeout(this.refreshTimer);
      if (page !== this.page) {this.detailRequest?.abort(); this.clearPreview();}
      const request = ++this.requestId;
      this.page = this.pages[this.source] = page; this.loading = true; this.navigation();
      const requestedSelection = new Set(this.selected);
      try {
        const state = await this.transport.request('workspace_viewer', {workspace: this.workspace, source: this.source, page: Math.max(0, page), selected: [...this.selected], initial});
        if (request !== this.requestId || !this.dialog.open) return;
        state.items.forEach(item => {if (item.thumbnail) item.thumbnail = new URL(item.thumbnail, this.transport.base).href;});
        this.pages[state.source] = state.page;
        this.renderSources(state.sources);
        this.renderActivity(state);
        if (keepScroll && this.state?.revision === state.revision && this.state.source === state.source && this.state.page === state.page) {this.state = state; this.selectionChanged(); return;}
        const retained = new Set(state.selected);
        if (initial) {this.source = state.source; this.selections[this.source] = retained; this.anchor = state.selected[0];}
        else requestedSelection.forEach(key => {if (!retained.has(key)) this.selected.delete(key);});
        this.state = state; this.page = state.page;
        const range = state.items.length ? `${state.items[0].index + 1}–${state.items[state.items.length - 1].index + 1} of ${state.total} media` : '0 media';
        this.dialog.querySelector('[data-summary]').textContent = `${range} · ${state.visible} visible in the main gallery`;
        this.dialog.querySelectorAll('[data-source]').forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.source === state.source)));
        this.grid.replaceChildren(...state.items.map(item => this.tile(item)));
        this.dialog.querySelector('.wv-empty').hidden = state.total > 0;
        this.dialog.querySelector('[data-page]').value = state.page + 1; this.dialog.querySelector('[data-page]').max = state.pages;
        this.dialog.querySelector('[data-pages]').textContent = '/ ' + state.pages;
        this.navigation();
        if (!keepScroll) this.browser.scrollTop = 0;
        this.notice(''); this.selectionChanged();
        if (initial && this.anchor) {
          this.grid.querySelector('[aria-selected="true"]').scrollIntoView({block: 'nearest'});
          this.showDetails(this.anchor);
        }
      } catch (error) {if (request === this.requestId) this.notice(error.message);}
      finally {if (request === this.requestId) {this.loading = false; this.navigation();}}
    }
    tile(item) {
      const tile = document.createElement('article'); tile.className = 'wv-tile'; tile.dataset.key = item.key; tile.dataset.visible = String(item.visible); tile.draggable = true; tile.tabIndex = 0; tile.setAttribute('role', 'option'); tile.setAttribute('aria-label', item.name);
      const thumb = document.createElement('div'); thumb.className = 'wv-thumb';
      if (!item.url) thumb.textContent = 'File missing';
      else if (item.kind === 'audio') {thumb.innerHTML = '<svg viewBox="0 0 120 70" aria-hidden="true"><path d="M12 30v10m12-24v38m12-44v50m12-37v24m12-42v60m12-50v40m12-30v20m12-38v56m12-39v22"/></svg>';}
      else {const image = WanGPMediaView.thumbnail(item); image.draggable = false; image.onerror = () => {image.remove(); thumb.prepend(document.createTextNode('Preview unavailable'));}; thumb.append(image);}
      if (item.url && item.kind === 'video') thumb.append(WanGPMediaView.playButton(() => {
        this.choose(item.key, {});
        this.play(item);
      }));
      const name = document.createElement('div'); name.className = 'wv-name'; name.textContent = item.name; name.title = item.name;
      const badge = document.createElement('span'); badge.className = 'wv-badge'; badge.textContent = item.visible ? 'In gallery' : 'Stored';
      const index = document.createElement('span'); index.className = 'wv-index'; index.textContent = `${item.index + 1} · ${item.kind}`;
      tile.append(thumb, name, index, badge); return tile;
    }
    choose(key, event) {
      const keys = this.state.items.map(item => item.key), multi = event.ctrlKey || event.metaKey;
      if (event.shiftKey && keys.includes(this.anchor)) {
        if (!multi) this.selected.clear();
        const a = keys.indexOf(this.anchor), b = keys.indexOf(key); keys.slice(Math.min(a,b), Math.max(a,b)+1).forEach(id => this.selected.add(id));
      } else if (multi) {if (this.selected.has(key)) this.selected.delete(key); else this.selected.add(key);}
      else {this.selected.clear(); this.selected.add(key);}
      this.anchor = key; this.selectionChanged(); this.showDetails(key);
    }
    selectionChanged() {
      this.navigation();
      this.grid.querySelectorAll('[data-key]').forEach(tile => tile.setAttribute('aria-selected', String(this.selected.has(tile.dataset.key))));
      this.dialog.querySelector('[data-count]').textContent = `${this.selected.size} selected`;
      this.dialog.querySelector('[data-actions]').hidden = !this.selected.size;
      this.dialog.querySelector('[data-clear]').hidden = !this.selected.size;
      if (!this.selected.size) {this.detailRequest?.abort(); this.detailKey = null; this.clearPreview(); this.info.srcdoc = WanGPMediaView.documentHtml('Select media to view its properties.');}
    }
    clearPreview() {
      this.playerRequest?.abort(); this.playerRequest = null;
      window.WanGPAppPlayer?.closeOwner(this.preview);
      this.preview.querySelectorAll('video, audio').forEach(media => {media.pause(); media.removeAttribute('src'); media.load();});
      WanGPImageZoom.destroy(this.preview.querySelector('img'));
      this.preview.replaceChildren();
      this.previewItem = null;
    }
    hidePreview() {
      this.playerRequest?.abort(); this.playerRequest = null;
      if (this.previewItem?.kind !== 'app') this.clearPreview();
    }
    stopPreview() {
      const item = this.previewItem;
      if (!item) return;
      // Apps keep their browsing context while another scope is active.
      if (item.kind === 'app') { window.WanGPAppPlayer?.refresh(); return; }
      if (!['3d', 'world', 'app'].includes(item.kind)) {
        this.preview.querySelectorAll('video, audio').forEach(media => media.pause());
        return;
      }
      this.clearPreview();
      this.showPoster(item);
    }
    showPoster(item) {
      this.previewItem = item;
      let preview = WanGPMediaView.thumbnail(item);
      if (item.kind !== 'image') {
        const play = document.createElement('button'); play.type = 'button'; play.className = 'wv-open';
        const label = document.createElement('span'); label.textContent = item.kind === '3d' ? '▶ View 3D' : item.kind === 'world' ? '▶ Open World' : '▶ Play';
        play.append(preview, label); play.onclick = () => this.play(item); preview = play;
      }
      this.preview.append(preview, this.previewActions(item));
      if (item.kind === 'image') WanGPImageZoom.mount(preview, {viewport: this.preview, source: new URL(item.url, location.href).href});
    }
    async showDetails(key) {
      const item = this.state.items.find(item => item.key === key); if (!item) return;
      const identity = this.workspace + this.source + key;
      if (item.kind === 'app' && this.detailKey === identity && this.preview.querySelector('.wangp-app-player__frame')) return;
      this.detailKey = identity;
      this.detailRequest?.abort();
      const request = this.detailRequest = new AbortController();
      this.clearPreview();
      if (item.url) this.showPoster(item);
      this.info.srcdoc = WanGPMediaView.documentHtml('Loading media information…');
      try {const result = await this.transport.request(this.query('workspace_viewer/info', {key}), undefined, request.signal); if (!request.signal.aborted && this.detailKey === identity) this.info.srcdoc = WanGPMediaView.documentHtml(result.html);}
      catch (error) {if (!request.signal.aborted && this.detailKey === identity) this.notice(error.message);}
    }
    previewActions(item) {
      const actions = document.createElement('div'); actions.className = 'wv-preview-actions'; actions.setAttribute('role', 'group'); actions.setAttribute('aria-label', 'Preview actions');
      const link = document.createElement('a'), url = new URL(item.url, location.href);
      url.searchParams.set('download', 'true');
      link.href = url.href; link.download = item.name; link.title = 'Download'; link.setAttribute('aria-label', 'Download ' + item.name);
      // Match the download icon in Gradio's gallery; download the original file.
      link.innerHTML = '<svg viewBox="0 0 32 32" aria-hidden="true"><path fill="currentColor" stroke="none" d="M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10l10-10z"/></svg>';
      actions.append(link);
      if (item.kind === 'image' || item.kind === 'video') {
        const button = document.createElement('button'); button.type = 'button';
        const update = () => {
          const fullscreen = document.fullscreenElement === this.preview;
          button.title = button.ariaLabel = fullscreen ? 'Exit full screen' : 'View in full screen';
          button.innerHTML = svg(fullscreen ? 'minimize' : 'fullscreen');
        };
        this.preview.onfullscreenchange = update; update();
        button.onclick = () => {
          if (document.fullscreenElement === this.preview) document.exitFullscreen().catch(error => this.notice(error.message));
          else {this.loadPreview(item); this.preview.requestFullscreen().catch(error => this.notice(error.message));}
        };
        actions.append(button);
      }
      return actions;
    }
    loadPreview(item) {
      let media = this.preview.firstElementChild;
      if (item.kind === 'image') {
        media.classList.add('wv-media'); media.loading = 'eager'; media.src = item.url;
        return media;
      }
      if (!media.classList.contains('wv-media')) {
        const original = WanGPMediaView.media(item); original.className = 'wv-media';
        media.replaceWith(original); media = original;
      }
      return media;
    }
    async play(item) {
      if (['3d', 'world', 'app'].includes(item.kind)) {
        this.playerRequest?.abort();
        const request = this.playerRequest = new AbortController();
        try {
          const result = await this.transport.request(this.query('workspace_viewer/player', {key: item.key}), undefined, request.signal);
          if (request.signal.aborted || !this.dialog.open || this.previewItem !== item || document.hidden || !document.hasFocus()) return;
          const template = document.createElement('template'); template.innerHTML = result.html;
          const mesh = template.content.querySelector('wangp-mesh-viewer');
          if (mesh) mesh.dataset.preview = item.thumbnail;
          const app = template.content.querySelector('.app-viewer-play');
          if (app) {app.querySelector('img, .app-viewer-icon')?.remove(); app.prepend(WanGPMediaView.thumbnail(item));}
          // Use the gallery's own controls, with playback owned by this preview.
          template.content.querySelectorAll('video').forEach(media => {media.autoplay = false; media.controls = true; media.preload = 'metadata'; if (!media.getAttribute('src')) media.removeAttribute('src');});
          this.preview.replaceChildren(template.content);
          if (item.kind === 'app') {
            const link = this.preview.querySelector('[data-app-id]');
            if (link) {link.onclick = event => {event.preventDefault(); event.stopPropagation(); window.WanGPAppPlayer.open(link);}; link.click();}
          }
        } catch (error) {if (!request.signal.aborted) this.notice(error.message);}
        return;
      }
      const media = this.loadPreview(item);
      if (item.kind === 'video' || item.kind === 'audio') media.play().catch(() => {});
    }
    async run(action, extra = {}) {
      if (this.busy) return false;
      this.setBusy(true);
      try {
        const result = await this.transport.request('workspace_viewer/' + action, {workspace: this.workspace, source: this.source, revision: this.state.revision, keys: [...this.selected], ...extra});
        if (action === 'archive') {const link = document.createElement('a'); link.href = result.url; link.download = result.filename; document.body.append(link); link.click(); link.remove();}
        if (['eject', 'delete', 'move'].includes(action)) this.selected.clear();
        await this.load(this.page, true);
        this.notice(result.errors?.length ? result.errors.join(' · ') : action === 'copy' ? `${result.count} media added to the target workspace.` : action === 'move' ? `${result.count} media moved to the target workspace.` : '');
        return true;
      } catch (error) {this.notice(error.message); if (this.modal.open) {const node = this.modal.querySelector('[role=alert]'); node.hidden = false; node.textContent = error.message;} await this.load(this.state.page, true); this.notice(error.message); return false;}
      finally {this.setBusy(false);}
    }
    async action(action) {
      if (this.busy || (action !== 'retention' && !this.selected.size)) return;
      let retention;
      if (action === 'retention') {
        this.setBusy(true);
        try {retention = await this.transport.request('workspace_viewer/retention');}
        finally {this.setBusy(false);}
        if (!this.dialog.open) return;
      }
      if (action === 'eject') return this.run(action);
      if (action === 'last') return this.run('reorder', {before: null});
      if (action === 'first') return this.run('reorder', {edge: 'start'});
      this.modal.dataset.action = action;
      this.modal.querySelector('h2').textContent = {delete:'Delete files permanently',copy:'Copy to workspace',move:'Move to workspace',archive:'Download ZIP archive',retention:'Automatic workspace archiving'}[action];
      this.modal.querySelector('p').textContent = action === 'retention' ? 'At the next WanGP startup, archive unlocked workspaces with no media additions, removals or reordering during this period. Their dedicated Deepy sessions are archived too. Media files stay in place. Opening or renaming a workspace does not reset its activity date.' : action === 'delete' ? `Permanently delete ${this.selected.size} selected media files from disk? They will also be removed from every workspace that references them. This cannot be undone.` : action === 'copy' ? 'Append the selected media to another workspace. The original files are shared, without copying them on disk.' : 'Choose the name of the ZIP to download.';
      if (['copy', 'move'].includes(action)) this.modal.querySelector('p').textContent = 'Dedicated destinations receive independent file copies; shared destinations receive references. Original files remain on disk.' + (action === 'move' ? ' After success, remove these entries from the source gallery.' : '');
      this.modal.querySelector('[data-name-label]').hidden = action !== 'archive'; this.modal.querySelector('input').disabled = action !== 'archive';
      this.modal.querySelector('[data-name-caption]').textContent = 'Archive name'; this.modal.querySelector('input').maxLength = 120;
      this.modal.querySelector('input').value = action === 'archive' ? this.state.name + '.zip' : '';
      const target = this.modal.querySelector('select'); target.disabled = !['copy', 'move', 'retention'].includes(action); target.closest('label').hidden = target.disabled;
      this.modal.querySelector('[data-select-label]').textContent = action === 'retention' ? 'Archive after inactivity of' : 'Target workspace';
      target.replaceChildren(...(retention ? retention.choices.map(([label, days]) => new Option(label, days)) : this.picker.state.items.filter(item => item.id !== this.workspace).map(item => new Option(WanGPWorkspacePicker.label(item), item.id))));
      if (retention) target.value = retention.days;
      this.modal.querySelector('[role=alert]').hidden = true;
      this.modal.querySelector('[type=submit]').textContent = action === 'retention' ? 'Save' : action === 'delete' ? 'Delete permanently' : action === 'copy' ? 'Copy' : action === 'move' ? 'Move' : 'Download';
      this.modal.querySelector('[type=submit]').disabled = ['copy', 'move'].includes(action) && !target.options.length;
      this.stopPreview(); this.modal.showModal(); document.dispatchEvent(new Event('wangp-viewer-scope')); this.modal.querySelector(action === 'delete' ? '[data-cancel]' : action === 'archive' ? 'input' : 'select').focus();
    }
    async confirm() {
      const button = this.modal.querySelector('[type=submit]'); button.disabled = true;
      if (['storage', 'folder'].includes(this.modal.dataset.action)) {
        this.setBusy(true);
        try {
          const relocation = this.modal.dataset.action === 'folder';
          const payload = {workspace:this.modal.dataset.workspace, mode:this.modal.dataset.storageMode};
          if (relocation && !this.modal.dataset.targetFolder) {
            const preview = await this.transport.request('workspace_viewer/storage', {...payload, folder:this.modal.querySelector('input').value.trim()});
            if (preview.missing.length) throw new Error('Restore missing files before changing folders: ' + preview.missing.join(', '));
            this.modal.dataset.targetFolder = preview.folder;
            this.modal.querySelector('input').value = preview.folder; this.modal.querySelector('input').disabled = true;
            this.modal.querySelector('p').textContent = `Copy and verify ${preview.media_count} media, ${preview.file_count} files (${(preview.bytes / 1024 / 1024).toFixed(1)} MB) into this folder, then use it for future outputs. Originals stay in place. Existing files are not overwritten or automatically added to the gallery. Agents assigned to this workspace will have access to this folder's contents.`;
            this.modal.querySelector('[role=alert]').hidden = true;
            button.textContent = 'Copy and Use Folder';
          } else {
            await this.transport.request('workspace_viewer/storage', {...payload, ...(relocation ? {folder:this.modal.dataset.targetFolder} : {}), confirmed:true});
            this.modal.close(); await this.load(this.page || 0); this.notice(relocation ? 'Workspace folder changed. Original files were retained.' : 'Workspace storage converted. Original files were retained.');
          }
        } catch (error) {const node = this.modal.querySelector('[role=alert]'); node.hidden = false; node.textContent = error.message;}
        finally {button.disabled = false; this.setBusy(false);}
        return;
      }
      if (this.modal.dataset.action === 'retention') {
        this.setBusy(true);
        try {
          await this.transport.request('workspace_viewer/retention', {days:Number(this.modal.querySelector('select').value)});
          this.modal.close(); this.notice('Automatic archiving preference saved. It applies at the next WanGP startup.');
        } catch (error) {const node = this.modal.querySelector('[role=alert]'); node.hidden = false; node.textContent = error.message;}
        finally {button.disabled = false; this.setBusy(false);}
        return;
      }
      const ok = await this.run(this.modal.dataset.action, {confirmed: this.modal.dataset.action === 'delete', name: this.modal.querySelector('input').value, target: this.modal.querySelector('select').value});
      button.disabled = false; if (ok) this.modal.close();
    }
    async importFiles(files) {
      if (this.busy) return;
      this.setBusy(true);
      const workspace = this.workspace;
      let done = 0; const failures = [], reused = [];
      for (const file of files) {
        this.notice(`Importing ${++done} / ${files.length}: ${file.name}`);
        const data = new FormData(); data.append('file', file);
        try {
          const result = await this.transport.request('workspace_viewer/import?' + new URLSearchParams({workspace}), data);
          if (result.reused) reused.push(`Existing file reused: ${result.filename}`);
        }
        catch (error) {failures.push(`${file.name}: ${error.message}`);}
      }
      this.setBusy(false); this.dialog.querySelector('[data-files]').value = '';
      await this.load(this.state.page); this.notice([...reused, ...failures].join(' · '));
    }
    installDivider() {
      const panes = this.dialog.querySelector('.wv-detail-panes'), divider = panes.querySelector('.wv-divider');
      const resize = percent => {
        percent = Math.max(10, Math.min(80, percent));
        panes.style.setProperty('--wv-preview-height', percent + '%');
        divider.setAttribute('aria-valuenow', String(Math.round(percent)));
        window.__meshGalleryNS?.refresh();
      };
      let offset = 0;
      divider.onpointerdown = event => {
        if (event.button !== 0) return;
        event.preventDefault(); divider.focus();
        offset = event.clientY - this.preview.getBoundingClientRect().bottom;
        divider.setPointerCapture(event.pointerId); panes.classList.add('wv-resizing');
      };
      divider.onpointermove = event => {
        if (!divider.hasPointerCapture(event.pointerId)) return;
        const box = panes.getBoundingClientRect(); resize(100 * (event.clientY - box.top - offset) / box.height);
      };
      divider.onpointerup = divider.onpointercancel = event => {if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId);};
      divider.onlostpointercapture = () => panes.classList.remove('wv-resizing');
      divider.onkeydown = event => {
        const current = 100 * this.preview.getBoundingClientRect().height / panes.clientHeight;
        const next = {ArrowUp: current - 5, ArrowDown: current + 5, Home: 10, End: 80}[event.key];
        if (next === undefined) return;
        event.preventDefault(); resize(next);
      };
      divider.ondblclick = () => resize(34);
    }
    installDrag() {
      this.grid.ondragstart = event => {
        if (event.target.closest('button')) {event.preventDefault(); return;}
        const tile = event.target.closest('[data-key]'); if (!tile) return;
        if (!this.selected.has(tile.dataset.key)) {this.selected.clear(); this.selected.add(tile.dataset.key); this.selectionChanged();}
        this.dragging = true; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', `${this.selected.size} media`);
      };
      this.dialog.ondragend = () => {this.dragging = false; clearTimeout(this.pageHover); this.grid.querySelectorAll('.wv-drop').forEach(tile => tile.classList.remove('wv-drop'));};
      this.grid.ondragover = event => {
        if (!this.dragging) return; event.preventDefault(); event.dataTransfer.dropEffect = 'move';
        this.grid.querySelectorAll('.wv-drop').forEach(tile => tile.classList.remove('wv-drop'));
        event.target.closest('[data-key]')?.classList.add('wv-drop');
      };
      this.grid.ondrop = async event => {if (!this.dragging) return; event.preventDefault(); const tile = event.target.closest('[data-key]'); this.dragging = false; clearTimeout(this.pageHover); await this.run('reorder', {before: tile?.dataset.key || null});};
      for (const [selector, delta] of [['[data-prev]', -1], ['[data-next]', 1]]) {
        const button = this.dialog.querySelector(selector);
        button.ondragover = event => {if (!this.dragging || button.disabled) return; event.preventDefault(); if (!this.pageHover) this.pageHover = setTimeout(async () => {await this.load(this.state.page + delta); this.pageHover = null;}, 650);};
        button.ondragleave = () => {clearTimeout(this.pageHover); this.pageHover = null;};
      }
    }
    installRubber() {
      this.browser.onpointerdown = event => {
        if (event.button !== 0 || event.pointerType !== 'mouse' || event.target.closest('[data-key]')) return;
        const box = this.browser.getBoundingClientRect(), origin = {x: event.clientX - box.left + this.browser.scrollLeft, y: event.clientY - box.top + this.browser.scrollTop};
        const base = event.ctrlKey || event.metaKey ? new Set(this.selected) : new Set();
        let client = {x:event.clientX, y:event.clientY}, moved = false;
        const rubber = this.dialog.querySelector('.wv-rubber');
        this.browser.setPointerCapture(event.pointerId);
        const update = () => {
          const bounds = this.browser.getBoundingClientRect();
          if (client.y > bounds.bottom - 35) this.browser.scrollTop += 12;
          if (client.y < bounds.top + 35) this.browser.scrollTop -= 12;
          const x = client.x - bounds.left + this.browser.scrollLeft, y = client.y - bounds.top + this.browser.scrollTop;
          const rect = {left:Math.min(x,origin.x), top:Math.min(y,origin.y), right:Math.max(x,origin.x), bottom:Math.max(y,origin.y)};
          moved ||= Math.abs(x-origin.x)+Math.abs(y-origin.y)>5;
          if (moved) {
            rubber.hidden = false; Object.assign(rubber.style, {left:rect.left+'px', top:rect.top+'px', width:rect.right-rect.left+'px', height:rect.bottom-rect.top+'px'});
            this.selected.clear(); base.forEach(key => this.selected.add(key));
            this.grid.querySelectorAll('[data-key]').forEach(tile => {const r=tile.getBoundingClientRect(), left=r.left-bounds.left+this.browser.scrollLeft, top=r.top-bounds.top+this.browser.scrollTop; if(left<rect.right && left+r.width>rect.left && top<rect.bottom && top+r.height>rect.top) this.selected.add(tile.dataset.key);});
            this.selectionChanged();
          }
          this.rubberFrame = requestAnimationFrame(update);
        };
        this.browser.onpointermove = move => {client={x:move.clientX,y:move.clientY};};
        const end = () => {cancelAnimationFrame(this.rubberFrame); rubber.hidden=true; this.suppressClick=moved; setTimeout(() => {this.suppressClick=false;}, 0); this.browser.onpointermove=null; this.browser.onpointerup=null; this.browser.onpointercancel=null;};
        this.browser.onpointerup=end; this.browser.onpointercancel=end; update();
      };
    }
  }
  window.WanGPWorkspaceViewer = WorkspaceViewer;
})();
