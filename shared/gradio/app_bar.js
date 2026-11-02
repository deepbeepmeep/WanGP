(function () {
    if (window.WanGPAppBar) return;
    const root = () => window.gradioApp ? window.gradioApp() : document;
    const host = () => root().querySelector('#wangp-app-bar');
    const nativeTabs = () => root().querySelector('.wangp-native-app-tabs');
    const catalogKey = 'wangp.appDiscovery.catalog:' + (window.WanGPAppLaunch?.base || new URL(window.gradio_config?.root || '/', location.href).pathname.replace(/\/?$/, '/'));
    let current = 'media_gen', opener;
    function send(id, data) {
        const input = root().querySelector('#' + id + ' textarea, #' + id + ' input');
        if (!input) return; // A window focus can arrive before Gradio mounts the controls.
        input.value = JSON.stringify({...data, nonce: Date.now() + Math.random()});
        input.dispatchEvent(new Event('input', {bubbles: true}));
        input.dispatchEvent(new Event('change', {bubbles: true}));
    }
    function closeMenu() {
        const menu = host()?.querySelector('.app-bar-menu');
        if (menu) menu.hidden = true;
        host()?.querySelector('.app-bar-more')?.setAttribute('aria-expanded', 'false');
    }
    function closeIntro() {
        const backdrop = host()?.querySelector('.app-welcome-backdrop');
        if (backdrop) backdrop.hidden = true;
        opener?.focus({preventScroll: true});
    }
    function intro(plugin, button, onboarding = true) {
        closeMenu();
        opener = button;
        const backdrop = host().querySelector('.app-welcome-backdrop');
        const card = backdrop.querySelector(`[data-app-intro="${CSS.escape(plugin)}"]`);
        if (!card) return;
        backdrop.querySelectorAll('[data-app-intro]').forEach(card => card.hidden = card.dataset.appIntro !== plugin);
        card.querySelectorAll('[data-app-onboarding]').forEach(section => section.hidden = !onboarding);
        card.querySelector('[data-app-information]').hidden = onboarding;
        backdrop.hidden = false;
        card.querySelector(onboarding ? '[data-app-decision="keep"]' : '[data-app-close]').focus();
    }
    function layout() {
        const shell = host();
        if (!shell) return;
        const viewport = shell.querySelector('.app-bar-viewport');
        if (!viewport) return;
        const overflow = shell.querySelector('.app-bar-overflow');
        overflow.hidden = true;
        if (viewport.scrollWidth > viewport.clientWidth + 1) overflow.hidden = false;
        viewport.scrollLeft = 0;
        const bounds = viewport.getBoundingClientRect();
        const hiddenTargets = new Set();
        viewport.querySelectorAll('.app-bar-link').forEach(button => {
            const rect = button.getBoundingClientRect();
            const fits = rect.left >= bounds.left - 1 && rect.right <= bounds.right + 1;
            button.style.visibility = fits ? '' : 'hidden';
            if (!button.hidden && !fits) hiddenTargets.add(button.dataset.appRoute);
        });
        shell.querySelectorAll('.app-bar-menu .app-bar-link').forEach(button => button.hidden = !hiddenTargets.has(button.dataset.appRoute));
        const more = shell.querySelector('.app-bar-more');
        if (hiddenTargets.has(current)) more.setAttribute('aria-current', 'page');
        else more.removeAttribute('aria-current');
    }
    function sync() {
        const shell = host(), native = nativeTabs();
        if (!shell?.querySelector('.app-bar')) return;
        const active = native?.querySelector(':scope > .tab-wrapper button[data-tab-id][aria-selected="true"]');
        if (!active) { layout(); return; }
        native.querySelector(':scope > .tab-wrapper').setAttribute('aria-hidden', 'true');
        current = active.dataset.tabId;
        document.documentElement.classList.toggle('wangp-main-tab', current === 'media_gen');
        if (JSON.parse(shell.querySelector('.app-bar').dataset.removedApps).includes(current)) {
            activate('media_gen');
            return;
        }
        const highlighted = current === 'plugin_model_manager' ? 'plugin_configuration' : current;
        shell.querySelectorAll('[data-app-route]').forEach(button => {
            const target = button.dataset.appRoute;
            button.hidden = !native.querySelector(`:scope > .tab-wrapper button[data-tab-id="${CSS.escape(target)}"]`);
            if (target === highlighted) button.setAttribute('aria-current', 'page');
            else button.removeAttribute('aria-current');
        });
        layout();
    }
    function activate(target) {
        closeMenu();
        const button = nativeTabs().querySelector(`:scope > .tab-wrapper button[data-tab-id="${CSS.escape(target)}"]`);
        if (button?.getAttribute('aria-selected') === 'true') window.WanGPAddonNavigation?.selected(target);
        else button?.click();
    }
    window.WanGPAppBar = {
        introduce(target) {
            const link = host().querySelector(`[data-app-route="${CSS.escape(target)}"]`);
            if (link) intro(link.dataset.appPlugin, link);
        },
        navigate(target, initial = false) { send('wangp_app_route', {target, initial}); },
        commitRoute(value) {
            const {target, initial} = JSON.parse(value);
            window.WanGPAddonNavigation?.beforeRoute(target);
            const link = host().querySelector(`[data-app-route="${CSS.escape(target)}"]`);
            if (!initial && link?.dataset.appIntroPending === 'true') intro(link.dataset.appPlugin, link);
            else activate(target);
        },
        selected(target) {
            current = target;
            closeMenu();
            const link = host()?.querySelector(`[data-app-route="${CSS.escape(target)}"]`);
            if (link?.dataset.appIntroPending === 'true') intro(link.dataset.appPlugin, link);
            window.WanGPAddonNavigation?.selected(target);
            requestAnimationFrame(sync);
        },
        refresh() { requestAnimationFrame(() => { sync(); window.WanGPAddonNavigation?.refresh(); }); },
        order(plugins) {
            // Links of these Plugins take their places in this order; the other links stay where they are.
            const rank = link => plugins.indexOf(link.dataset.appPlugin);
            host()?.querySelectorAll('.app-bar-apps, .app-bar-menu').forEach(list => {
                const links = [...list.children], moved = links.filter(link => rank(link) >= 0).sort((a, b) => rank(a) - rank(b));
                list.replaceChildren(...links.map(link => rank(link) >= 0 ? moved.shift() : link));
            });
            layout();
        },
        visited(value) {
            const visited = JSON.parse(value);
            host()?.querySelectorAll('[data-app-plugin]').forEach(button => {
                if (visited.includes(button.dataset.appPlugin)) {
                    const badge = button.querySelector('.app-bar-new');
                    badge?.classList.add('is-seen');
                    badge?.setAttribute('aria-hidden', 'true');
                    button.dataset.appIntroPending = 'false';
                }
            });
            requestAnimationFrame(sync);
            try { localStorage.setItem(catalogKey, String(Date.now())); } catch (_) { /* The next page focus refreshes shared state. */ }
        },
        changed() {
            closeIntro();
            requestAnimationFrame(sync);
            try { localStorage.setItem(catalogKey, String(Date.now())); } catch (_) { /* The next page focus refreshes shared state. */ }
        },
        reviewed(value) {
            const {plugin, action} = JSON.parse(value);
            closeIntro();
            if (action === 'keep') {
                const link = host().querySelector(`[data-app-plugin="${CSS.escape(plugin)}"]`);
                if (link) {
                    host().querySelectorAll(`[data-app-plugin="${CSS.escape(plugin)}"]`).forEach(button => button.dataset.appIntroPending = 'false');
                    activate(link.dataset.appRoute);
                }
            } else activate('media_gen');
            this.changed();
        }
    };
    document.addEventListener('click', event => {
        const button = event.target.closest('button');
        if (button?.dataset.appInfo) { intro(button.dataset.appInfo, button, false); return; }
        const route = button?.dataset.appRoute;
        if (route) { window.WanGPAppBar.navigate(route); return; }
        if (button?.classList.contains('app-bar-more')) {
            const menu = host().querySelector('.app-bar-menu');
            menu.hidden = !menu.hidden;
            button.setAttribute('aria-expanded', String(!menu.hidden));
            if (!menu.hidden) menu.querySelector('button:not([hidden])')?.focus();
            return;
        }
        if (!event.target.closest('.app-bar-overflow')) closeMenu();
        if (button?.classList.contains('app-welcome-close') || button?.hasAttribute('data-app-close') || event.target.classList.contains('app-welcome-backdrop')) closeIntro();
        if (button?.dataset.appDecision) {
            button.disabled = true;
            send('wangp_app_review', {plugin: button.closest('[data-app-intro]').dataset.appIntro, action: button.dataset.appDecision});
        }
        if (button?.dataset.appEnable) {
            button.disabled = true;
            send('plugin_action_input', {action: 'enable_local', plugin_id: button.dataset.appEnable});
        }
        if (button?.dataset.appRemove) send('wangp_app_review', {plugin: button.dataset.appRemove, action: 'remove'});
        if (button?.hasAttribute('data-app-restart')) window.handleRestart();
    });
    document.addEventListener('keydown', event => {
        const backdrop = host()?.querySelector('.app-welcome-backdrop');
        if (event.key === 'Escape') { closeMenu(); closeIntro(); }
        if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.repeat && event.key.length === 1) {
            const link = host()?.querySelector(`.app-bar-utilities [aria-keyshortcuts="Alt+${CSS.escape(event.key.toUpperCase())}"]:not([hidden])`);
            if (link) { event.preventDefault(); window.WanGPAppBar.navigate(link.dataset.appRoute); }
        }
        if (event.key === 'Tab' && backdrop && !backdrop.hidden) {
            const buttons = Array.from(backdrop.querySelectorAll('button')).filter(button => button.offsetParent !== null && !button.disabled);
            const first = buttons[0], last = buttons[buttons.length - 1];
            if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
        const menu = event.target.closest('.app-bar-menu');
        if (menu && ['ArrowUp', 'ArrowDown'].includes(event.key)) {
            event.preventDefault();
            const buttons = Array.from(menu.querySelectorAll('button:not([hidden])'));
            buttons[(buttons.indexOf(event.target) + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length].focus();
        }
    });
    // Measure the mounted links before paint, including when a catalog update replaces them.
    customElements.define('wangp-app-links', class extends HTMLElement {
        connectedCallback() { queueMicrotask(sync); }
    });
    const refresh = () => send('wangp_app_refresh', {});
    window.addEventListener('focus', refresh);
    window.addEventListener('storage', event => { if (event.key === catalogKey) refresh(); });
    window.addEventListener('resize', layout);
    document.fonts.addEventListener('loadingdone', layout);
})();
