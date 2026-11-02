// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(function () {
    if (window.WanGPAddonNavigation) return;
    const root = () => window.gradioApp ? window.gradioApp() : document;
    const linkFor = target => root().querySelector(`.app-bar-apps [data-app-route="${CSS.escape(target)}"]`);
    const panelFor = target => root().querySelector(`#${CSS.escape(linkFor(target)?.dataset.appPanel || 'missing-addon-panel')}`);
    const base = window.WanGPAppLaunch?.base || new URL(window.gradio_config?.root || '/', location.href).pathname.replace(/\/?$/, '/');
    // Keep relative assets and embedded apps rooted at WanGP after changing the folder URL.
    if (!document.querySelector('base')) {
        const element = document.createElement('base');
        element.href = base;
        document.head.prepend(element);
    }
    let initialized = false, pending = null, fullscreenTarget = null, fullscreenPanel = null, scrollY = 0, message = '';

    function updateUrl(target, useFolder) {
        const url = new URL(location.href), link = linkFor(target);
        url.searchParams.delete('app');
        url.searchParams.delete('fullscreen');
        if (link?.dataset.appUrl && useFolder) {
            url.pathname = base.replace(/\/$/, '') + link.dataset.appUrl + '/';
        } else {
            url.pathname = base;
        }
        if (url.href !== location.href) history.replaceState(history.state, '', url);
    }

    function setFullscreen(target) {
        if (target === fullscreenTarget) return;
        const previous = fullscreenTarget;
        fullscreenPanel?.classList.remove('wangp-addon-fullscreen');
        fullscreenTarget = target;
        document.documentElement.classList.toggle('wangp-addon-fullscreen-open', !!target);
        if (target) {
            if (!previous) scrollY = window.scrollY;
            fullscreenPanel = panelFor(target);
            fullscreenPanel.classList.add('wangp-addon-fullscreen');
            fullscreenPanel.querySelector('[data-addon-return]')?.focus({preventScroll: true});
        } else if (previous) {
            panelFor(previous)?.querySelector('[data-addon-fullscreen]')?.focus({preventScroll: true});
            window.scrollTo({top: scrollY, behavior: 'instant'});
        }
        if (!target) fullscreenPanel = null;
        window.dispatchEvent(new Event('resize'));
    }

    // An embedded web app (an iframe with data-src in .wangp-addon-web-view) loads when its tab is first shown, then hears whether
    // its tab is shown or hidden, so that a hidden one keeps nothing running. Without a target, the selected tabs decide.
    function syncWebViews(target) {
        const tabs = Array.from(root().querySelectorAll('button[role="tab"][aria-controls]'));
        root().querySelectorAll('.wangp-addon-web-view iframe[data-src]').forEach(frame => {
            const tab = tabs.find(button => root().querySelector(`#${CSS.escape(button.getAttribute('aria-controls'))}`)?.contains(frame));
            const shown = !!tab && (target ? tab.dataset.tabId === target : tab.getAttribute('aria-selected') === 'true');
            if (frame.getAttribute('src')) frame.contentWindow?.postMessage({wangpWebView: shown ? 'shown' : 'hidden'}, location.origin);
            else if (shown) {
                frame.addEventListener('load', () => syncWebViews(null), {once: true});  // a tab left while the app was loading
                frame.src = frame.dataset.src;
            }
        });
    }

    function request(target, fullscreen, launch = null, folder = false) {
        const link = linkFor(target);
        if (!link || (fullscreen && link.dataset.appFullscreen !== 'true')) return;
        pending = {target, fullscreen, intro: launch?.intro, folder};
        window.WanGPAppBar.navigate(target, !!launch);
    }

    function readUrl(launch = null) {
        const query = new URL(location.href).searchParams;
        const path = location.pathname.replace(/\/$/, '');
        const folder = Array.from(root().querySelectorAll('.app-bar-apps [data-app-url]')).find(link => base.replace(/\/$/, '') + link.dataset.appUrl === path);
        const plugin = launch?.plugin || folder?.dataset.appPlugin || query.get('app');
        message = '';
        if (plugin) {
            const link = root().querySelector(`.app-bar-apps [data-app-plugin="${CSS.escape(plugin)}"][data-app-url]`);
            if (link && !link.hidden && !launch?.unavailable) {
                const fullscreen = (launch ? launch.fullscreen : query.has('fullscreen') ? query.get('fullscreen') === '1' : !!folder) && link.dataset.appFullscreen === 'true';
                request(link.dataset.appRoute, fullscreen, launch, !!folder);
            } else message = `The plugin “${plugin}” is unavailable. Open Apps & PlugIn Manager to install or enable it, then restart WanGP.`;
        } else {
            pending = null;
            setFullscreen(null);
        }
        const status = root().querySelector('.wangp-addon-route-status');
        if (status) { status.textContent = message; status.hidden = !message; }
        if (message) document.documentElement.classList.remove('wangp-app-launch');
    }

    window.WanGPAddonNavigation = {
        beforeRoute(target) { if (pending?.target !== target) pending = null; },
        selected(target) {
            syncWebViews(target);
            if (!initialized) return;
            message = '';
            const status = root().querySelector('.wangp-addon-route-status');
            if (status) status.hidden = true;
            const fullscreen = pending?.target === target ? pending.fullscreen : fullscreenTarget === target;
            const useFolder = fullscreen || (pending?.target === target && pending.folder);
            if (pending?.target === target && pending.intro) window.WanGPAppBar.introduce(target);
            pending = null;
            setFullscreen(fullscreen ? target : null);
            updateUrl(target, useFolder);
            document.documentElement.classList.remove('wangp-app-launch');
        },
        refresh() {
            if (!initialized) { initialized = true; syncWebViews(null); readUrl(window.WanGPAppLaunch || null); }
            if (fullscreenTarget && !linkFor(fullscreenTarget)) setFullscreen(null);
            const status = root().querySelector('.wangp-addon-route-status');
            if (status) { status.textContent = message; status.hidden = !message; }
        }
    };

    document.addEventListener('click', event => {
        const control = event.target.closest('[data-addon-fullscreen], [data-addon-return]');
        if (!control || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        const target = control.closest('[data-addon-target]').dataset.addonTarget;
        // The selected tab already exists: keep its form, iframe and event subscriptions alive.
        const fullscreen = control.hasAttribute('data-addon-fullscreen');
        setFullscreen(fullscreen ? target : null);
        updateUrl(target, fullscreen);
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && fullscreenTarget && !event.defaultPrevented) {
            const target = fullscreenTarget;
            setFullscreen(null);
            updateUrl(target, false);
        }
    });
    window.addEventListener('popstate', () => readUrl());
})();
