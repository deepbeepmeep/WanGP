// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// The backend can see HTTP after a proxy terminates browser HTTPS. Gradio's
// embedded root then points the browser at mixed-content API and queue URLs.
(() => {
  // Event results can still contain absolute HTTP media URLs after config.root
  // is corrected. Upgrade subresources only on HTTPS pages, preserving LAN HTTP.
  if (location.protocol === 'https:' &&
      !document.querySelector('meta[data-wangp-upgrade-insecure-requests]')) {
    const policy = document.createElement('meta');
    policy.httpEquiv = 'Content-Security-Policy';
    policy.content = 'upgrade-insecure-requests';
    policy.setAttribute('data-wangp-upgrade-insecure-requests', 'true');
    document.head.appendChild(policy);
  }
  const config = window.gradio_config;
  if (!config?.root) return;
  let root;
  try { root = new URL(config.root, location.href); }
  catch { return; }
  if (root.host !== location.host || root.protocol !== 'http:' || location.protocol !== 'https:') return;
  root.protocol = 'https:';
  config.root = root.href.replace(/\/$/, '');
})();
