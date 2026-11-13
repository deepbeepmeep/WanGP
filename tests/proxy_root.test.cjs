// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Run with Node.js 18+: node --test tests/proxy_root.test.cjs
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { runInNewContext } = require('node:vm');
const { test } = require('node:test');
const assert = require('node:assert/strict');

const source = readFileSync(join(__dirname, '../shared/gradio/proxy_root.js'), 'utf8');
function page(href, root, repeats = 1) {
  const metas = [];
  const config = root === undefined ? undefined : { root };
  const document = {
    querySelector: () => metas.find(m => m.attributes['data-wangp-upgrade-insecure-requests']) || null,
    createElement: () => ({ attributes: {}, setAttribute(k, v) { this.attributes[k] = v; } }),
    head: { appendChild(meta) { metas.push(meta); } },
  };
  const context = { window: { gradio_config: config }, document, location: new URL(href), URL };
  for (let i = 0; i < repeats; i++) runInNewContext(source, context);
  return { config, metas };
}
const secure = 'https://wangp.example.test/';
test('HTTPS adds an upgrade policy for later media URLs', () => {
  const { metas, config } = page(secure, 'http://wangp.example.test');
  assert.equal(config.root, 'https://wangp.example.test');
  assert.equal(metas.length, 1);
  assert.equal(metas[0].httpEquiv, 'Content-Security-Policy');
  assert.equal(metas[0].content, 'upgrade-insecure-requests');
});
test('direct LAN HTTP stays HTTP and gets no upgrade policy', () => {
  const { config, metas } = page('http://192.0.2.10:7860/', 'http://192.0.2.10:7860');
  assert.equal(config.root, 'http://192.0.2.10:7860');
  assert.equal(metas.length, 0);
});
test('repeated initialization does not duplicate the policy', () => {
  assert.equal(page(secure, secure.slice(0, -1), 2).metas.length, 1);
});
test('HTTPS policy does not depend on config availability', () => {
  assert.equal(page(secure, undefined).metas.length, 1);
});
test('malformed config roots do not throw or skip the HTTPS policy', () => {
  assert.equal(page(secure, 'http://[invalid').metas.length, 1);
});
test('root correction does not rewrite a different authority', () => {
  assert.equal(page(secure, 'http://other.example.test').config.root, 'http://other.example.test');
  assert.equal(page(secure, 'http://wangp.example.test:7860').config.root, 'http://wangp.example.test:7860');
});
test('already secure roots retain their path', () => {
  assert.equal(page(secure, 'https://wangp.example.test/app').config.root, 'https://wangp.example.test/app');
});
test('HTTP without config remains a no-op', () => {
  assert.equal(page('http://192.0.2.10:7860/', undefined).metas.length, 0);
});
