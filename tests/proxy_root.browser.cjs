// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Optional browser regression: install playwright, then run this file with Node.
// Uses installed Microsoft Edge unless WANGP_TEST_BROWSER is set to chromium.
// An optional first argument selects a baseline proxy_root.js for red/green checks.
const { chromium } = require('playwright');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const assert = require('node:assert/strict');
const source = readFileSync(process.argv[2] || join(__dirname, '../shared/gradio/proxy_root.js'), 'utf8');
// A silent PCM WAV with a valid header, used only in the isolated browser fixture.
const wav = Buffer.alloc(44 + 1600);
wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(8000, 24); wav.writeUInt32LE(16000, 28);
wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(1600, 40);
(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.WANGP_TEST_BROWSER === 'chromium' ? {} : { channel: 'msedge' }) });
  try {
    for (const scheme of ['https', 'http']) {
      const context = await browser.newContext();
      try {
        const requests = [];
        await context.route('**/*', async route => {
          const url = new URL(route.request().url());
          assert.equal(url.hostname, 'wangp.example.test', 'Fixture must never contact real services');
          if (url.pathname === '/gradio_api/file=fixture.wav') {
            requests.push(url.protocol);
            await route.fulfill({ contentType: 'audio/wav', body: wav });
          } else {
            await route.fulfill({ contentType: 'text/html', body:
              `<html><head><script>window.gradio_config={root:'http://wangp.example.test'};</script>` +
              `<script>${source}</script></head><body>Proxy media test</body></html>` });
          }
        });
        const page = await context.newPage();
        await page.goto(`${scheme}://wangp.example.test/`);
        const result = await page.evaluate(async () => {
          const url = 'http://wangp.example.test/gradio_api/file=fixture.wav';
          // Models the absolute HTTP URL returned by a later Gradio event.
          try {
            const response = await fetch(url);
            const bytes = await response.arrayBuffer();
            const audio = document.createElement('audio');
            audio.preload = 'metadata';
            const metadata = new Promise((resolve, reject) => {
              audio.onloadedmetadata = () => resolve(audio.duration);
              audio.onerror = () => reject(new Error('Audio metadata failed'));
              setTimeout(() => reject(new Error('Audio metadata timeout')), 10000);
            });
            audio.src = url;
            document.body.appendChild(audio);
            return { status: response.status, bytes: bytes.byteLength, duration: await metadata };
          } catch (error) { return { error: String(error) }; }
        });
        assert.equal(result.error, undefined, JSON.stringify(result));
        assert.equal(result.status, 200);
        assert.equal(result.bytes, wav.length);
        assert.ok(Math.abs(result.duration - 0.1) < 0.01);
        assert.ok(requests.length >= 1);
        assert.ok(requests.every(protocol => protocol === `${scheme}:`), JSON.stringify(requests));
        console.log(`PASS: ${scheme.toUpperCase()} fetch and audio metadata; requests ${requests.join(', ')}`);
      } finally { await context.close(); }
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
