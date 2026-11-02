// WanGP wrapper: Phaser's ESM build only has named exports. Re-export them and add the namespace as the
// default export, so that both `import Phaser from 'phaser'` and `import * as Phaser from 'phaser'` work.
// The namespace is also published as the global `Phaser`, which plugins and script-style code expect.
import * as Phaser from './phaser.esm.js';
export * from './phaser.esm.js';
export default Phaser;
globalThis.Phaser ??= Phaser;
