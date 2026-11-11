// App windows on Linux: WebKitGTK no longer gives pages the files dropped from other applications (see file_drops.rs).
// The Manager offers those files here; drops and Files types reach the page's handlers as in a browser.
(() => {
  if (window.__wangpFileDrop) return;
  const typesOf = Object.getOwnPropertyDescriptor(DataTransfer.prototype, 'types').get;
  const filesOf = Object.getOwnPropertyDescriptor(DataTransfer.prototype, 'files').get;
  let pending = null, waiting = [];
  const external = transfer => !!pending && typesOf.call(transfer).includes('text/uri-list') && !filesOf.call(transfer).length;
  Object.defineProperty(DataTransfer.prototype, 'types', {configurable: true, enumerable: true, get() {
    const types = typesOf.call(this);
    return external(this) && !types.includes('Files') ? Object.freeze([...types, 'Files']) : types;
  }});
  Object.defineProperty(DataTransfer.prototype, 'files', {configurable: true, enumerable: true, get() {
    return external(this) && pending.list ? pending.list : filesOf.call(this);
  }});
  const load = async ({token, files}) => {
    const transfer = new DataTransfer();
    for (const [index, file] of files.entries()) {
      const blob = await (await fetch(`wangp-drop://${token}/${index}`)).blob();
      transfer.items.add(new File([blob], file.name, {type: file.type, lastModified: file.modified}));
    }
    return transfer.files;
  };
  window.__wangpFileDrop = {
    offer(offer) {
      const key = offer.token + JSON.stringify(offer.files);
      if (pending?.key === key) return;
      const entry = {key, list: null};
      entry.ready = load(offer).then(list => (entry.list = list));
      pending = entry;
      waiting.splice(0).forEach(resolve => resolve());
    },
    clear() { pending = null; },
  };
  // A drag that starts in the page carries no offered files.
  window.addEventListener('dragstart', () => { pending = null; }, true);
  window.addEventListener('drop', event => {
    if (!event.isTrusted || filesOf.call(event.dataTransfer).length) return;
    const fileUris = /^file:/m.test(event.dataTransfer.getData('text/uri-list'));
    if (!external(event.dataTransfer) && !fileUris) return;
    if (pending?.list) { setTimeout(() => { pending = null; }, 1000); return; }
    // Files still loading, or the offer not yet here: the drop is replayed once they are.
    event.preventDefault();
    event.stopImmediatePropagation();
    const {target, clientX, clientY, screenX, screenY} = event;
    const offered = pending ? Promise.resolve() : new Promise(resolve => { waiting.push(resolve); setTimeout(resolve, 2000); });
    offered.then(() => pending?.ready).then(list => {
      if (!list) return;
      const transfer = new DataTransfer();
      for (const file of list) transfer.items.add(file);
      if (target instanceof HTMLInputElement && target.type === 'file') {
        target.files = transfer.files;
        target.dispatchEvent(new Event('input', {bubbles: true}));
        target.dispatchEvent(new Event('change', {bubbles: true}));
      } else target.dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, composed: true, clientX, clientY, screenX, screenY, dataTransfer: transfer}));
      setTimeout(() => { pending = null; }, 1000);
    });
  }, true);
})();
