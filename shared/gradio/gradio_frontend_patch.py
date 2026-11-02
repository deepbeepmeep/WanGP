# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Avoid propagating Gradio layout updates into unaffected Svelte branches."""
from functools import lru_cache, wraps
from hashlib import sha256
import json
from pathlib import Path
import re

from fastapi.responses import JSONResponse, Response
from gradio import routes

from shared.gradio.gradio_model_change_queue import _REPLACEMENTS as _QUEUE_REPLACEMENTS
from shared.gradio import metadata_events


_EDITOR_PATH = Path(__file__).parent / 'wangp_image_editor/templates/component/index.js'
# Gradio's upload control and the editor's bundled upload action must resolve
# gallery previews identically. Fetch the original only when a drop occurs.
_GALLERY_DROP_FILES = r'''async function wangpGalleryDropFiles(event) {
    const files = Array.from(event.dataTransfer.files);
    const media = event.dataTransfer.getData('application/x-wangp-gallery-media');
    const source = media || event.dataTransfer.getData('text/uri-list').split(/\r?\n/).find(line => line && !line.startsWith('#'));
    if (!source || !URL.canParse(source)) return files;
    const url = new URL(source);
    if (url.origin !== location.origin || !url.pathname.includes('/file=') || (!media && !url.searchParams.has('__wangp_gallery_preview'))) return files;
    // Replace the browser's reduced-resolution image File entirely.
    url.searchParams.delete('__wangp_gallery_preview');
    const response = await fetch(url);
    if (!response.ok) throw new Error('Could Not Load Original Gallery Media.');
    const blob = await response.blob();
    const name = decodeURIComponent(url.pathname.split('/file=')[1]).split(/[\\/]/).pop();
    return [new File([blob], name, {type: blob.type})];
}
'''
# Pixi's scheduler only serves wall-clock GC tasks in this bundle. Retain its
# callbacks, offsets and repeat bookkeeping, but wake at their actual deadlines.
# The separate pointer ticker must still run whenever hit testing unpauses it.
_EDITOR_SCHEDULER = """
XC = class extends XC {
    init() {}
    repeat(...args) {
        const id = super.repeat(...args);
        this.wangpSchedule();
        return id;
    }
    cancel(id) {
        super.cancel(id);
        this.wangpSchedule();
    }
    wangpSchedule() {
        clearTimeout(this.wangpTimer);
        if (!this._tasks.length) return;
        const due = Math.min(...this._tasks.map(task => task.last + task.offset + task.duration));
        this.wangpTimer = setTimeout(() => {
            super._update();
            this.wangpSchedule();
        }, Math.max(0, Math.ceil(due - performance.now())));
    }
    destroy() {
        clearTimeout(this.wangpTimer);
        super.destroy();
    }
};
"""


# Gradio mutates layout nodes in place and publishes the entire tree each frame.
# Mark affected nodes and their ancestors before publishing. A Node can skip an
# incoming $set only when both its revision and ALL incoming props are unchanged.
# Same-value outputs still mark their nodes; normal event/prop handling is kept.
# Dynamic layout rebuilds invalidate every node, preserving upstream rendering.
_NODE_INPUTS = '"root"in w&&t(1,n=w.root),"node"in w&&t(0,s=w.node)'
_NODE_SKIP = """
const wangpVersion = ("node" in w ? w.node : s).__wangp_revision;
if (wangpVersion === wangpNodeVersion && Object.keys(w).every(key => w[key] === wangpNodeInputs[key])) return;
wangpNodeVersion = wangpVersion;
Object.assign(wangpNodeInputs, w);
"""
# Gradio's helper holds only its constructor inputs and bound dispatch/load
# methods. Reuse it across value/status updates; replace it on context changes.
_GRADIO_CONTEXT = """
let wangpGradioArgs, wangpGradioValue;
function wangpGradio(...args) {
    if (!wangpGradioArgs || args.some((arg, index) => arg !== wangpGradioArgs[index])) {
        wangpGradioArgs = args;
        wangpGradioValue = new er(...args);
    }
    return wangpGradioValue;
}
"""
# WebKit's fetch body reader can strand SSE bytes while its consumer is busy
# (WebKit bug 322545). Native EventSource does not use that reader. Preserve
# Gradio's fetch transport for clients that require custom request headers.
_NATIVE_EVENT_STREAM = """
if (typeof window !== "undefined" && new Headers(o.headers).keys().next().done) {
    const source = new EventSource(e, {withCredentials: o.credentials === "include"});
    const close = () => {
        source.close();
        o.signal?.removeEventListener("abort", close);
    };
    // Gradio owns stream completion/reopening; never auto-reconnect an old job.
    source.addEventListener("error", close, {once: true});
    if (o.signal?.aborted) close();
    else o.signal?.addEventListener("abort", close, {once: true});
    return source;
}
"""
_MARK_ANCESTORS = """
for (let node = f; node; node = node.parent) wangpDirty.add(node);
}
}
for (const node of wangpDirty) node.__wangp_revision = (node.__wangp_revision || 0) + 1;
"""
# These two upstream frame callbacks flush component props and dispatch events;
# neither paints. A posted task preserves the Svelte microtask checkpoint before
# them without waiting for Chrome's (possibly occluded) rendering clock. Using a
# microtask here would clear value_is_output before the value reaches a component,
# incorrectly turning server updates into user input events. Keep each callback
# in its own task, and keep the user's background-processing preference.
_UPDATE_SCHEDULER = """
let wangpUpdateChannel;
const wangpUpdateJobs = [];
export function wangpScheduleUpdate(callback) {
    const focus = window.__gradioFocusQueuePatch;
    if (focus && !focus.enableBackgroundScheduler && focus.isBackground()) {
        requestAnimationFrame(callback);
        return;
    }
    if (!wangpUpdateChannel) {
        wangpUpdateChannel = new MessageChannel();
        wangpUpdateChannel.port1.onmessage = () => wangpUpdateJobs.shift()(performance.now());
    }
    wangpUpdateJobs.push(callback);
    wangpUpdateChannel.port2.postMessage(0);
}
// Gradio copies every incoming object prop, so Svelte sees a new choices list,
// elem_classes array or editor layer/brush option on each form refresh even when
// the content is identical, and re-runs every reactive statement reading it
// (option lists, editor tool setup with its forced layouts). A content-equal
// object prop is the prop the component already holds: do not assign it.
// value and loading_status keep Gradio's handling: a value carries output
// semantics and a status is published once per actual change by Mn.
function wangpSameProp(previous, next) {
    if (previous === null || typeof previous !== "object" || previous instanceof Map || previous instanceof Set || next instanceof Map || next instanceof Set) return false;
    try { return JSON.stringify(previous) === JSON.stringify(next); } catch { return false; }
}
// An unchanged primitive value is the value the bound component already holds
// and shows; Svelte would not invalidate it. Skip it together with its
// value_is_output mark (the mark carries the incoming value), so the component
// runs no update cycle for it: Textbox reads layout before every update and
// Slider rewrites its range input, both forcing a layout of the whole form.
// Object values keep Gradio's handling: components react to their identity.
function wangpSkipUpdate(node, update, next) {
    const prop = update.prop;
    // A label/visibility-only update clears the mark; a node whose mark is already
    // cleared has nothing to apply (Svelte would not invalidate it either).
    if (prop === "value_is_output" && next === false && node.props.value_is_output === false) return true;
    if (prop === "value" || (prop === "value_is_output" && next === true && "wangpValue" in update)) {
        const value = prop === "value" ? next : update.wangpValue;
        return (value === null || typeof value !== "object") && Object.is(node.props.value, value);
    }
    return prop !== "loading_status" && next !== null && typeof next === "object" && wangpSameProp(node.props[prop], next);
}
"""
# Gradio's status store replaces an entry on each actual status/progress change.
# Mn also runs after data messages and visits previously completed outputs. Do
# not publish those same entries again: doing so dirties whole old forms.
# The input map also retains completed entries. Re-publishing pending=False
# after a media-selection callback invalidates its gallery on unrelated events.
_STATUS_ORIGINAL = 'function Mn(S){let J=[];Object.entries(S).forEach(([R,oe])=>{if(d.closed&&oe.status==="error")return;let de=u.find(he=>he.id==oe.fn_index);de!==void 0&&(oe.scroll_to_output=de.scroll_to_output,oe.show_progress=de.show_progress,J.push({id:parseInt(R),prop:"loading_status",value:oe}))});const K=ie.get_inputs_to_update(),pe=Array.from(K).map(([R,oe])=>({id:R,prop:"pending",value:oe==="pending"}));T([...J,...pe])}'
_STATUS_UPDATE = """const wangpStatusCache=new Map(), wangpPendingCache=new Map();
function Mn(S){
    const updates=[], dependencies=new Map(u.map(fn=>[fn.id,fn]));
    for(const [id,status] of Object.entries(S)){
        if(d.closed&&status.status==="error")continue;
        const dependency=dependencies.get(status.fn_index);
        if(dependency===undefined)continue;
        const previous=wangpStatusCache.get(id);
        if(previous&&previous[0]===status&&previous[1]===dependency.scroll_to_output&&previous[2]===dependency.show_progress)continue;
        status.scroll_to_output=dependency.scroll_to_output;
        status.show_progress=dependency.show_progress;
        wangpStatusCache.set(id,[status,status.scroll_to_output,status.show_progress]);
        updates.push({id:parseInt(id),prop:"loading_status",value:status});
    }
    const inputs=[];
    for(const [id,status] of ie.get_inputs_to_update()){
        const pending=status==="pending";
        if(wangpPendingCache.get(id)===pending)continue;
        wangpPendingCache.set(id,pending);
        inputs.push({id,prop:"pending",value:pending});
    }
    T([...updates,...inputs]);
}
"""

# Reuse a thumbnail's decoded first frame immediately, before loading the player.
# Posters are bounded to preview size and cached on thumbnail nodes (so removing
# media also releases its cache). Never remove a paused Chrome player's poster:
# loadeddata can precede painting and removing it can leave the player blank.
_GALLERY_VIDEO_SOURCE = Path(__file__).with_name('gallery_save.js').read_text(encoding='utf-8') + """
const wangpGalleryFrames = new WeakMap();
const wangpGalleryPosters = new WeakMap();
const wangpGalleryEmptyPoster = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/%3E";
function wangpGalleryVideoPoster(video) {
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    // Encode off the click path; toDataURL synchronously stalls the UI here.
    return new Promise(resolve => canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(blob);
    }, "image/jpeg", 0.9));
}
function wangpGalleryVideoMount(video) {
    if (video.dataset.testid?.startsWith("thumbnail ")) wangpGalleryThumbnailPoster(video, video.src);
    if (video.dataset.testid === "detailed-video") queueMicrotask(() => {
        if (video.isConnected) wangpGalleryVideoSource(video, video.src);
    });
}
function wangpGalleryThumbnailPoster(video, src) {
    const url = new URL(src, document.baseURI);
    if (url.origin !== location.origin || !url.pathname.includes('/file=')) return;
    // Set this during mount, before the browser starts resource selection.
    video.preload = 'none';
    url.searchParams.set('__wangp_gallery_preview', 'video');
    video.poster = url.href;
}
function wangpGalleryPosterEntry(video, src) {
    const entry = {src, value: null};
    entry.poster = wangpGalleryVideoPoster(video).then(value => entry.value = value);
    return entry;
}
function wangpGalleryVideoClear(video) {
    const state = wangpGalleryFrames.get(video);
    state?.saveCleanup();
    if (state?.request != null) video.cancelVideoFrameCallback(state.request);
    if (state?.ready) video.removeEventListener("loadeddata", state.ready);
    wangpGalleryFrames.delete(video);
    video.removeEventListener("error", wangpGalleryVideoError);
}
function wangpGalleryVideoError(event) {
    wangpGalleryVideoClear(event.currentTarget);
    event.currentTarget.removeAttribute("poster");
}
function wangpGalleryVideoSource(video, src) {
    if (video.dataset.testid?.startsWith("thumbnail ")) wangpGalleryThumbnailPoster(video, src);
    if (video.dataset.testid !== "detailed-video" || !video.closest(".gallery-container")) {
        j(video, "src", src);
        return;
    }
    wangpGalleryVideoClear(video);
    const requestedSrc = new URL(src, document.baseURI).href;
    const thumbnail = Array.from(video.closest(".gallery-container").querySelectorAll(".thumbnail-small video, .thumbnail-lg video")).find(item => item.src === requestedSrc);
    let cached = thumbnail && wangpGalleryPosters.get(thumbnail);
    if (cached?.src !== requestedSrc) cached = null;
    if (!cached && thumbnail?.currentSrc === requestedSrc && thumbnail.readyState >= 2) {
        cached = wangpGalleryPosterEntry(thumbnail, requestedSrc);
        wangpGalleryPosters.set(thumbnail, cached);
    }
    // A poster survives a video src change. Never show the previous media while
    // a new thumbnail is being encoded or its first frame is still loading.
    video.poster = cached?.value || wangpGalleryEmptyPoster;
    if (video.src !== requestedSrc) j(video, "src", src);
    const state = {request: null, presented: false};
    wangpGalleryFrames.set(video, state);
    wangpGallerySave(video, state);
    video.addEventListener("error", wangpGalleryVideoError);
    function show(poster) {
        poster.then(value => {
            if (wangpGalleryFrames.get(video) !== state || video.src !== requestedSrc) return;
            video.poster = value;
        });
    }
    if (cached) {show(cached.poster); return;}
    state.ready = () => {
        if (!state.presented || video.readyState < 2 || video.currentSrc !== requestedSrc) return;
        video.removeEventListener("loadeddata", state.ready);
        const entry = wangpGalleryPosterEntry(video, requestedSrc);
        if (thumbnail) wangpGalleryPosters.set(thumbnail, entry);
        show(entry.poster);
    };
    function presented() {
        state.request = null;
        // A playing source may still have an old frame awaiting composition.
        if (video.currentSrc !== requestedSrc) {
            state.request = video.requestVideoFrameCallback(presented);
            return;
        }
        state.presented = true;
        state.ready();
    }
    // These signals can arrive in either order, even while the video is paused.
    video.addEventListener("loadeddata", state.ready);
    state.request = video.requestVideoFrameCallback(presented);
}
"""

_PATCHES = {
    'index-Dk7BRz4q.js': [
        # Hidden progress still mounted the animated Gradio logo and started
        # its elapsed-time loop on every timer request. Keep both dormant.
        ('function u(f,_){return f[4]==="pending"?0:f[4]==="error"?1:-1}', 'function u(f,_){return f[6]==="hidden"?-1:f[4]==="pending"?0:f[4]==="error"?1:-1}'),
        ('i.$$.dirty[0]&16&&(a==="pending"?Re():_e())', 'i.$$.dirty[0]&80&&(a==="pending"&&k!=="hidden"?Re():_e())'),
        # A removed logo's awaited spring sequence kept running on detached
        # nodes. Stop its springs and do not start the next animation phase.
        ('async function f(){await Promise.all([n.set([125,140]),r.set([-125,-140])]),await Promise.all([n.set([-125,140]),r.set([125,-140])]),await Promise.all([n.set([-125,0]),r.set([125,-0])]),await Promise.all([n.set([125,0]),r.set([-125,0])])}', 'async function f(){for(const [left,right] of [[[125,140],[-125,-140]],[[-125,140],[125,-140]],[[-125,0],[125,0]],[[125,0],[-125,0]]]){if(u)return;await Promise.all([n.set(left),r.set(right)])}}'),
        ('return rt(()=>(a(),()=>u=!0))', 'return rt(()=>(a(),()=>{u=!0;n.set([0,0],{hard:!0});r.set([0,0],{hard:!0})}))'),
    ],
    'Textbox-pyOssf2O.js': [
        # The autoscroll check reads the textarea's scroll geometry before every
        # update, forcing a layout of the whole form between the flush's DOM
        # writes. Only a value change can move the scroll position: a label or
        # visibility update needs no read and performs no scroll.
        ('tt(()=>{x=v&&v.offsetHeight+v.scrollTop>v.scrollHeight-100})', 'tt(()=>{x=v&&(l.$$.dirty[0]&1)!==0&&v.offsetHeight+v.scrollTop>v.scrollHeight-100})'),
    ],
    'AudioPlayer-DG1QBBp6.js': [
        # Svelte invalidates timeRef after this DOM write, re-running the
        # reactive waveform.on() statement and leaking a listener each tick.
        # Update the DOM without invalidating the ref; avoid same-time writes.
        ('y&&t(12,y.textContent=ze(m),y)', 'y&&y.textContent!==ze(m)&&(y.textContent=ze(m))'),
        ('b&&t(13,b.textContent=ze(m),b)', 'b&&b.textContent!==ze(m)&&(b.textContent=ze(m))'),
    ],
    'Video-C-llMUaJ.js': [
        ('function ki(t){', _GALLERY_VIDEO_SOURCE + 'function ki(t){'),
        ('t[25](e),s=!0', 't[25](e),wangpGalleryVideoMount(e),s=!0'),
        ('&&j(e,"src",p),(!s||c&16)', '&&wangpGalleryVideoSource(e,p),(!s||c&16)'),
        ('d(l){l&&(A(i),A(a),A(e))', 'd(l){wangpGalleryVideoClear(e);l&&(A(i),A(a),A(e))'),
    ],
    'utils-BsGrhMNe.js': [
        # Round once before splitting units so 119.999 seconds displays as 2:00.
        ('const w=t=>{const o=Math.floor(t/3600)', 'const w=t=>{t=Math.round(t);const o=Math.floor(t/3600)'),
    ],
    'Dropdown-DSZkNuau.js': [
        ('function ce(l,t,e){', Path(__file__).with_name('model_status.js').read_text(encoding='utf-8') + '\nfunction ce(l,t,e){'),
        ('X(t,u),X(t,r)},p(o,a){', 'X(t,u),X(t,r),He(u,wangpModelLabel(t,h))},p(o,a){'),
        ('&&He(u,h),a&2&&n', '&&He(u,wangpModelLabel(t,h)),a&2&&n'),
        ('me(n,l[10]),l[30](n)', 'wangpModelInput(n,l[10]),l[30](n)'),
        ('c[0]&1024&&n.value!==i[10]&&me(n,i[10])', 'c[0]&1024&&wangpModelInput(n,i[10])'),
    ],
    'Tabs-DtiML6DO.js': [
        # Every tab registration during a form refresh starts a measurement after
        # the same tick; each one reads layout between the flush's DOM writes.
        # One measurement after that tick sees the same final DOM.
        ('async function V(){if(!w)return;await We();', 'let wangpTabsMeasuring=false;async function V(){if(!w||wangpTabsMeasuring)return;wangpTabsMeasuring=true;await We();wangpTabsMeasuring=false;'),
        # Test whether all tabs fit without the overflow trigger first. Its
        # existing width otherwise keeps the last tab trapped in overflow.
        ('let p=s.width;const D=ye(b,J);', '''if (!s.width) return;
        const overflowHidden = I.classList.contains("hide");
        I.classList.remove("hide");
        const overflowWidth = I.getBoundingClientRect().width + parseFloat(getComputedStyle(I).marginLeft);
        if (overflowHidden) I.classList.add("hide");
        let p=s.width + (overflowHidden ? 0 : overflowWidth);const D=ye(b,J);
        if (b.some(tab => tab?.visible && D[tab.id] && D[tab.id].right - s.left > p)) p -= overflowWidth;'''),
        ('le&&le.right-ze<p', 'le&&le.right-ze<=p'),
        # Overflow choices must keep the same visibility and availability as
        # their full-size tabs, including after model-dependent updates.
        ('l(8,T=b.slice(W+1))', 'l(8,T=b.slice(W+1).filter(tab=>tab?.visible))'),
        ('S(e,"selected",t[33]?.id===t[6])},m(d,u)', 'S(e,"selected",t[33]?.id===t[6]),e.disabled=!t[33].interactive,m(e,"aria-disabled",e.disabled)},m(d,u)'),
        ('u[0]&320&&S(e,"selected",t[33]?.id===t[6])', 'u[0]&320&&S(e,"selected",t[33]?.id===t[6]),u[0]&256&&(e.disabled=!t[33].interactive,m(e,"aria-disabled",e.disabled))'),
    ],
    'Image-CnqB5dbD.js': [
        ('function k(s){', Path(__file__).with_name('image_zoom.js').read_text(encoding='utf-8') + '\nfunction k(s){'),
        ('x(t,e,o),i||', 'x(t,e,o),window.WanGPImageZoom.mount(e),i||'),
        ('d(e,"svelte-1pijsyv",!0)},i:m', 'd(e,"svelte-1pijsyv",!0),window.WanGPImageZoom.mount(e)},i:m'),
        ('d(t){t&&b(e),i=!1', 'd(t){window.WanGPImageZoom.destroy(e);t&&b(e),i=!1'),
    ],
    'Upload-B_zkKtR-.js': [
        ('function it(){', _GALLERY_DROP_FILES + 'function it(){'),
        ('function s(f){if(f.preventDefault(),f.stopPropagation(),e.on_drag_change?.(!1),!f.dataTransfer?.files)return;const c=Array.from(f.dataTransfer.files);c.length>0&&e.on_files?.(c)}',
         'async function s(f){if(f.preventDefault(),f.stopPropagation(),e.on_drag_change?.(!1),!f.dataTransfer?.files)return;try{const c=await wangpGalleryDropFiles(f);c.length>0&&e.on_files?.(c)}catch(error){e.on_error?.(error.message)}}'),
        ('on_files:t[31],accepted_types:', 'on_files:t[31],on_error:t[32],accepted_types:'),
        ('on_files:a[31],accepted_types:', 'on_files:a[31],on_error:a[32],accepted_types:'),
        ('l,o,i,qe]}class Pt', 'l,o,i,qe,message=>A("error",message)]}class Pt'),
        ('const h=Array.from(r.dataTransfer.files).filter(ae);', 'let h;try{h=(await wangpGalleryDropFiles(r)).filter(ae)}catch(error){A("error",error.message);return;}'),
    ],
    'Gallery-D7vc32lN.js': [
        ('function Ve(s){', '''function wangpGalleryGap(event) {
            const container = event.target.closest('.thumbnails, .grid-container');
            if (!container || !this.contains(container) || event.target.closest('.thumbnail-item')) return;
            let closest, distance = Infinity;
            for (const button of container.querySelectorAll('.thumbnail-item')) {
                const box = button.getBoundingClientRect();
                const dx = Math.max(box.left - event.clientX, 0, event.clientX - box.right);
                const dy = Math.max(box.top - event.clientY, 0, event.clientY - box.bottom);
                const next = dx * dx + dy * dy;
                if (next < distance) {closest = button; distance = next;}
            }
            if (closest) {event.preventDefault(); event.stopPropagation(); closest.click();}
        }
        function wangpGalleryDrag(event) {
            const thumbnail = event.target.closest('.thumbnail-item[data-wangp-video-url]');
            if (!thumbnail || !this.contains(thumbnail)) return;
            const source = new URL(thumbnail.dataset.wangpVideoUrl, location.href).href;
            event.dataTransfer.effectAllowed = 'copy';
            event.dataTransfer.setData('application/x-wangp-gallery-media', source);
            event.dataTransfer.setData('text/uri-list', source);
            event.dataTransfer.setData('text/plain', source);
        }
        function Ve(s){'''),
        ('V=s,l(18,V)', 'V?.removeEventListener("click",wangpGalleryGap),V?.removeEventListener("dragstart",wangpGalleryDrag),V=s,V?.addEventListener("click",wangpGalleryGap),V?.addEventListener("dragstart",wangpGalleryDrag),l(18,V)'),
        ('B(e,"class","thumbnail-item thumbnail-small svelte-842rpi")', 'B(e,"class","thumbnail-item thumbnail-small svelte-842rpi"),wangpGalleryVideoThumbnail(e,n[50])'),
        ('B(e,"class","thumbnail-item thumbnail-lg svelte-842rpi")', 'B(e,"class","thumbnail-item thumbnail-lg svelte-842rpi"),wangpGalleryVideoThumbnail(e,n[47])'),
        ('n=w;let m=l;', 'n=w;y[0]&65536&&wangpGalleryVideoThumbnail(e,n[50]);let m=l;'),
        ('n=$;let y=l;', 'n=$;w[0]&65536&&wangpGalleryVideoThumbnail(e,n[47]);let y=l;'),
        ('function dt(n){', '''function wangpGalleryVideoThumbnail(button, item) {
            B(button, 'draggable', item.video ? 'true' : null);
            B(button, 'data-wangp-video-url', item.video?.url);
        }
        function wangpGalleryImage(src, edge) {
            if (!src || !edge) return src;
            const url = new URL(src, window.location.href);
            if (url.origin !== window.location.origin || !url.pathname.includes('/file=') || !/\\.(png|jpe?g|webp|bmp|tiff?|gif)$/i.test(url.pathname)) return src;
            url.searchParams.set('__wangp_gallery_preview', edge);
            return url.href;
        }
        function dt(n){'''),
        ('src:n[22].image.url,alt:', 'src:wangpGalleryImage(n[22].image.url,n[17]?0:1600),alt:'),
        ('o[0]&4194304&&(i.src=t[22].image.url)', 'o[0]&4325376&&(i.src=wangpGalleryImage(t[22].image.url,t[17]?0:1600))'),
        ('src:n[50].image.url,title:', 'src:wangpGalleryImage(n[50].image.url,320),title:'),
        ('o[0]&65536&&(i.src=t[50].image.url)', 'o[0]&65536&&(i.src=wangpGalleryImage(t[50].image.url,320))'),
        ('src:typeof n[47].image=="string"?n[47].image:n[47].image.url', 'src:wangpGalleryImage(typeof n[47].image=="string"?n[47].image:n[47].image.url,320)'),
        ('i.src=typeof t[47].image=="string"?t[47].image:t[47].image.url', 'i.src=wangpGalleryImage(typeof t[47].image=="string"?t[47].image:t[47].image.url,320)'),
        # Selection is a user event. Server values/indices notify change once,
        # after normalization; index-only updates must still refresh consumers.
        ('let ne=m;function se(s){', 'let ne=m,wangpGalleryUser=false,wangpGalleryExplicit=false,wangpGalleryChanged=false;function se(s){'),
        # Preview clicks zoom the current image; thumbnails/arrow keys select.
        ('function se(s){const S=s.target,H=s.offsetX,X=S.offsetWidth/2;H<X?l(1,m=t):l(1,m=o)}', 'function se(s){(document.fullscreenElement?document.exitFullscreen():V.requestFullscreen()).catch(console.error)}'),
        # Restoring or selecting a thumbnail must not scroll the whole page.
        ('W[s]?.focus();', 'W[s]?.focus({preventScroll:true});'),
        # The selected image is wanted now; only offscreen thumbnails are lazy.
        ('class:n[22].caption&&"with-caption",loading:"lazy"', 'class:n[22].caption&&"with-caption",loading:"eager"'),
        # Preview already has a thumbnail strip. Do not mount a second full
        # media grid behind it; retain the empty wrapper's height for layout.
        ('b=te(n[16]),u=[];', 'b=te(n[22]&&n[7]?[]:n[16]),u=[];'),
        ('_[0]&8454274){b=te(a[16]);', '_[0]&12648706){b=te(a[22]&&a[7]?[]:a[16]);'),
        # Do not animate the strip when the selected thumbnail is already visible.
        ('Q=x-S+X/2-H/2+A.scrollLeft;A&&', 'Q=x-S+X/2-H/2+A.scrollLeft;if(x>=S&&x+X<=S+H)return;A&&'),
        # Escape exits fullscreen without discarding the selected preview and
        # exposing the large grid underneath. Preview=False galleries can close.
        ('function Re(s){switch(s.code){', 'function Re(s){if(s.code==="Escape"&&a){s.preventDefault();if(document.fullscreenElement)document.exitFullscreen().catch(console.error);return;}if(["Escape","ArrowLeft","ArrowRight"].includes(s.code))wangpGalleryUser=true;switch(s.code){'),
        ('const qe=s=>l(1,m=s);', 'const qe=s=>{wangpGalleryUser=true;return l(1,m=s)};'),
        ('Oe=s=>{m===null', 'Oe=s=>{wangpGalleryUser=true;m===null'),
        ('Ge=()=>{l(1,m=null)', 'Ge=()=>{wangpGalleryUser=true;l(1,m=null)'),
        ('n.$$set=s=>{"show_label"', 'n.$$set=s=>{wangpGalleryExplicit||="selected_index"in s&&("value"in s||s.selected_index!==m);if(wangpGalleryUser&&(("selected_index"in s&&s.selected_index!==m)||("value"in s&&!et(r,s.value))))wangpGalleryUser=false;"show_label"'),
        ('K?(l(1,m=a&&r?.length?0:null),l(29,K=!1))', 'K?(!wangpGalleryExplicit&&l(1,m=a&&r?.length?0:null),l(29,K=r==null||r.length===0))'),
        ('Math.min(m,r.length-1))):l(1,m=null)', 'Math.min(m,r.length-1))):l(1,m=a&&r?.length?0:null)'),
        ('J("change"),l(30,le=r)', 'wangpGalleryChanged=true,l(30,le=r)'),
        ('(l(31,ne=m),m!==null&&(P!=null&&l(1,m=Math.max(0,Math.min(m,P.length-1))),J("select",{index:m,value:P?.[m]})))', '(m!==null&&P!=null&&l(1,m=Math.max(0,Math.min(m,P.length-1))),l(31,ne=m),wangpGalleryUser?(m!==null&&J("select",{index:m,value:P?.[m]})):wangpGalleryChanged=true)'),
        ('l(22,i=m!=null&&P!=null?P[m]:null)},[r,m', 'l(22,i=m!=null&&P!=null?P[m]:null);if(wangpGalleryChanged)J("change");wangpGalleryChanged=wangpGalleryUser=wangpGalleryExplicit=false},[r,m'),
        # Full-resolution media belongs only to the gallery being enlarged.
        # Release the listener when a form rebuild destroys this component.
        ('_t(()=>{document.addEventListener("fullscreenchange",()=>{l(17,j=!!document.fullscreenElement)})})', '_t(()=>{const changed=()=>l(17,j=document.fullscreenElement===V);document.addEventListener("fullscreenchange",changed);return()=>document.removeEventListener("fullscreenchange",changed)})'),
    ],
    'index.js': [
        ('function yq() {', _GALLERY_DROP_FILES + 'function yq() {'),
        ('''function l(f) {
        if (f.preventDefault(), f.stopPropagation(), e.on_drag_change?.(!1), !f.dataTransfer?.files) return;
        const m = Array.from(f.dataTransfer.files);
        m.length > 0 && e.on_files?.(m);
      }''', '''async function l(f) {
        if (f.preventDefault(), f.stopPropagation(), e.on_drag_change?.(!1), !f.dataTransfer?.files) return;
        try {
          const m = await wangpGalleryDropFiles(f);
          if (m.length > 0) await e.on_files?.(m);
        } catch (error) { e.on_error?.(error.message); }
      }'''),
        ('r[33]\n        ),\n        accepted_types:', 'r[33]\n        ),\n        on_error: message => r[32]("download_error", message),\n        accepted_types:'),
        ('E[33]\n        ),\n        accepted_types:', 'E[33]\n        ),\n        on_error: message => E[32]("download_error", message),\n        accepted_types:'),
        ('jC.SchedulerSystem = XC;', _EDITOR_SCHEDULER + 'jC.SchedulerSystem = XC;'),
        ('this._pauseUpdate = e;', 'this._pauseUpdate = e; e ? this.removeTickerListener() : this.addTickerListener();'),
        ('this._tickerAdded || !this.domElement ||', 'this._pauseUpdate || this._tickerAdded || !this.domElement ||'),
        # Every form refresh marks the editor value as output again (afterUpdate
        # clears the flag), which re-runs the value sync with force_empty. The
        # forced clear reset both Pixi canvases, rebuilt their cursors and read
        # layout for each tool change even when nothing was drawn or loaded.
        # Clear only when the editors hold something, as a user clear does.
        ('if (await R0(), !wn() || (await Q(!0, !1, wn), !wn())) return;', 'if (await R0(), !wn() || (await Q(!1, !1, wn), !wn())) return;'),
    ],
    'Blocks-BMC4HgbM.js': [
        ('requestAnimationFrame(()=>{Jt(he,pe,oe)})', 'wangpScheduleUpdate(()=>{Jt(he,pe,oe)})'),
        ('async function No(S,J,K){', 'const wangpMetadataPending=new Map(),wangpMetadataSent=new Map();async function No(S,J,K){'),
        ('const R=pe;if(dn.length>0)', 'const R=pe;' + metadata_events.PREPARE_JS + 'if(dn.length>0)'),
        ('R.inputs.map(W=>No(W,J,K))', 'R.inputs.map((W,index)=>wangpMetadata&&index===1?wangpMetadata:No(W,J,K))'),
        # Gradio only clears a queued interaction on a data response. A failed
        # request has no data, so a second click can leave pending_request set
        # forever. Retire it on failure so the next user gesture can submit.
        ('else if(ne.stage==="error"){', 'else if(ne.stage==="error"){R.pending_request=!1;R.final_event=null;wangpMetadataSent.delete(S);'),
        ('if(d.closed)return;t(21,ce=[st("Error",String(ae)', 'R.pending_request=!1;R.final_event=null;wangpMetadataSent.delete(S);if(d.closed)return;t(21,ce=[st("Error",String(ae)'),
        # Gradio already shows its lost-connection status. A failed request for
        # each pending event would otherwise add the same error toast again.
        ('if(ne.message){const ge=ne.message.replace(rf,', 'if(ne.message&&!ne.message.startsWith("Connection errored out.")){const ge=ne.message.replace(rf,'),
        ('function Jt(S,J=null,K=null){', 'function Jt(S,J=null,K=null){if(window.__wangpGradioStale)return;'),
        # Hide the API footer fragment (including its divider), not the API.
        ('y=l[5]&&Qi(l);', 'y=false;'),
        ('b[5]?y?y.p(b,q):(y=Qi(b),y.c(),y.m(e,t)):y&&(y.d(1),y=null),', ''),
        ('l[22]("common.built_with_gradio")+""', '"Powered by Gradio-GP"'),
        ('b[22]("common.built_with_gradio")+""', '"Powered by Gradio-GP"'),
        # Reuse the native settings action from the credit link. Omit the
        # separate settings button/divider and the panel's PWA section.
        ('le(n,"href","https://gradio.app")', 'le(n,"href","?view=settings")'),
        ('le(n,"target","_blank"),le(n,"rel","noreferrer"),', ''),
        ('ue(e,_),ue(e,u),ue(e,f),ue(e,p),ue(p,h),ue(p,$),ue(p,m),', ''),
        ('v=Nn(p,"click",l[44])', 'v=Nn(n,"click",event=>{event.preventDefault();l[44]()})'),
        ('Je(q,f,j),Je(q,p,j),ve(p,g),ve(g,$),ve(p,m),ve(p,w),b.m(w,null),', ''),
        (_STATUS_ORIGINAL, _STATUS_UPDATE),
        # A label/visibility/options update carries no new value. Marking it as
        # one makes editors interpret their old empty prop as a server clear.
        ('S?.map((de,he)=>({id:pe[he],prop:"value_is_output",value:!0}))',
         'S?.map((de,he)=>({id:pe[he],prop:"value_is_output",value:!(de&&de.__type__==="update"&&!Object.prototype.hasOwnProperty.call(de,"value")),wangpValue:de&&de.__type__==="update"?de.value:de}))'),
        # Re-publish after a component clears its indicator or a layout rebuild.
        ('function Ao(S,J,K){', 'function Ao(S,J,K){wangpStatusCache.delete(String(S));'),
        ('function Bo(ae){', 'function Bo(ae){wangpStatusCache.clear();wangpPendingCache.clear();'),
        # Own the wrapper's props once instead of copying the accumulated object
        # on every $set. Rest props are still freshly derived by Svelte's ji().
        ('function $u(l,e,t){', 'function $u(l,e,t){e=el({},e);'),
        ('e=el(el({},e),au(b)),t(9,s=ji(e,n))', 'el(e,au(b)),t(9,s=ji(e,n))'),
        ('function Pu(l,e,t){', 'function Pu(l,e,t){' + _GRADIO_CONTEXT + 'let wangpNodeVersion=e.node.__wangp_revision,wangpNodeInputs={...e};'),
        ('new er(s.id,o,a,c,n,r,_,eu,u,Yo)', 'wangpGradio(s.id,o,a,c,n,r,_,eu,u,Yo)'),
        ('return l.$$set=w=>{' + _NODE_INPUTS, 'return l.$$set=w=>{' + _NODE_SKIP + _NODE_INPUTS),
    ],
    'index-Do3LSwBC.js': [
        # UI submissions include State slots and private events absent from
        # /info. Their dependency inputs describe the actual positional payload.
        # Named API calls retain the public parameter/keyword validation.
        ('const Ig=(e=[],o)=>{', 'const Ig=(e=[],o,wangpInputCount)=>{'),
        ('return e.length>t.length&&console.warn("Too many arguments provided for the endpoint."),e;', 'return e.length>(wangpInputCount??t.length)&&console.warn("Too many arguments provided for the endpoint."),e;'),
        ('P=Ig(o,k)', 'P=Ig(o,k,typeof e==="number"?g.inputs.length:undefined)'),
        ('function Qo(e,o,t,r){', _UPDATE_SCHEDULER + 'function Qo(e,o,t,r){'),
        ('uv(q)', 'wangpScheduleUpdate(q)'),
        ('function lf(e,o={}){', 'function lf(e,o={}){' + _NATIVE_EVENT_STREAM),
        # Deliver every partial output in its own task: gr.skip() must not erase
        # an earlier queue/abort trigger, and consumers must read object updates
        # before the next diff mutates them. Posted tasks avoid hidden-tab timer
        # throttling while preserving Gradio's microtask boundaries.
        ('let h=e[_];typeof window<"u"&&typeof document<"u"?setTimeout(h,0,u):h(u)', 'let h=e[_];typeof window<"u"&&typeof document<"u"?wangpScheduleUpdate(()=>h(u)):h(u)'),
        # Inspect only the actual gallery-view output, never unrelated text.
        # Its whole response must be discarded before an old index can paint.
        ('for(let g=0;g<ge.length;g++)for(let P=0;', 'for(let g=0;g<ge.length;g++){const view=ge[g].find(v=>v&&v.prop==="value"&&s[v.id]?.props.elem_id==="wangp-gallery-view");if(view&&window.WanGPGallerySelection?.acceptView(view.value)===false)continue;for(let P=0;'),
        ('function H(k){let g=o.get(k);', 'function H(k,wangpKind){let g=o.get(k);'),
        ('g=G(P,k)}return g?g.instance?.get_value', 'g=G(P,k)}' + metadata_events.READ_JS + 'return g?g.instance?.get_value'),
        ('let s=null,d=new URLSearchParams({session_hash:this.session_hash}).toString()',
         'let s=null,d=new URLSearchParams({session_hash:this.session_hash,stream_reuse:"1"}).toString()'),
        # A reused stream can deliver a fast result before /queue/join returns.
        # Register before replaying buffered messages so completion can remove
        # the callback and pending ID instead of resurrecting them afterwards.
        ('I in z&&(z[I].forEach(Xe=>Ao(Xe)),delete z[I]),q[I]=Ao,D.add(I),C.open||await this.open_stream()',
         'q[I]=Ao,D.add(I),I in z&&(z[I].forEach(Xe=>{Xe.msg==="process_completed"&&D.delete(I);Ao(Xe)}),delete z[I]),D.size>0&&!C.open&&await this.open_stream()'),
        # If a new job joins while the idle close is in transit, reopen for it.
        # Let earlier completion callbacks retire first; idle sessions stay shut.
        ('if(u.msg==="close_stream"){ze(r,n.abort_controller);return}',
         'if(u.msg==="close_stream"){ze(r,n.abort_controller);setTimeout(()=>{if(!n.closed&&!r.open&&Object.keys(e).some(id=>o.has(id)))n.open_stream()},0);return}'),
        # A server restart also closes the stream without a close_stream message.
        # Clear its open flag so the next queued audio/UI update can reconnect.
        ('s.onerror=async function(){await Promise.all', 's.onerror=async function(){ze(r,n.abort_controller);await Promise.all'),
        ('function q(){l.update(k=>{for(let g=0;', 'function q(){l.update(k=>{const wangpDirty=new Set;for(let g=0;'),
        (':j=v.value,f.props[v.prop]=j}return k}),ge=[]', ':j=v.value;if(wangpSkipUpdate(f,v,j))continue;f.props[v.prop]=j;' + _MARK_ANCESTORS + 'return k}),ge=[]'),
        ('l.set(h)', 'Object.values(s).forEach(node=>node.__wangp_revision=(node.__wangp_revision||0)+1),l.set(h)'),
    ],
    'Index-Cuq53rtW.js': [
        # This handshake is for a Hugging Face iframe host. A standalone page
        # has no parent receiver and Chrome reports a target-origin mismatch.
        ('window.parent.postMessage(b,Te)', 'window.parent!==window&&window.parent.postMessage(b,Te)'),
    ],
}


@lru_cache(maxsize=len(_PATCHES))
def _asset(path):
    source = Path(path).read_text(encoding='utf-8')
    name = Path(path).name
    replacements = list(_QUEUE_REPLACEMENTS.items()) if name.startswith('Blocks-') else []
    replacements += _PATCHES[name]
    for old, new in replacements:
        expected = 2 if old == 'l.set(h)' else 1
        if source.count(old) != expected:
            raise RuntimeError('Gradio frontend patches require the pinned Gradio 5.29 and WanGP editor assets')
        source = source.replace(old, new)
    if name == 'Blocks-BMC4HgbM.js':
        source = 'import {wangpScheduleUpdate} from "./index-Do3LSwBC.js";\n' + source
    return source


_BOOT_SCRIPT = re.compile(r'<script type="module" crossorigin src="(?P<base>\./assets/)(?P<name>index-[^"]+\.js)"></script>')


def _ui_signature(config):
    # Initial values/choices, visibility and styling can change without changing
    # the wire contract. Hash only at app construction, never per page/request.
    # Asset hashes version browser imports separately; a display-only patch does
    # not invalidate an existing page's component IDs or event bindings.
    # ImageEditor.type converts decoded images to PIL/numpy/filepaths in Python;
    # the browser always exchanges the same EditorData structure.
    components = [
        [component['id'], component['type'], component['component_class_id'], component['key'],
         {key: component['props'][key] for key in ('elem_id', 'type', 'multiselect', 'file_count')
          if key in component['props'] and not (key == 'type' and component['type'] in ('imageeditor', 'wangpimageeditor'))}]
        for component in config['components']
    ]
    contract = [config['version'], config['protocol'], config['api_prefix'], components, config['layout'], config['dependencies']]
    return 'ui-' + sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class _BrowserInstanceGuard:
    def __init__(self, app, signature):
        self.app, self.signature = app, signature.encode()

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            instance = next((value for key, value in scope['headers'] if key == b'x-wangp-app-id'), None)
            if instance is not None and instance != self.signature:
                response = JSONResponse({'error': 'The interface changed. Reload this page.'}, status_code=409, headers={'X-WanGP-Reload': '1'})
                return await response(scope, receive, send)
        return await self.app(scope, receive, send)


def _version_html(source, versions):
    def replace(match):
        base = match['base']
        imports = {base + name: f'{base}{name}?__wangp_ui={version}' for name, version in versions.items()}
        # All imports of a patched chunk must resolve to the SAME module, including
        # imports from unchanged chunks. Rewriting only dynamic imports duplicates
        # Gradio's stores and breaks event ordering.
        import_map = '<script type="importmap">' + json.dumps({'imports': imports}) + '</script>'
        return import_map + match[0].replace(base + match['name'], imports[base + match['name']])

    return _BOOT_SCRIPT.sub(replace, source)


def install():
    metadata_events.install()
    original = routes.FileResponse
    if getattr(original, '_wangp_frontend', False):
        return
    # Validate the pinned assets at startup rather than fail a browser import.
    asset_paths = {_EDITOR_PATH if name == 'index.js' else Path(routes.BUILD_PATH_LIB) / name for name in _PATCHES}
    for path in asset_paths:
        _asset(str(path))

    @wraps(original)
    def file_response(path, *args, **kwargs):
        if Path(path) in asset_paths:
            return Response(_asset(str(path)), media_type='application/javascript', headers={'Cache-Control': 'no-store'})
        return original(path, *args, **kwargs)

    file_response._wangp_frontend = True
    routes.FileResponse = file_response

    versions = {path.name: sha256(_asset(str(path)).encode()).hexdigest()[:16] for path in asset_paths if path != _EDITOR_PATH}
    original_template = routes.templates.TemplateResponse
    session_script = Path(__file__).with_name('session_guard.js').read_text(encoding='utf-8')
    proxy_root_script = Path(__file__).with_name('proxy_root.js').read_text(encoding='utf-8')

    @wraps(original_template)
    def template_response(*args, **kwargs):
        response = original_template(*args, **kwargs)
        source = response.body.decode('utf-8')
        patched = _version_html(source, versions)
        if patched != source:
            patched = patched.replace('<script type="importmap">', '<script>' + proxy_root_script + '</script><script type="importmap">', 1)
            config = response.context['config']
            # Gradio advertises this URL even when its route returns 404.
            if not config.get('pwa'):
                patched = patched.replace('<link rel="manifest" href="/manifest.json" />', '')
            if not config.get('auth_required'):
                guard = session_script.replace('__WANGP_UI_SIGNATURE__', json.dumps(config['wangp_ui_signature']))
                patched = patched.replace('<script type="importmap">', '<script>' + guard + '</script><script type="importmap">', 1)
            response.body = patched.encode('utf-8')
            response.headers['content-length'] = str(len(response.body))
            response.headers['cache-control'] = 'no-store'
        return response

    routes.templates.TemplateResponse = template_response

    original_create_app = routes.App.create_app

    @wraps(original_create_app)
    def create_app(*args, **kwargs):
        app = original_create_app(*args, **kwargs)
        blocks = app.get_blocks()
        # Blocks.__init__ creates a provisional app before the UI exists.
        if hasattr(blocks, 'config'):
            config = blocks.config
            config['wangp_ui_signature'] = _ui_signature(config)
            app.add_middleware(_BrowserInstanceGuard, signature=config['wangp_ui_signature'])
        return app

    routes.App.create_app = staticmethod(create_app)
