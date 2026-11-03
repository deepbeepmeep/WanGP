// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import {I as WangpFrameIconButton} from "./IconButton-DSs-aBrk.js";

function wangpSubtitlesShown() {
    try { return localStorage.getItem("wangp-subtitles") !== "off"; } catch (_) { return true; }
}

// Subtitles of a video: its .srt side file, as a WebVTT track (browsers ignore the subtitle tracks stored inside the video file).
// The subtitles button appears only when there are some; whether they show is the browser's choice, kept across videos.
async function wangpGallerySubtitles(video, src, button, control, isDisposed) {
    let response;
    try { response = await fetch(src.replace(/\.[^./\\?#]+(?=[?#]|$)/, ".srt")); } catch (_) { return null; }
    if (!response.ok || isDisposed()) return null;
    const text = await response.text();
    if (isDisposed()) return null;
    const url = URL.createObjectURL(new Blob(["WEBVTT\n\n" + text.replace(/\r/g, "").replace(/(\d\d:\d\d:\d\d),(\d\d\d)/g, "$1.$2")], {type: "text/vtt"}));
    const track = Object.assign(document.createElement("track"), {kind: "subtitles", label: "Subtitles", src: url, default: true});
    video.appendChild(track);
    const show = on => {
        track.track.mode = on ? "showing" : "hidden";
        control.$set({highlight: on});
        button.setAttribute("aria-pressed", String(on));
    };
    show(wangpSubtitlesShown());
    button.hidden = false;
    return {track, url, show};
}

// Reuse Gradio's actual icon buttons, including their theme and hover styles.
function wangpGallerySave(video, state) {
    const preview = video.closest(".preview");
    const menu = document.createElement("div");
    menu.className = "icon-button-wrapper svelte-9lsba8 wangp-video-actions no-border";
    menu.setAttribute("role", "group");
    menu.setAttribute("aria-label", "Frame Actions");
    const controls = [];
    for (const [action, label, title, drawing] of [
        ["add", "Add Frame to Image Gallery", "Add Current Frame to Image Gallery", '<path d="M14 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v9M3 17l6-6 5 5M15 19h8m-4-4v8"/><circle cx="15" cy="8" r="1.5"/>'],
        ["save", "Save Image", "Save Current Frame as PNG - Original Resolution", '<path d="M12 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v7M3 17l6-6 4 4M18 14v8m-3-3 3 3 3-3"/><circle cx="15" cy="8" r="1.5"/>'],
        ["subtitles", "Subtitles", "Show or Hide the Subtitles", '<rect x="2.5" y="5" width="19" height="14" rx="2"/><path d="M10.5 10a2.5 2.5 0 1 0 0 4M17 10a2.5 2.5 0 1 0 0 4"/>']
    ]) {
        const control = new WangpFrameIconButton({target: menu, props: {label}});
        controls.push(control);
        const button = menu.lastElementChild;
        button.dataset.action = action;
        button.title = title;
        button.querySelector('div').innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${drawing}</svg>`;
    }
    const status = document.createElement('span');
    status.setAttribute('role', 'status');
    menu.appendChild(status);
    const src = video.src;
    const filename = decodeURIComponent(new URL(src).pathname).split(/[\\/]/).pop();
    const stem = filename.replace(/\.[^.]+$/, "");
    let disposed = false, subtitles = null;
    const subtitlesButton = menu.querySelector('[data-action="subtitles"]');
    subtitlesButton.hidden = true;
    wangpGallerySubtitles(video, src, subtitlesButton, controls[2], () => disposed).then(found => { subtitles = found; });
    // Anchor to the video controls, not the taller gallery/thumbnail area.
    // Gradio fullscreen resizes the video independently of the preview.
    const positionActions = () => {
        if (!menu.offsetParent) return; // The gallery's tab may be hidden.
        const bounds = video.getBoundingClientRect();
        const parent = menu.offsetParent.getBoundingClientRect();
        menu.style.left = `${bounds.right - parent.left - 24}px`;
        menu.style.top = `${bounds.bottom - parent.top - 48}px`;
    };
    const resizeObserver = new ResizeObserver(positionActions);
    state.saveCleanup = () => {
        disposed = true;
        if (subtitles) {
            subtitles.track.remove();
            URL.revokeObjectURL(subtitles.url);
        }
        resizeObserver.disconnect();
        document.removeEventListener('fullscreenchange', positionActions);
        controls.forEach(control => control.$destroy());
        menu.remove();
    };
    menu.addEventListener("keydown", event => {
        event.stopPropagation();
    });
    menu.addEventListener("click", async event => {
        event.stopPropagation();
        const button = event.target.closest("button[data-action]");
        if (!button) return;
        if (button.dataset.action === "subtitles") {
            const on = !wangpSubtitlesShown();
            try { localStorage.setItem("wangp-subtitles", on ? "on" : "off"); } catch (_) {}
            subtitles?.show(on);
            return;
        }
        status.textContent = "";
        const canvas = document.createElement("canvas");
        try {
            if (video.readyState < 2 || video.seeking || video.currentSrc !== src) {
                throw new Error("Wait for the Video Frame to Load, Then Try Again.");
            }
            const time = video.currentTime;
            if (button.dataset.action === "add") {
                const request = document.querySelector('#wangp-gallery-add-frame textarea');
                const view = document.querySelector('#wangp-gallery-view textarea');
                const thumbnails = [...video.closest('.gallery-container').querySelectorAll('.thumbnail-small')];
                const index = thumbnails.findIndex(item => item.classList.contains('selected'));
                if (!request || !view?.value || index < 0) throw new Error("Open a Video in the Output Gallery First.");
                const gallery = JSON.parse(view.value);
                const payload = {workspace: gallery.workspace, path: gallery.video[index], time, id: Date.now()};
                Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(request, JSON.stringify(payload));
                request.dispatchEvent(new Event('input', {bubbles: true}));
                return;
            }
            button.disabled = true;
            status.textContent = "Saving Frame…";
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d").drawImage(video, 0, 0);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
            if (disposed) return;
            if (!blob) throw new Error("Could Not Encode Frame.");
            const url = URL.createObjectURL(blob);
            const download = document.createElement("a");
            download.href = url;
            download.download = `${stem}_frame_${time.toFixed(3)}s.png`;
            document.body.appendChild(download);
            download.click();
            download.remove();
            setTimeout(() => URL.revokeObjectURL(url), 60000);
            status.textContent = "";
        } catch (error) {
            if (!disposed) status.textContent = error.message;
        } finally {
            canvas.width = canvas.height = 0;
            button.disabled = false;
        }
    });
    // Reference galleries still offer downloads, but cannot append output media.
    menu.querySelector('[data-action="add"]').hidden = !video.closest('#gallery');
    preview.appendChild(menu);
    positionActions();
    resizeObserver.observe(video);
    resizeObserver.observe(preview);
    document.addEventListener('fullscreenchange', positionActions);
}
