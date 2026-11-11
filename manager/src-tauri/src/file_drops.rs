//! Files dropped on an App window from other applications (Linux).
//!
//! WebKitGTK stopped giving pages the files dropped from a file manager: `dataTransfer.files` stays empty (WebKit bug
//! 323277, the workaround for CVE-2025-13947, where a page could forge a file list and read the files back). For a drop
//! that comes from another application (GTK reports no source widget, so neither the page nor the Manager made it),
//! the Manager offers its files to that window under a random token; the page fetches exactly those files through
//! `wangp-drop://` and SCRIPT hands them to the page's drop handlers.

use serde_json::{json, Value};
use std::{cell::{Cell, RefCell}, collections::HashMap, path::PathBuf, time::UNIX_EPOCH};
use gtk::prelude::{DragContextExtManual, WidgetExt};
use webkit2gtk::{gio, glib::{self, prelude::ObjectExt}, SecurityManagerExt, URISchemeRequest, URISchemeRequestExt, URISchemeResponse, URISchemeResponseExt, WebContextExt, WebView, WebViewExt};

pub const SCRIPT: &str = include_str!("file_drops.js");

struct Offer { owner: glib::WeakRef<WebView>, paths: Vec<PathBuf> }

thread_local! {
    static OFFERS: RefCell<HashMap<String, Offer>> = RefCell::new(HashMap::new());
    static REGISTERED: Cell<bool> = const { Cell::new(false) };
}

pub fn bridge(webview: tauri::webview::PlatformWebview, window: tauri::WebviewWindow) {
    let view = webview.inner();
    let Some(context) = view.context() else { return };
    if !REGISTERED.replace(true) {
        context.register_uri_scheme("wangp-drop", serve);
        if let Some(security) = context.security_manager() { security.register_uri_scheme_as_cors_enabled("wangp-drop"); }
    }
    let owner = view.downgrade();
    view.connect_drag_data_received(move |view, drag, _, _, data, _, _| {
        if drag.drag_get_source_widget().is_some() || data.uris().is_empty() { return; }
        let paths: Vec<PathBuf> = data.uris().iter().filter_map(|uri| glib::filename_from_uri(uri).ok()).map(|(path, _)| path).filter(|path| path.is_file()).collect();
        if paths.is_empty() { let _ = window.eval("window.__wangpFileDrop&&window.__wangpFileDrop.clear()"); return; }
        let token = OFFERS.with(|offers| {
            let mut offers = offers.borrow_mut();
            let mine = |offer: &Offer| offer.owner.upgrade().as_ref() == Some(view);
            if let Some((token, _)) = offers.iter().find(|(_, offer)| mine(offer) && offer.paths == paths) { return token.clone(); }
            offers.retain(|_, offer| !mine(offer) && offer.owner.upgrade().is_some());
            let token = uuid::Uuid::new_v4().simple().to_string();
            offers.insert(token.clone(), Offer { owner: owner.clone(), paths: paths.clone() });
            token
        });
        let files: Vec<Value> = paths.iter().map(|path| {
            let (kind, _) = gio::content_type_guess(Some(path), &[]);
            let modified = path.metadata().and_then(|meta| meta.modified()).ok().and_then(|time| time.duration_since(UNIX_EPOCH).ok()).map_or(0, |time| time.as_millis() as u64);
            json!({"name": path.file_name().unwrap_or_default().to_string_lossy(), "type": gio::content_type_get_mime_type(&kind).unwrap_or_default().as_str(), "modified": modified})
        }).collect();
        let _ = window.eval(&format!("window.__wangpFileDrop&&window.__wangpFileDrop.offer({})", json!({"token": token, "files": files})));
    });
    view.add_weak_ref_notify_local(|| OFFERS.with(|offers| offers.borrow_mut().retain(|_, offer| offer.owner.upgrade().is_some())));
}

/// Serves `wangp-drop://<token>/<index>` to the window the files were dropped on, and nothing else.
fn serve(request: &URISchemeRequest) {
    let uri = request.uri().unwrap_or_default();
    let (token, index) = uri.trim_start_matches("wangp-drop://").split_once('/').unwrap_or_default();
    let path = OFFERS.with(|offers| {
        let offers = offers.borrow();
        let offer = offers.get(token)?;
        (offer.owner.upgrade()? == request.web_view()?).then(|| offer.paths.get(index.parse::<usize>().ok()?).cloned()).flatten()
    });
    let opened = path.and_then(|path| Some((gio::prelude::FileExt::read(&gio::File::for_path(&path), None::<&gio::Cancellable>).ok()?, path.metadata().ok()?.len())));
    let Some((stream, size)) = opened else {
        request.finish_error(&mut glib::Error::new(gio::IOErrorEnum::NotFound, "This file was not dropped on this window"));
        return;
    };
    let response = URISchemeResponse::new(&stream, size as i64);
    let headers = soup::MessageHeaders::new(soup::MessageHeadersType::Response);
    headers.append("Access-Control-Allow-Origin", "*");
    response.set_http_headers(headers);
    request.finish_with_response(&response);
}
