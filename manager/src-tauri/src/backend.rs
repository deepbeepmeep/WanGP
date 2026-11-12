use crate::process::ProcessTree;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap, fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
    thread, time::Duration,
};
use tauri::{Emitter, Manager, WebviewUrl, WebviewWindow};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_clipboard_manager::ClipboardExt;

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Action { Hardware, CondaEnvironments, DetectInstallation, ManagerUpdate, LatestVersions, Catalog, Sources, Register, Inspect, Preview, Apply, Recover, Versions, RemoveVersion, Launch, Demo }

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LaunchOptions {
    listen: bool,
    verbose: bool,
    authentication: bool,
    password_mode: String,
    debug_deepy: bool,
    debug_directory: String,
    telemetry: bool,
    telemetry_file: String,
}
impl Default for LaunchOptions {
    fn default() -> Self { Self { listen: false, verbose: false, authentication: false, password_mode: "generated".into(), debug_deepy: false, debug_directory: String::new(), telemetry: false, telemetry_file: String::new() } }
}
#[derive(Clone, Serialize, Deserialize)]
struct Preferences {
    schema_version: u32,
    installations: Vec<Value>,
    selected: Option<String>,
    port: u16,
    #[serde(default)]
    launch_options: LaunchOptions,
    #[serde(default)]
    authentication_secret: Option<Vec<u8>>,
    /// The App window's web engine draws without the graphics card, leaving its memory to WanGP.
    #[serde(default)]
    app_without_gpu: bool,
    /// The one-time GitHub star suggestion was answered: it does not come back.
    #[serde(default)]
    star_prompt_answered: bool,
}
impl Default for Preferences {
    fn default() -> Self { Self { schema_version: 1, installations: vec![], selected: None, port: 7860, launch_options: LaunchOptions::default(), authentication_secret: None, app_without_gpu: false, star_prompt_answered: false } }
}
const PROJECT_PAGE: &str = "https://github.com/deepbeepmeep/WanGP";
/// WebView2 fixes its browser arguments for every window of the Manager when the first one opens: wry's defaults,
/// plus software rendering when the App window should leave the graphics card to WanGP.
#[cfg(windows)]
const WITHOUT_GPU_ARGUMENTS: &str = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --disable-gpu";
fn preferences_view(preferences: &Preferences) -> Value {
    let mut view = serde_json::to_value(preferences).unwrap();
    view.as_object_mut().unwrap().remove("authentication_secret");
    view["password_saved"] = json!(preferences.authentication_secret.is_some());
    view
}

#[derive(Clone, Serialize)]
struct JobSnapshot {
    id: String, action: Action, args: Value, done: bool,
    events: Vec<Value>, result: Option<Value>, server_url: Option<String>, stopping: bool,
}
struct Job {
    snapshot: Mutex<JobSnapshot>, input: Mutex<Option<ChildStdin>>,
    tree: Mutex<Option<ProcessTree>>, requested: AtomicBool, cancelled: AtomicBool,
}
struct AppState {
    data: PathBuf, runtime: PathBuf, runtime_error: Mutex<Option<String>>, runtime_ready: AtomicBool,
    preferences: Mutex<Preferences>, jobs: Mutex<BTreeMap<String, Arc<Job>>>, exiting: AtomicBool,
    /// The setting read at start: the web engine keeps it until the Manager restarts.
    app_without_gpu: bool,
}

fn authorize(window: &WebviewWindow) -> Result<(), String> {
    let url = window.url().map_err(|e| e.to_string())?;
    if !manager_origin(window.label(), &url) {
        return Err("This command is only available in the bundled Manager window".into());
    }
    Ok(())
}
fn manager_origin(label: &str, url: &tauri::Url) -> bool {
    label == "main" && url.port().is_none() && url.username().is_empty() && url.password().is_none()
        && ((url.scheme() == "tauri" && url.host_str() == Some("localhost"))
            || (url.scheme() == "http" && url.host_str() == Some("tauri.localhost")))
}
fn verify_runtime(root: &Path) -> Result<(), String> {
    let manifest: Value = serde_json::from_str(include_str!("../runtime-manifest.json")).map_err(|e| e.to_string())?;
    let files = manifest["files"].as_object().ok_or("Invalid bundled runtime inventory")?;
    for (relative, expected) in files {
        if Path::new(relative).components().any(|part| !matches!(part, std::path::Component::Normal(_))) { return Err("Invalid bundled runtime path".into()); }
        let mut file = fs::File::open(root.join(relative)).map_err(|_| format!("A bundled runtime file is missing: {relative}. Restore the complete portable app folder."))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 { break; }
            hasher.update(&buffer[..count]);
        }
        if format!("{:x}", hasher.finalize()) != expected.as_str().unwrap_or("") { return Err(format!("Bundled runtime verification failed: {relative}. Restore the portable app before running operations.")); }
    }
    Ok(())
}
fn save_preferences(state: &AppState, preferences: &Preferences) -> Result<(), String> {
    let path = state.data.join("preferences.json");
    let temporary = state.data.join("preferences.next.json");
    let bytes = serde_json::to_vec_pretty(preferences).map_err(|e| e.to_string())?;
    let mut file = fs::File::create(&temporary).map_err(|e| e.to_string())?;
    file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(|e| e.to_string())?;
    drop(file);
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH};
        let from: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
        let to: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) } == 0 { return Err(std::io::Error::last_os_error().to_string()); }
    }
    #[cfg(not(windows))]
    fs::rename(temporary, path).map_err(|e| e.to_string())?;
    Ok(())
}
fn register_result(app: &tauri::AppHandle, info: &Value) -> Result<(), String> {
    let state = app.state::<AppState>();
    let root = info["root"].as_str().ok_or("Worker did not return an installation root")?;
    let mut preferences = state.preferences.lock().unwrap();
    let mut next = preferences.clone();
    next.installations.retain(|item| item["root"].as_str() != Some(root));
    next.installations.push(info.clone());
    next.selected = Some(root.to_string());
    save_preferences(&state, &next)?;
    *preferences = next;
    app.emit_to("main", "manager:registry", preferences_view(&preferences)).map_err(|e| e.to_string())?;
    Ok(())
}
fn event(app: &tauri::AppHandle, job: &Arc<Job>, value: Value) {
    let mut snapshot = job.snapshot.lock().unwrap();
    if value["type"] == "server_ready" && job.cancelled.load(Ordering::SeqCst) { return; }
    if value["type"] == "server_starting" || value["type"] == "server_stopping" { snapshot.server_url = None; }
    if value["type"] == "server_stopping" { snapshot.stopping = true; }
    if value["type"] == "server_ready" { snapshot.server_url = value["url"].as_str().map(str::to_string); }
    snapshot.events.push(value.clone());
    if snapshot.events.len() > 250 { snapshot.events.remove(0); }
    let envelope = json!({"id": snapshot.id, "action": snapshot.action, "event": value});
    drop(snapshot);
    let _ = app.emit_to("main", "manager:job", envelope);
}
fn finish(app: &tauri::AppHandle, job: &Arc<Job>, result: Value) {
    let action = job.snapshot.lock().unwrap().action.clone();
    let registration = if matches!(action, Action::Register) && result.get("ownership").is_some() { Some(&result) } else { result.get("installation") };
    if let Some(info) = registration {
        if let Err(error) = register_result(app, info) { event(app, job, json!({"type":"notice","title":"Registration Needs Attention","message":error})); }
    }
    {
        let mut snapshot = job.snapshot.lock().unwrap();
        snapshot.done = true;
        snapshot.result = Some(result.clone());
        snapshot.server_url = None;
    }
    if matches!(action, Action::Launch) {
        let label = format!("wangp-{}", job.snapshot.lock().unwrap().id);
        if let Some(window) = app.get_webview_window(&label) {
            let _ = window.close();
        }
    }
    event(app, job, json!({"type":"finished","result":result}));
}
fn send_control(job: &Arc<Job>) {
    job.cancelled.store(true, Ordering::SeqCst);
    if job.requested.load(Ordering::SeqCst) {
        if let Some(input) = job.input.lock().unwrap().as_mut() { let _ = input.write_all(b"{\"cancel\":true}\n").and_then(|_| input.flush()); }
    }
}
#[cfg(windows)]
const RUNTIME_PYTHON: &str = "python/python.exe";
#[cfg(windows)]
const RUNTIME_UV: &str = "uv.exe";
#[cfg(not(windows))]
const RUNTIME_PYTHON: &str = "python/bin/python3";
#[cfg(not(windows))]
const RUNTIME_UV: &str = "uv";

/// Linux counterpart of --disable-gpu, set before GTK starts when the App window should leave the graphics card to
/// WanGP: Mesa's software rasterizer for EGL and GLX (glvnd passes over NVIDIA's libraries) and no DMA-BUF buffers, so
/// no web process of the Manager opens the graphics device. The vendor file is used only where Mesa installed it.
#[cfg(target_os = "linux")]
const SOFTWARE_RENDERING: [(&str, &str); 4] = [("LIBGL_ALWAYS_SOFTWARE", "1"), ("__GLX_VENDOR_LIBRARY_NAME", "mesa"), ("WEBKIT_DISABLE_DMABUF_RENDERER", "1"), ("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")];
/// The values these variables had before the Manager set them: the programs it starts get them back.
#[cfg(target_os = "linux")]
static HOST_GRAPHICS: std::sync::OnceLock<Vec<(&'static str, Option<std::ffi::OsString>)>> = std::sync::OnceLock::new();

#[cfg(target_os = "linux")]
fn software_rendering(identifier: &str) {
    let data = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()).map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")));
    let preferences = data.and_then(|data| fs::read(data.join(identifier).join("preferences.json")).ok()).and_then(|bytes| serde_json::from_slice::<Preferences>(&bytes).ok());
    if !preferences.is_some_and(|preferences| preferences.app_without_gpu) { return; }
    let saved = SOFTWARE_RENDERING.iter().filter(|(_, value)| !value.starts_with('/') || Path::new(value).exists())
        .map(|(key, value)| { let before = std::env::var_os(key); std::env::set_var(key, value); (*key, before) }).collect();
    let _ = HOST_GRAPHICS.set(saved);
}

/// The AppImage launcher points library, GTK and GStreamer variables into the mounted image, and software rendering
/// belongs to the Manager's own windows: the host programs the Manager starts (the worker's tools, WanGP, the browser)
/// inherit neither.
#[cfg(target_os = "linux")]
fn host_environment(command: &mut Command) {
    for (key, before) in HOST_GRAPHICS.get().into_iter().flatten() {
        match before { Some(value) => { command.env(key, value); } None => { command.env_remove(key); } }
    }
    let Some(appdir) = std::env::var_os("APPDIR") else { return };
    let appdir = appdir.to_string_lossy().into_owned();
    for (key, value) in std::env::vars_os() {
        let value = value.to_string_lossy();
        if !value.contains(&appdir) { continue; }
        let kept: Vec<&str> = value.split(':').filter(|part| !part.is_empty() && !part.starts_with(&appdir)).collect();
        if kept.is_empty() { command.env_remove(&key); } else { command.env(&key, kept.join(":")); }
    }
}

/// Opens an address in the user's default browser, with their own sign-ins.
#[cfg(windows)]
fn open_url(url: &str) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = std::ffi::OsStr::new(url).encode_wide().chain(Some(0)).collect();
    let result = unsafe { windows_sys::Win32::UI::Shell::ShellExecuteW(std::ptr::null_mut(), std::ptr::null(), wide.as_ptr(), std::ptr::null(), std::ptr::null(), 1) };
    if result as isize <= 32 { return Err("Could not open the default browser".into()); }
    Ok(())
}
#[cfg(not(any(windows, target_os = "linux")))]
fn open_url(_url: &str) -> Result<(), String> { Err("Browser opening is not available on this preview target".into()) }
#[cfg(target_os = "linux")]
fn open_url(url: &str) -> Result<(), String> {
    let mut command = Command::new("xdg-open");
    // The Manager's own directory may be inside the AppImage mount, which ends with the Manager.
    command.arg(url).current_dir(std::env::var_os("HOME").ok_or("HOME is not set")?).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    host_environment(&mut command);
    let mut child = command.spawn().map_err(|e| format!("Could not open the default browser: {e}"))?;
    thread::spawn(move || { let _ = child.wait(); });
    Ok(())
}

fn run_worker(app: tauri::AppHandle, job: Arc<Job>, action: Action, args: Value) {
    let state = app.state::<AppState>();
    let id = job.snapshot.lock().unwrap().id.clone();
    let mut command = Command::new(state.runtime.join(RUNTIME_PYTHON));
    command.args(["-I", "-B", "-u"]).arg(state.runtime.join("worker/entry.py"))
        .arg("--data").arg(&state.data).arg("--uv").arg(state.runtime.join(RUNTIME_UV))
        .current_dir(state.runtime.join("worker")).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (key, _) in std::env::vars() {
        let upper = key.to_uppercase();
        if upper.starts_with("PYTHON") || upper.starts_with("CONDA") || upper.starts_with("UV_") || upper.starts_with("PIP_") || upper == "VIRTUAL_ENV" { command.env_remove(key); }
    }
    #[cfg(windows)]
    { use std::os::windows::process::CommandExt; command.creation_flags(0x08000000); }
    #[cfg(target_os = "linux")]
    { use std::os::unix::process::CommandExt; command.process_group(0); host_environment(&mut command); }
    let outcome = (|| -> Result<Value, String> {
        let mut child = command.spawn().map_err(|e| format!("Could not start the bundled Python worker: {e}"))?;
        let tree = match ProcessTree::attach(&child) {
            Ok(tree) => tree,
            Err(error) => { let _ = child.kill(); let _ = child.wait(); return Err(error); }
        };
        *job.tree.lock().unwrap() = Some(tree);
        *job.input.lock().unwrap() = child.stdin.take();
        let stderr = child.stderr.take().unwrap();
        let log_job = job.clone();
        let log_app = app.clone();
        let stderr_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            loop {
                let mut line = String::new();
                match (&mut reader).take(65536).read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => event(&log_app, &log_job, json!({"type":"log","text":line})),
                }
            }
        });
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let mut hello = false;
        let mut result = None;
        let mut protocol_error = None;
        loop {
            let mut line = String::new();
            let count = (&mut reader).take(8 * 1024 * 1024).read_line(&mut line).map_err(|e| e.to_string())?;
            if count == 0 { break; }
            if !line.ends_with('\n') { protocol_error = Some("Worker response exceeded the protocol limit".to_string()); break; }
            let message: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(_) => { protocol_error = Some("The worker returned an invalid protocol message".to_string()); break; }
            };
            if !hello {
                if message["type"] != "hello" || message["protocol"] != 1 { protocol_error = Some("Incompatible installation worker protocol".to_string()); break; }
                hello = true;
                let request = json!({"protocol":1, "id":id, "action":action, "args":args});
                {
                    let mut input = job.input.lock().unwrap();
                    let pipe = input.as_mut().ok_or("Worker input is unavailable")?;
                    writeln!(pipe, "{request}").and_then(|_| pipe.flush()).map_err(|e| e.to_string())?;
                    job.requested.store(true, Ordering::SeqCst);
                }
                if job.cancelled.load(Ordering::SeqCst) { send_control(&job); }
                continue;
            }
            if message["protocol"] != 1 || message["request_id"] != id { protocol_error = Some("Worker response identity does not match this operation".to_string()); break; }
            if message["type"] == "finished" { result = Some(message["result"].clone()); }
            else { event(&app, &job, message); }
        }
        if protocol_error.is_some() { if let Some(tree) = job.tree.lock().unwrap().as_ref() { tree.terminate(); } }
        let status = child.wait().map_err(|e| e.to_string())?;
        job.tree.lock().unwrap().take();
        job.input.lock().unwrap().take();
        let _ = stderr_thread.join();
        if let Some(error) = protocol_error { return Err(error); }
        result.ok_or_else(|| if job.cancelled.load(Ordering::SeqCst) { "Operation stopped; inspect the installation for interrupted work".into() } else { format!("The worker stopped without a result ({status}). Check the operation logs.") })
    })();
    job.tree.lock().unwrap().take();
    job.input.lock().unwrap().take();
    match outcome {
        Ok(result) => finish(&app, &job, result),
        Err(message) => finish(&app, &job, json!({"status":if job.cancelled.load(Ordering::SeqCst) {"cancelled"} else {"failed"}, "error":{"code":"native_worker_error","message":message}})),
    }
}

#[tauri::command]
async fn get_state(window: WebviewWindow, state: tauri::State<'_, AppState>) -> Result<Value, String> {
    authorize(&window)?;
    if !state.runtime_ready.load(Ordering::SeqCst) && state.runtime_error.lock().unwrap().is_none() {
        let root = state.runtime.clone();
        let error = tauri::async_runtime::spawn_blocking(move || verify_runtime(&root)).await.map_err(|e| e.to_string())?.err();
        state.runtime_ready.store(error.is_none(), Ordering::SeqCst);
        *state.runtime_error.lock().unwrap() = error;
    }
    let jobs: Vec<JobSnapshot> = state.jobs.lock().unwrap().values().map(|job| job.snapshot.lock().unwrap().clone()).collect();
    let home = window.app_handle().path().home_dir().map_err(|e| e.to_string())?;
    let mut destination = home.join("WanGP");
    let mut suffix = 2;
    while destination.exists() { destination = home.join(format!("WanGP-{suffix}")); suffix += 1; }
    Ok(json!({"version":env!("CARGO_PKG_VERSION"), "platform":std::env::consts::OS, "preferences":preferences_view(&state.preferences.lock().unwrap()), "runtime_error":state.runtime_error.lock().unwrap().clone(), "app_without_gpu_active":state.app_without_gpu, "data_directory":state.data, "default_installation":destination, "jobs":jobs}))
}
#[tauri::command]
fn start_job(window: WebviewWindow, app: tauri::AppHandle, state: tauri::State<AppState>, action: Action, mut args: Value) -> Result<String, String> {
    authorize(&window)?;
    if let Some(error) = &*state.runtime_error.lock().unwrap() { return Err(error.clone()); }
    if !state.runtime_ready.load(Ordering::SeqCst) { return Err("Manager is still checking its runtime".into()); }
    if state.exiting.load(Ordering::SeqCst) { return Err("Manager is shutting down".into()); }
    if matches!(action, Action::Launch) {
        let preferences = state.preferences.lock().unwrap();
        let values = args.as_object_mut().ok_or("Invalid launch arguments")?;
        values.insert("launch_options".into(), serde_json::to_value(&preferences.launch_options).map_err(|e| e.to_string())?);
        values.insert("port".into(), json!(preferences.port));
        if preferences.launch_options.authentication && preferences.launch_options.password_mode == "custom" {
            let secret = preferences.authentication_secret.as_ref().ok_or("Set a custom password in Preferences")?;
            values.insert("auth_password".into(), json!(crate::secrets::reveal(secret)?));
        } else { values.remove("auth_password"); }

    }
    let mut jobs = state.jobs.lock().unwrap();
    if jobs.values().filter(|job| !job.snapshot.lock().unwrap().done).count() >= 8 { return Err("Wait for another operation to finish".into()); }
    // Worker cleanup reads the registered roots from preferences. An in-flight
    // registration or activation publishes its new environment in finish().
    let changes_registry = |action: &Action| matches!(action, Action::Register | Action::Apply | Action::Recover);
    if jobs.values().any(|job| {
        let other = job.snapshot.lock().unwrap();
        !other.done && ((matches!(action, Action::RemoveVersion) && changes_registry(&other.action))
            || (changes_registry(&action) && matches!(other.action, Action::RemoveVersion)))
    }) {
        return Err("Wait for installation changes and saved version removal to finish before starting another change.".into());
    }
    if matches!(action, Action::Launch) && jobs.values().any(|job| { let s = job.snapshot.lock().unwrap(); !s.done && matches!(s.action, Action::Launch) && s.args["root"] == args["root"] }) { return Err("This installation is already starting or running".into()); }
    if jobs.len() >= 40 {
        let completed: Vec<String> = jobs.iter().filter(|(_, job)| job.snapshot.lock().unwrap().done).map(|(id, _)| id.clone()).take(20).collect();
        for id in completed { jobs.remove(&id); }
    }
    let id = uuid::Uuid::new_v4().simple().to_string();
    let mut public_args = args.clone();
    if let Some(values) = public_args.as_object_mut() { values.remove("auth_password"); }
    let job = Arc::new(Job { snapshot: Mutex::new(JobSnapshot { id:id.clone(), action:action.clone(), args:public_args, done:false, events:vec![], result:None, server_url:None, stopping:false }), input:Mutex::new(None), tree:Mutex::new(None), requested:AtomicBool::new(false), cancelled:AtomicBool::new(false) });
    jobs.insert(id.clone(), job.clone());
    thread::spawn(move || run_worker(app, job, action, args));
    Ok(id)
}
#[tauri::command]
fn cancel_job(window: WebviewWindow, app: tauri::AppHandle, state: tauri::State<AppState>, id: String) -> Result<(), String> {
    authorize(&window)?;
    let job = state.jobs.lock().unwrap().get(&id).cloned().ok_or("Unknown operation")?;
    send_control(&job);
    let is_launch = matches!(job.snapshot.lock().unwrap().action, Action::Launch);
    if is_launch { event(&app, &job, json!({"type":"server_stopping"})); }
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(10));
        if !job.snapshot.lock().unwrap().done { if let Some(tree) = job.tree.lock().unwrap().as_ref() { tree.terminate(); } }
    });
    Ok(())
}
#[tauri::command]
async fn pick_path(window: WebviewWindow, app: tauri::AppHandle, kind: String) -> Result<Option<String>, String> {
    authorize(&window)?;
    if kind != "folder" && kind != "python" && kind != "log" { return Err("Unsupported file picker".into()); }
    tauri::async_runtime::spawn_blocking(move || {
        let selection = if kind == "folder" { app.dialog().file().blocking_pick_folder() } else if kind == "log" { app.dialog().file().set_file_name("wangp-telemetry.log").add_filter("Log", &["log", "txt"]).blocking_save_file() } else if cfg!(windows) { app.dialog().file().add_filter("Python executable", &["exe"]).blocking_pick_file() } else { app.dialog().file().blocking_pick_file() };
        selection.map(|path| path.into_path().map(|p| p.to_string_lossy().to_string()).map_err(|e| e.to_string())).transpose()
    }).await.map_err(|e| e.to_string())?
}
#[tauri::command]
fn select_installation(window: WebviewWindow, state: tauri::State<AppState>, root: String) -> Result<(), String> {
    authorize(&window)?;
    let mut preferences = state.preferences.lock().unwrap();
    if !preferences.installations.iter().any(|item| item["root"] == root) { return Err("Unknown installation".into()); }
    let mut next = preferences.clone(); next.selected = Some(root);
    save_preferences(&state, &next)?; *preferences = next; Ok(())
}
#[tauri::command]
fn set_port(window: WebviewWindow, state: tauri::State<AppState>, port: u16) -> Result<(), String> {
    authorize(&window)?;
    if port < 1024 { return Err("Choose a port between 1024 and 65535".into()); }
    let mut preferences = state.preferences.lock().unwrap();
    let mut next = preferences.clone(); next.port = port;
    save_preferences(&state, &next)?; *preferences = next; Ok(())
}
fn updated_launch_preferences(preferences: &Preferences, data: &Path, mut options: LaunchOptions, password: Option<String>, port: Option<u16>) -> Result<Preferences, String> {
    if port.is_some_and(|value| value < 1024) { return Err("Choose a port between 1024 and 65535".into()); }
    if !["generated", "custom"].contains(&options.password_mode.as_str()) { return Err("Choose a generated or custom password".into()); }
    if options.debug_directory.trim().is_empty() { options.debug_directory = data.join("logs/deepy").to_string_lossy().to_string(); }
    if options.telemetry_file.trim().is_empty() { options.telemetry_file = data.join("logs/telemetry.log").to_string_lossy().to_string(); }
    for (enabled, path) in [(options.debug_deepy, &options.debug_directory), (options.telemetry, &options.telemetry_file)] {
        if enabled && (!Path::new(path).is_absolute() || path.contains('\0')) { return Err("Choose an absolute output path".into()); }
    }
    if options.telemetry && ![Some("log"), Some("txt")].contains(&Path::new(&options.telemetry_file).extension().and_then(|s| s.to_str())) { return Err("Choose a .log or .txt telemetry file".into()); }
    let mut next = preferences.clone();
    if let Some(value) = password { next.authentication_secret = Some(crate::secrets::protect(&value)?); }
    if options.authentication && options.password_mode == "custom" && next.authentication_secret.is_none() { return Err("Enter your custom password before saving".into()); }
    next.launch_options = options;
    if let Some(port) = port { next.port = port; }
    Ok(next)
}

#[tauri::command]
fn set_launch_options(window: WebviewWindow, state: tauri::State<AppState>, options: LaunchOptions, password: Option<String>, port: Option<u16>, app_without_gpu: Option<bool>) -> Result<Value, String> {
    authorize(&window)?;
    let mut preferences = state.preferences.lock().unwrap();
    let mut next = updated_launch_preferences(&preferences, &state.data, options, password, port)?;
    if let Some(value) = app_without_gpu { next.app_without_gpu = value; }
    save_preferences(&state, &next)?; *preferences = next;
    Ok(preferences_view(&preferences))
}

/// COM or GObject objects of the main thread, carried through the dialog's thread and used again only on the main thread.
#[cfg(any(windows, target_os = "linux"))]
struct MainThreadOnly<T>(T);
#[cfg(any(windows, target_os = "linux"))]
unsafe impl<T> Send for MainThreadOnly<T> {}
#[cfg(any(windows, target_os = "linux"))]
impl<T> MainThreadOnly<T> { fn take(self) -> T { self.0 } }

// WebView2 saves to the Downloads folder without asking. The deferral holds the download while the dialog is open,
// so that its event ends at once and the main thread keeps pumping messages for the dialog's owner.
#[cfg(windows)]
fn ask_where_to_save(webview: tauri::webview::PlatformWebview, window: WebviewWindow) -> windows_core::Result<()> {
    use webview2_com::{take_pwstr, DownloadStartingEventHandler, Microsoft::Web::WebView2::Win32::ICoreWebView2_4};
    use windows_core::{Interface, HSTRING, PWSTR};
    let handler = DownloadStartingEventHandler::create(Box::new(move |_, args| {
        let Some(args) = args else { return Ok(()) };
        let mut path = PWSTR::null();
        let deferral = unsafe { args.ResultFilePath(&mut path)?; args.GetDeferral()? };
        let suggested = PathBuf::from(take_pwstr(path));
        let extension = suggested.extension().unwrap_or_default().to_string_lossy().into_owned();
        let mut dialog = window.dialog().file().set_parent(&window).set_file_name(suggested.file_name().unwrap_or_default().to_string_lossy());
        if !extension.is_empty() { dialog = dialog.add_filter(extension.to_uppercase(), &[&extension]); }
        let (app, pending) = (window.app_handle().clone(), MainThreadOnly((args, deferral)));
        dialog.save_file(move |choice| {
            let _ = app.run_on_main_thread(move || unsafe {
                let (args, deferral) = pending.take();
                match choice.and_then(|file| file.into_path().ok()) {
                    Some(path) => { let _ = args.SetResultFilePath(&HSTRING::from(path.as_os_str())); }
                    None => { let _ = args.SetCancel(true); }
                }
                let _ = deferral.Complete();
            });
        });
        Ok(())
    }));
    unsafe { webview.controller().CoreWebView2()?.cast::<ICoreWebView2_4>()?.add_DownloadStarting(&handler, &mut 0) }
}

// WebKitGTK downloads ask for a destination with decide-destination. wry answers it on every download of its views,
// saving into the Downloads folder without asking, and it connects after this handler: when the response arrives,
// just before decide-destination, its handler is blocked and this one kept. Returning true holds the download until
// a destination is set, so the dialog stays asynchronous as on Windows. App windows share one web context: each
// handler serves its own view only and leaves with it.
#[cfg(target_os = "linux")]
fn ask_where_to_save(webview: tauri::webview::PlatformWebview, window: WebviewWindow) {
    use webkit2gtk::{glib::{self, prelude::{ObjectExt, ObjectType, StaticType}, subclass::signal::SignalId, translate::IntoGlib}, Download, DownloadExt, WebContextExt, WebViewExt};
    let view = webview.inner();
    let Some(context) = view.context() else { return };
    let owner = view.downgrade();
    let handler = context.connect_download_started(move |_, download| {
        if download.web_view() != owner.upgrade() { return; }
        let window = window.clone();
        let ours = download.connect_decide_destination(move |download, suggested| {
            let suggested = PathBuf::from(suggested);
            let extension = suggested.extension().unwrap_or_default().to_string_lossy().into_owned();
            let mut dialog = window.dialog().file().set_parent(&window).set_file_name(suggested.file_name().unwrap_or_default().to_string_lossy());
            if !extension.is_empty() { dialog = dialog.add_filter(extension.to_uppercase(), &[&extension]); }
            let (app, pending) = (window.app_handle().clone(), MainThreadOnly(download.clone()));
            dialog.save_file(move |choice| {
                let _ = app.run_on_main_thread(move || {
                    let download = pending.take();
                    match choice.and_then(|file| file.into_path().ok()) {
                        Some(path) => { download.set_allow_overwrite(true); download.set_destination(&path.to_string_lossy()); }
                        None => download.cancel(),
                    }
                });
            });
            true
        });
        download.connect_notify_local(Some("response"), move |download, _| {
            let Some(decide) = SignalId::lookup("decide-destination", Download::static_type()) else { return };
            // Matching by signal alone: g_signal_handlers_block_matched needs a closure, function or data to match.
            let unblocked = glib::gobject_ffi::G_SIGNAL_MATCH_ID | glib::gobject_ffi::G_SIGNAL_MATCH_UNBLOCKED;
            while let id @ 1.. = unsafe { glib::gobject_ffi::g_signal_handler_find(download.as_ptr().cast(), unblocked, decide.into_glib(), 0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) } {
                unsafe { glib::gobject_ffi::g_signal_handler_block(download.as_ptr().cast(), id) };
            }
            download.unblock_signal(&ours);
        });
    });
    view.add_weak_ref_notify_local(move || context.disconnect(handler));
}

#[tauri::command]
async fn open_server(window: WebviewWindow, app: tauri::AppHandle, state: tauri::State<'_, AppState>, id: String, presentation: String) -> Result<(), String> {
    authorize(&window)?;
    let job = state.jobs.lock().unwrap().get(&id).cloned().ok_or("Unknown server")?;
    let snapshot = job.snapshot.lock().unwrap();
    if snapshot.done || job.cancelled.load(Ordering::SeqCst) || !matches!(snapshot.action, Action::Launch) { return Err("The server is no longer running".into()); }
    let url = snapshot.server_url.clone().ok_or("Wait for this installation's server to be ready")?;
    let installation_name = snapshot.args["root"].as_str().and_then(|root| Path::new(root).file_name()).and_then(|name| name.to_str()).unwrap_or("WanGP").to_owned();
    drop(snapshot);
    let parsed: tauri::Url = url.parse().map_err(|_| "Invalid server URL")?;
    if parsed.scheme() != "http" || parsed.host_str() != Some("127.0.0.1") || parsed.port().is_none() { return Err("Only the verified loopback server can be opened".into()); }
    match presentation.as_str() {
        "app" => {
            let label = format!("wangp-{id}");
            if let Some(existing) = app.get_webview_window(&label) {
                existing.unminimize().map_err(|e| e.to_string())?;
                return existing.set_focus().map_err(|e| e.to_string());
            }
            let origin = parsed.origin().ascii_serialization();
            let title = format!("WanGP App - {installation_name}");
            let builder = tauri::WebviewWindowBuilder::new(&app, label, WebviewUrl::External(parsed)).title(&title).inner_size(1280.0, 850.0)
                .disable_drag_drop_handler();  // the web view keeps its own drag and drop: pages drag their items and take dropped files, as in a browser
            #[cfg(windows)]
            let builder = if state.app_without_gpu { builder.additional_browser_args(WITHOUT_GPU_ARGUMENTS) } else { builder };
            #[cfg(target_os = "linux")]
            let builder = builder.initialization_script(crate::file_drops::SCRIPT);
            let window = builder.icon(tauri::include_image!("../assets/wangp.png")).map_err(|e| e.to_string())?
                .on_document_title_changed(move |window, _| { let _ = window.set_title(&title); })
                // WebKitGTK also asks before a download link opens, and data: links (Motion Designer's background image)
                // have no origin: they stay allowed there, and WebKit never shows them as a page.
                .on_navigation(move |url| url.origin().ascii_serialization() == origin || cfg!(target_os = "linux") && url.scheme() == "data")
                // Links that open a new window (GitHub, Discord, model pages) go to the user's browser, with their sign-ins.
                .on_new_window(|url, _| {
                    if matches!(url.scheme(), "http" | "https") { let url = url.to_string(); thread::spawn(move || { let _ = open_url(&url); }); }
                    tauri::webview::NewWindowResponse::Deny
                }).build().map_err(|e| e.to_string())?;
            #[cfg(any(windows, target_os = "linux"))]
            window.with_webview({ let window = window.clone(); move |webview| { let _ = ask_where_to_save(webview, window); } }).map_err(|e| e.to_string())?;
            #[cfg(target_os = "linux")]
            window.with_webview({ let window = window.clone(); move |webview| crate::file_drops::bridge(webview, window) }).map_err(|e| e.to_string())?;
            #[cfg(target_os = "linux")]
            if state.app_without_gpu {
                window.with_webview(|webview| {
                    use webkit2gtk::{HardwareAccelerationPolicy, SettingsExt, WebViewExt};
                    if let Some(settings) = WebViewExt::settings(&webview.inner()) { settings.set_hardware_acceleration_policy(HardwareAccelerationPolicy::Never); }
                }).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        "browser" => open_url(&url),
        _ => Err("Unknown presentation".into()),
    }
}

#[tauri::command]
async fn open_manager_download(window: WebviewWindow) -> Result<(), String> {
    authorize(&window)?;
    open_url(&format!("{PROJECT_PAGE}/releases/latest/download/{}", if cfg!(windows) { "WanGPManager.exe" } else { "WanGPManager-x86_64.AppImage" }))
}

/// Opens the project page in the user's browser, where they are signed in to GitHub: Manager never stars for them.
/// Either answer to the one-time suggestion retires it.
#[tauri::command]
fn github_star(window: WebviewWindow, state: tauri::State<AppState>, open: bool) -> Result<Value, String> {
    authorize(&window)?;
    if open { open_url(PROJECT_PAGE)?; }
    let mut preferences = state.preferences.lock().unwrap();
    if !preferences.star_prompt_answered {
        let mut next = preferences.clone(); next.star_prompt_answered = true;
        save_preferences(&state, &next)?; *preferences = next;
    }
    Ok(preferences_view(&preferences))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_preferences_default_to_no_extra_launch_options() {
        let mut preferences: Preferences = serde_json::from_value(json!({"schema_version":1,"installations":[],"selected":null,"port":7866})).unwrap();
        assert!(!preferences.launch_options.listen && !preferences.launch_options.verbose && !preferences.launch_options.authentication);
        preferences.launch_options = LaunchOptions { listen:true, verbose:true, authentication:true, ..LaunchOptions::default() };
        let restored: Preferences = serde_json::from_slice(&serde_json::to_vec(&preferences).unwrap()).unwrap();
        assert!(restored.launch_options.listen && restored.launch_options.verbose && restored.launch_options.authentication);
        assert!(serde_json::from_value::<LaunchOptions>(json!({"listen":"true"})).is_err());
        assert!(serde_json::from_value::<LaunchOptions>(json!({"extra":"--server-port 1"})).is_err());
    }

    #[test]
    fn port_and_launch_preferences_are_validated_together() {
        let current = Preferences::default();
        let options = LaunchOptions { verbose:true, ..LaunchOptions::default() };
        let data = std::env::temp_dir();
        assert!(updated_launch_preferences(&current, &data, options.clone(), None, Some(80)).is_err());
        assert_eq!(current.port, 7860);
        assert!(!current.launch_options.verbose);
        let saved = updated_launch_preferences(&current, &data, options.clone(), None, Some(7877)).unwrap();
        assert_eq!(saved.port, 7877);
        assert!(saved.launch_options.verbose);
        let preserved = updated_launch_preferences(&saved, &data, options, None, None).unwrap();
        assert_eq!(preserved.port, 7877);
    }

    #[cfg(windows)]
    #[test]
    fn authentication_preferences_preserve_encrypted_password_without_exposing_it() {
        let options = LaunchOptions { authentication:true, password_mode:"custom".into(), ..LaunchOptions::default() };
        let data = std::env::temp_dir();
        assert!(updated_launch_preferences(&Preferences::default(), &data, options.clone(), None, None).is_err());
        let saved = updated_launch_preferences(&Preferences::default(), &data, options.clone(), Some("A private test passphrase".into()), None).unwrap();
        assert!(!serde_json::to_string(&saved).unwrap().contains("A private test passphrase"));
        let public = preferences_view(&saved);
        assert_eq!(public["password_saved"], true);
        assert!(public.get("authentication_secret").is_none());
        let next = updated_launch_preferences(&saved, &data, options, None, None).unwrap();
        assert_eq!(crate::secrets::reveal(next.authentication_secret.as_ref().unwrap()).unwrap(), "A private test passphrase");
        let telemetry = LaunchOptions { telemetry:true, telemetry_file:"relative.log".into(), ..LaunchOptions::default() };
        assert!(updated_launch_preferences(&saved, &data, telemetry, None, None).is_err());
    }

    #[test]
    fn only_the_bundled_control_window_has_authority() {
        for url in ["http://tauri.localhost/", "tauri://localhost/index.html"] {
            let url = url.parse().unwrap();
            assert!(manager_origin("main", &url));
            assert!(!manager_origin("wangp-test", &url));
        }
        for url in ["http://127.0.0.1:7866/", "http://localhost:7866/", "https://example.org/", "http://tauri.localhost:7866/", "http://user@tauri.localhost/", "file:///C:/index.html"] {
            assert!(!manager_origin("main", &url.parse().unwrap()));
        }
    }

    #[test]
    fn runtime_inventory_rejects_missing_and_modified_workers() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.bundle");
        assert!(verify_runtime(&root).is_ok());
        assert!(verify_runtime(&root.join("missing")).unwrap_err().contains("missing"));
        let target = root.join("worker/entry.py");
        let original = fs::read(&target).unwrap();
        fs::write(&target, b"invalid worker").unwrap();
        let result = verify_runtime(&root);
        fs::write(&target, original).unwrap();
        assert!(result.unwrap_err().contains("verification failed"));
    }
}
#[tauri::command]
async fn export_diagnostics(window: WebviewWindow, app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<Option<String>, String> {
    authorize(&window)?;
    let report = json!({"manager_version":env!("CARGO_PKG_VERSION"), "runtime_error":state.runtime_error.lock().unwrap().clone(), "installations":state.preferences.lock().unwrap().installations, "jobs":state.jobs.lock().unwrap().values().map(|job| job.snapshot.lock().unwrap().clone()).collect::<Vec<_>>()});
    tauri::async_runtime::spawn_blocking(move || {
        let Some(file) = app.dialog().file().set_file_name("wangp-manager-diagnostics.json").blocking_save_file() else { return Ok(None); };
        let path = file.into_path().map_err(|e| e.to_string())?;
        fs::write(&path, serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        Ok(Some(path.to_string_lossy().to_string()))
    }).await.map_err(|e| e.to_string())?
}
#[tauri::command]
fn set_console_fullscreen(window: WebviewWindow, enabled: bool) -> Result<(), String> {
    authorize(&window)?;
    // Windows can restore the window without notifying Tao. Sync its minimized
    // state before leaving fullscreen so it reapplies the normal title bar.
    #[cfg(windows)]
    if !enabled {
        window.unminimize().map_err(|e| e.to_string())?;
    }
    window.set_fullscreen(enabled).map_err(|e| e.to_string())
}
#[tauri::command]
async fn console_text(window: WebviewWindow, app: tauri::AppHandle, text: String, save: bool) -> Result<Option<String>, String> {
    authorize(&window)?;
    if text.len() > 1024 * 1024 { return Err("Console text exceeds the export limit".into()); }
    if !save {
        app.clipboard().write_text(text).map_err(|e| e.to_string())?;
        return Ok(None);
    }
    tauri::async_runtime::spawn_blocking(move || {
        let Some(file) = app.dialog().file().set_file_name("wangp-console.txt").add_filter("Text", &["txt"]).blocking_save_file() else { return Ok(None); };
        let path = file.into_path().map_err(|e| e.to_string())?;
        fs::write(&path, text.as_bytes()).map_err(|e| e.to_string())?;
        Ok(Some(path.to_string_lossy().to_string()))
    }).await.map_err(|e| e.to_string())?
}
#[tauri::command]
fn stop_and_exit(window: WebviewWindow, app: tauri::AppHandle, state: tauri::State<AppState>) -> Result<(), String> {
    authorize(&window)?;
    state.exiting.store(true, Ordering::SeqCst);
    let jobs: Vec<Arc<Job>> = state.jobs.lock().unwrap().values().cloned().collect();
    for job in &jobs { send_control(job); }
    thread::spawn(move || {
        for _ in 0..100 { if jobs.iter().all(|job| job.snapshot.lock().unwrap().done) { break; } thread::sleep(Duration::from_millis(100)); }
        for job in jobs { job.tree.lock().unwrap().take(); }
        app.exit(0);
    });
    Ok(())
}
pub fn run() {
    let context = tauri::generate_context!();
    #[cfg(target_os = "linux")]
    software_rendering(&context.config().identifier);
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") { let _ = window.unminimize(); let _ = window.set_focus(); }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .setup(|app| {
            let data = app.path().app_data_dir()?; fs::create_dir_all(&data)?;
            let runtime = app.path().resource_dir()?.join("runtime");
            let mut runtime_error = None;
            let preferences = match fs::read(data.join("preferences.json")) {
                Ok(bytes) => match serde_json::from_slice::<Preferences>(&bytes) {
                    Ok(value) if value.schema_version == 1 => value,
                    _ => { runtime_error = Some("Manager preferences are unreadable or use an unsupported schema. Preserve the file before repairing it.".into()); Preferences::default() }
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Preferences::default(),
                Err(error) => { runtime_error = Some(format!("Cannot read Manager preferences: {error}")); Preferences::default() },
            };
            let app_without_gpu = preferences.app_without_gpu;
            app.manage(AppState { data, runtime, runtime_error:Mutex::new(runtime_error), runtime_ready:AtomicBool::new(false), preferences:Mutex::new(preferences), jobs:Mutex::new(BTreeMap::new()), exiting:AtomicBool::new(false), app_without_gpu });
            // Built here rather than from the configuration: its browser arguments depend on the preferences.
            let config = app.config().app.windows.iter().find(|window| window.label == "main").ok_or("The main window is not configured")?.clone();
            let builder = tauri::WebviewWindowBuilder::from_config(app.handle(), &config)?;
            #[cfg(windows)]
            let builder = if app_without_gpu { builder.additional_browser_args(WITHOUT_GPU_ARGUMENTS) } else { builder };
            builder.build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    let state = window.state::<AppState>();
                    if !state.exiting.load(Ordering::SeqCst) && state.jobs.lock().unwrap().values().any(|job| !job.snapshot.lock().unwrap().done) {
                        api.prevent_close(); let _ = window.emit("manager:close-requested", ());
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_state, start_job, cancel_job, pick_path, select_installation, set_port, set_launch_options, open_server, open_manager_download, github_star, set_console_fullscreen, console_text, export_diagnostics, stop_and_exit])
        .run(context).expect("Unable to start WanGP Manager");
}
