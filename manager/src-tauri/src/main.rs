#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod process;
mod secrets;
mod backend;
#[cfg(target_os = "linux")]
mod file_drops;
fn main() { backend::run(); }
