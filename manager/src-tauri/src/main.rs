#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod process;
mod secrets;
mod backend;
fn main() { backend::run(); }
