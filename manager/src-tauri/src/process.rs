use std::process::Child;

#[cfg(windows)]
pub struct ProcessTree(windows_sys::Win32::Foundation::HANDLE);
#[cfg(windows)]
unsafe impl Send for ProcessTree {}
#[cfg(windows)]
unsafe impl Sync for ProcessTree {}

#[cfg(windows)]
impl ProcessTree {
    pub fn attach(child: &Child) -> Result<Self, String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() { return Err(std::io::Error::last_os_error().to_string()); }
            let tree = Self(handle);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(handle, JobObjectExtendedLimitInformation, &limits as *const _ as _, std::mem::size_of_val(&limits) as u32) == 0 ||
               AssignProcessToJobObject(handle, child.as_raw_handle() as _) == 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            Ok(tree)
        }
    }
    pub fn terminate(&self) {
        unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0, 1); }
    }
}
#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) { unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0); } }
}

/// The worker leads its own process group (`process_group(0)` at spawn) and starts each command in a session of its own.
/// Terminating freezes the worker, lists its descendants from /proc and kills every process group they belong to, so
/// processes a command left behind in its group go too. Dropping terminates, as closing a Windows job does.
#[cfg(target_os = "linux")]
pub struct ProcessTree { pid: i32, start: u64 }

#[cfg(target_os = "linux")]
fn stat(pid: i32) -> Option<(i32, i32, u64)> {
    // Fields after the command name, which may hold spaces and parentheses: state ppid pgrp ... starttime (22nd field).
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<&str> = text[text.rfind(')')? + 1..].split_whitespace().collect();
    Some((fields.get(1)?.parse().ok()?, fields.get(2)?.parse().ok()?, fields.get(19)?.parse().ok()?))
}

#[cfg(target_os = "linux")]
impl ProcessTree {
    pub fn attach(child: &Child) -> Result<Self, String> {
        let pid = child.id() as i32;
        // An unwaited child keeps its PID; the start time also tells it apart from a later process given the same PID.
        let (_, _, start) = stat(pid).ok_or("Could not read the worker process state")?;
        Ok(Self { pid, start })
    }
    pub fn terminate(&self) {
        if stat(self.pid).map(|(_, _, start)| start) != Some(self.start) { return; }
        unsafe { libc::kill(self.pid, libc::SIGSTOP); }
        let processes: Vec<(i32, i32, i32)> = std::fs::read_dir("/proc").into_iter().flatten().flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse::<i32>().ok())
            .filter_map(|pid| stat(pid).map(|(parent, group, _)| (pid, parent, group))).collect();
        let mut tree = vec![self.pid];
        let mut index = 0;
        while index < tree.len() {
            let parent = tree[index];
            tree.extend(processes.iter().filter(|process| process.1 == parent).map(|process| process.0));
            index += 1;
        }
        let own = unsafe { libc::getpgrp() };
        let mut groups: Vec<i32> = processes.iter().filter(|process| tree.contains(&process.0)).map(|process| process.2).filter(|group| *group > 1 && *group != own).collect();
        groups.sort_unstable();
        groups.dedup();
        for group in groups { unsafe { libc::killpg(group, libc::SIGKILL); } }
        for pid in tree { unsafe { libc::kill(pid, libc::SIGKILL); } }
    }
}
#[cfg(target_os = "linux")]
impl Drop for ProcessTree {
    fn drop(&mut self) { self.terminate(); }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub struct ProcessTree;
#[cfg(not(any(windows, target_os = "linux")))]
impl ProcessTree {
    pub fn attach(_child: &Child) -> Result<Self, String> { Err("This preview currently supports Windows and Linux process supervision".into()) }
    pub fn terminate(&self) {}
}
