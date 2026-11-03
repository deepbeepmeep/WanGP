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
#[cfg(not(windows))]
pub struct ProcessTree;
#[cfg(not(windows))]
impl ProcessTree {
    pub fn attach(_child: &Child) -> Result<Self, String> { Err("This preview currently supports Windows process supervision".into()) }
    pub fn terminate(&self) {}
}
