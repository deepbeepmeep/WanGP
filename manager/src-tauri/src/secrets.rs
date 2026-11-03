//! User-bound Windows protection for saved custom passphrases.

#[cfg(windows)]
fn transform(bytes: &[u8], protect: bool) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::{Foundation::LocalFree, Security::Cryptography::{CryptProtectData, CryptUnprotectData, CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN}};
    let input = CRYPT_INTEGER_BLOB { cbData: bytes.len() as u32, pbData: bytes.as_ptr() as *mut u8 };
    let mut output = CRYPT_INTEGER_BLOB { cbData: 0, pbData: std::ptr::null_mut() };
    let ok = unsafe {
        if protect { CryptProtectData(&input, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut output) }
        else { CryptUnprotectData(&input, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), std::ptr::null(), CRYPTPROTECT_UI_FORBIDDEN, &mut output) }
    };
    if ok == 0 { return Err("Windows could not access the saved password. Set your password again in Preferences.".into()); }
    let result = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe { std::ptr::write_bytes(output.pbData, 0, output.cbData as usize); LocalFree(output.pbData.cast()); }
    Ok(result)
}

#[cfg(not(windows))]
fn transform(_: &[u8], _: bool) -> Result<Vec<u8>, String> { Err("Saved custom passwords are not supported on this preview target".into()) }

pub fn protect(password: &str) -> Result<Vec<u8>, String> {
    if password.trim().is_empty() || password.len() > 1024 || password.contains('\0') { return Err("Enter a non-empty password of at most 1024 UTF-8 bytes".into()); }
    transform(password.as_bytes(), true)
}

pub fn reveal(bytes: &[u8]) -> Result<String, String> {
    String::from_utf8(transform(bytes, false)?).map_err(|_| "Invalid saved password".into())
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn user_bound_round_trip_and_tamper_rejection() {
        let text = "Manager test passphrase with spaces é";
        let mut sealed = super::protect(text).unwrap();
        assert!(!sealed.windows(text.len()).any(|part| part == text.as_bytes()));
        assert_eq!(super::reveal(&sealed).unwrap(), text);
        let last = sealed.len() - 1; sealed[last] ^= 1;
        assert!(super::reveal(&sealed).is_err());
    }
}
