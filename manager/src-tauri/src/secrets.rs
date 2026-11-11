//! Saved custom passphrases: user-bound Windows protection, or the Linux desktop keyring.

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
#[cfg(windows)]
fn seal(password: &str) -> Result<Vec<u8>, String> { transform(password.as_bytes(), true) }
#[cfg(windows)]
fn unseal(bytes: &[u8]) -> Result<String, String> { String::from_utf8(transform(bytes, false)?).map_err(|_| "Invalid saved password".into()) }

/// The passphrase lives in the Secret Service (GNOME Keyring, KWallet); preferences only record that it is saved there.
#[cfg(target_os = "linux")]
mod keyring {
    use std::collections::HashMap;
    use zbus::{blocking::Connection, zvariant::{ObjectPath, OwnedObjectPath, OwnedValue, Value}};

    pub const MARKER: &[u8] = b"secret-service:v1";
    const SERVICE: &str = "org.freedesktop.secrets";
    const UNAVAILABLE: &str = "No desktop keyring is available to save the password. Start your keyring (GNOME Keyring or KWallet) or use a generated password.";
    const LOCKED: &str = "Your desktop keyring is locked. Unlock it, then try again.";

    fn attributes() -> HashMap<&'static str, &'static str> { HashMap::from([("application", "com.wangp.manager"), ("purpose", "launch-password")]) }

    fn session() -> Result<(Connection, OwnedObjectPath), String> {
        // A keyring that D-Bus starts on demand and that never answers would otherwise hold the window for 25 s.
        let connection = zbus::blocking::connection::Builder::session().map(|builder| builder.method_timeout(std::time::Duration::from_secs(5))).and_then(|builder| builder.build()).map_err(|_| UNAVAILABLE.to_string())?;
        let reply = connection.call_method(Some(SERVICE), "/org/freedesktop/secrets", Some("org.freedesktop.Secret.Service"), "OpenSession", &("plain", Value::from(""))).map_err(|_| UNAVAILABLE.to_string())?;
        let (_, path): (OwnedValue, OwnedObjectPath) = reply.body().deserialize().map_err(|e| e.to_string())?;
        Ok((connection, path))
    }

    pub fn store(password: &str) -> Result<(), String> {
        let (connection, session) = session()?;
        let properties: HashMap<&str, Value> = HashMap::from([("org.freedesktop.Secret.Item.Label", Value::from("WanGP Manager launch password")), ("org.freedesktop.Secret.Item.Attributes", Value::from(attributes()))]);
        let secret = (ObjectPath::from(&session), Vec::<u8>::new(), password.as_bytes().to_vec(), "text/plain; charset=utf8");
        let reply = connection.call_method(Some(SERVICE), "/org/freedesktop/secrets/aliases/default", Some("org.freedesktop.Secret.Collection"), "CreateItem", &(properties, secret, true)).map_err(|e| format!("Your desktop keyring could not save the password: create or unlock its default keyring, or use a generated password. ({e})"))?;
        let (_, prompt): (OwnedObjectPath, OwnedObjectPath) = reply.body().deserialize().map_err(|e| e.to_string())?;
        if prompt.as_str() != "/" { return Err(LOCKED.into()); }
        Ok(())
    }

    pub fn load() -> Result<String, String> {
        let (connection, session) = session()?;
        let reply = connection.call_method(Some(SERVICE), "/org/freedesktop/secrets", Some("org.freedesktop.Secret.Service"), "SearchItems", &(attributes(),)).map_err(|e| e.to_string())?;
        let (unlocked, locked): (Vec<OwnedObjectPath>, Vec<OwnedObjectPath>) = reply.body().deserialize().map_err(|e| e.to_string())?;
        let Some(item) = unlocked.first() else { return Err(if locked.is_empty() { "The saved password is missing from your desktop keyring. Set your password again in Preferences." } else { LOCKED }.into()) };
        let reply = connection.call_method(Some(SERVICE), item.as_str(), Some("org.freedesktop.Secret.Item"), "GetSecret", &(ObjectPath::from(&session),)).map_err(|e| e.to_string())?;
        let ((_, _, value, _),): ((OwnedObjectPath, Vec<u8>, Vec<u8>, String),) = reply.body().deserialize().map_err(|e| e.to_string())?;
        String::from_utf8(value).map_err(|_| "Invalid saved password".into())
    }
}
#[cfg(target_os = "linux")]
fn seal(password: &str) -> Result<Vec<u8>, String> { keyring::store(password).map(|_| keyring::MARKER.to_vec()) }
#[cfg(target_os = "linux")]
fn unseal(bytes: &[u8]) -> Result<String, String> {
    if bytes != keyring::MARKER { return Err("This saved password was made on another system. Set your password again in Preferences.".into()); }
    keyring::load()
}

#[cfg(not(any(windows, target_os = "linux")))]
fn seal(_: &str) -> Result<Vec<u8>, String> { Err("Saved custom passwords are not supported on this preview target".into()) }
#[cfg(not(any(windows, target_os = "linux")))]
fn unseal(_: &[u8]) -> Result<String, String> { Err("Saved custom passwords are not supported on this preview target".into()) }

pub fn protect(password: &str) -> Result<Vec<u8>, String> {
    if password.trim().is_empty() || password.len() > 1024 || password.contains('\0') { return Err("Enter a non-empty password of at most 1024 UTF-8 bytes".into()); }
    seal(password)
}

pub fn reveal(bytes: &[u8]) -> Result<String, String> { unseal(bytes) }

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
