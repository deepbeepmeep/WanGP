# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json


SERVICE_NAME = "WanGP Notifications"
STORAGE_HELP = ('Install the Python keyring package and enable an OS credential service '
                '(on Linux/RunPod: GNOME Keyring with a running D-Bus session), '
                'or uncheck "Store Destinations in OS Credential Manager" to save destinations in plaintext in wgp_config.json.')


class SecureStorageError(RuntimeError):
    pass


def _keyring():
    try:
        import keyring
        backend = keyring.get_keyring()
        if backend.priority <= 0:
            raise RuntimeError("No usable credential-store backend")
        return keyring
    except Exception as error:
        raise SecureStorageError(f"OS credential storage is unavailable ({type(error).__name__}): no usable credential service was found. {STORAGE_HELP}") from error


def _check_availability() -> str:
    try:
        _keyring()
    except SecureStorageError as error:
        return str(error)
    return ""


def _start_availability_check():
    # keyring and its OS backend take a few tenths of a second to load: checked while WanGP goes on starting, for the settings UI
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="keyring_check")
    _startup_check.append(pool.submit(_check_availability))
    pool.shutdown(wait=False)


_startup_check = []


def availability_error() -> str:
    return _startup_check.pop().result() if _startup_check else _check_availability()


def load_urls(credential_id: str) -> list[str]:
    try:
        value = _keyring().get_password(SERVICE_NAME, credential_id)
        if value is None:
            return []
        urls = json.loads(value)
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError("Invalid stored notification destinations")
        return urls
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not read notification destinations from OS credential storage ({type(error).__name__}).") from error


def save_urls(credential_id: str, urls: list[str]) -> None:
    try:
        _keyring().set_password(SERVICE_NAME, credential_id, json.dumps(urls))
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not save notification destinations in OS credential storage ({type(error).__name__}). {STORAGE_HELP}") from error


def delete_urls(credential_id: str) -> None:
    try:
        keyring = _keyring()
        if keyring.get_password(SERVICE_NAME, credential_id) is not None:
            keyring.delete_password(SERVICE_NAME, credential_id)
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not remove notification destinations from OS credential storage ({type(error).__name__}).") from error


_start_availability_check()
