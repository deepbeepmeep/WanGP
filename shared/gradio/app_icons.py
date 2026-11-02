# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Small SVG marks shared by the app bar and the included-app cards."""
ICONS = {
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
    "expand": '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
    "back": '<path d="m10 5-7 7 7 7M3 12h18"/>',
    "link": '<path d="m10 13 4-4m-6 7-2 2a4 4 0 0 1-6-6l4-4a4 4 0 0 1 6 0m4 0 2-2a4 4 0 0 1 6 6l-4 4a4 4 0 0 1-6 0" transform="translate(1 -1)"/>',
    "media_gen": '<rect x="3" y="3" width="18" height="18" rx="5"/><path d="m10 8 6 4-6 4Z"/>',
    "edit": '<path d="m4 16 12-12 4 4L8 20H4Zm10-10 4 4"/>',
    "media_flow": '<path d="m9 5 10 7-10 7Z"/><path d="M4 5v14"/>',
    "showrunner": '<path d="M3 9h18v12H3ZM3 9l-1-5 18-3 1 5ZM7 3l3 5m4-6 3 5"/>',
    "models_manager": '<path d="m12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 16l10 5 10-5"/>',
    "video_mask_creator": '<path d="M3 4h18v8a9 9 0 0 1-18 0Z"/><path d="M7 9h2m6 0h2m-9 6c2 2 6 2 8 0"/>',
    "motion_designer": '<path d="M3 19c6 0 4-14 11-14h6m-5-4 5 4-5 4"/><circle cx="4" cy="19" r="2"/>',
    "configuration": '<path d="m10 2-1 3-3 1-3-1-2 4 2 2v3l-2 2 2 4 3-1 3 1 1 3h4l1-3 3-1 3 1 2-4-2-2v-3l2-2-2-4-3 1-3-1-1-3Z"/><circle cx="12" cy="12" r="3"/>',
    "guides": '<path d="M12 6C9 3 5 3 2 4v15c3-1 7-1 10 2 3-3 7-3 10-2V4c-3-1-7-1-10 2Zm0 0v15"/>',
    "app": '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 9v12M6 6h.01m3 0h.01"/>',
    "apps": '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><path d="M14 17.5h7m-3.5-3.5v7"/>',
    "more": '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
}


def icon(name):
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, ICONS["app"])}</svg>'
