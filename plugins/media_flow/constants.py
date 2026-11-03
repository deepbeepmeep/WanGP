# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from shared.utils.video_codecs import SUPPORTED_VIDEO_CONTAINERS

RATIO_CHOICES = [("1:1", "1:1"), ("4:3", "4:3"), ("3:4", "3:4"), ("16:9", "16:9"), ("9:16", "9:16"), ("21:9", "21:9"), ("9:21", "9:21")]
RATIO_CHOICES_WITH_EMPTY = [("", "")] + RATIO_CHOICES
DEFAULT_TARGET_RATIO = "4:3"
DEFAULT_SOURCE_PATH = ""
DEFAULT_OUTPUT_PATH = ""
ADD_USER_SETTINGS_MODEL_TYPE = "__add_user_settings__"
ADD_USER_SETTINGS_LABEL = "<Add User Settings>"
NO_USER_SETTINGS_VALUE = "__no_user_settings__"
NO_USER_SETTINGS_LABEL = "<No choice>"
USER_SETTINGS_HINT_HTML = "<div class='mf-hint'>* linked user settings</div>"
MAX_STATUS_REFRESH_HZ = 3.0
STATUS_REFRESH_INTERVAL_SECONDS = 1.0 / MAX_STATUS_REFRESH_HZ
SUPPORTED_OUTPUT_CONTAINERS = SUPPORTED_VIDEO_CONTAINERS
HIGH_OUTPUT_RESOLUTION_CHOICES = [("2160p", "2160p"), ("1440p", "1440p")]
STANDARD_OUTPUT_RESOLUTION_CHOICES = [("1080p", "1080p"), ("900p", "900p"), ("720p", "720p"), ("540p", "540p"), ("480p", "480p"), ("384p", "384p"), ("320p", "320p"), ("256p", "256p")]
SOURCE_AUDIO_TRACK_CHOICES = [("Auto", "")] + [(f"Audio Track {track_no}", str(track_no)) for track_no in range(1, 10)]


def output_resolution_choices(server_config: dict) -> list[tuple[str, str]]:
    return HIGH_OUTPUT_RESOLUTION_CHOICES + STANDARD_OUTPUT_RESOLUTION_CHOICES if server_config.get("enable_4k_resolutions", 0) == 1 else STANDARD_OUTPUT_RESOLUTION_CHOICES
