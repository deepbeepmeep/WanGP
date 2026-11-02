"""Initial config suggestions; callers decide whether to apply them."""


def recommend_settings(current, *, ram_gb=None, vram_gb=None, attention=None):
    proposed = {"audio_profile": 3, "compile": "disabled"}
    reasons = {"audio_profile": "Profile 3 is the audio default and uses dynamic preload.", "compile": "Compilation remains disabled unless explicitly enabled."}
    if ram_gb is not None and vram_gb is not None and ram_gb > 0 and vram_gb > 0:
        profile = 1 if ram_gb > 60 and vram_gb > 22 else 2 if ram_gb > 60 else 3 if ram_gb > 30 and vram_gb > 22 else 4 if ram_gb > 30 and vram_gb > 11 else 5
        for key in ("video_profile", "image_profile"):
            proposed[key] = profile
            reasons[key] = "Initial estimate from host RAM and selected-device memory; profiles 1 and 3 use dynamic preload, not guaranteed whole-model residency."
    if attention is not None:
        proposed["attention_mode"] = attention
        reasons["attention_mode"] = "Attention backend explicitly selected from the supported installation components."
    patch = {key: value for key, value in proposed.items() if key not in current}
    return {"patch": patch, "reasons": {key: reasons[key] for key in patch}, "preserved_keys": sorted(set(proposed).intersection(current))}
