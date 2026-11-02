# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from typing import List, Tuple, Dict, Callable


def preparse_loras_multipliers(loras_multipliers):
    if isinstance(loras_multipliers, list):
        return [multi.strip(" \r\n") if isinstance(multi, str) else multi for multi in loras_multipliers]

    loras_multipliers = loras_multipliers.strip(" \r\n")
    loras_mult_choices_list = loras_multipliers.replace("\r", "").split("\n")
    loras_mult_choices_list = [multi.strip() for multi in loras_mult_choices_list if multi.strip() and not multi.lstrip().startswith("#")]
    loras_multipliers = " ".join(loras_mult_choices_list)
    return loras_multipliers.replace("|"," ").split()

def expand_slist(slists_dict, mult_no, num_inference_steps, model_switch_step, model_switch_step2 ):
    def expand_one(slist, num_inference_steps):
        if not isinstance(slist, list): slist = [slist]
        if num_inference_steps <=0:
            return []
        return [slist[i * len(slist) // num_inference_steps] for i in range(num_inference_steps)]

    phase1 = slists_dict["phase1"][mult_no]
    phase2 = slists_dict["phase2"][mult_no]
    phase3 = slists_dict["phase3"][mult_no]
    shared = slists_dict["shared"][mult_no]
    if shared:
        if isinstance(phase1, float): return phase1
        return expand_one(phase1, num_inference_steps)    
    else:
        if isinstance(phase1, float) and isinstance(phase2, float) and isinstance(phase3, float) and phase1 == phase2 and phase2 == phase3: return phase1 
        return expand_one(phase1, model_switch_step) + expand_one(phase2, model_switch_step2 - model_switch_step) + expand_one(phase3, num_inference_steps - model_switch_step2)

def _new_slists_dict(nb_loras):
    return {
        "phase1": [1.] * nb_loras,
        "phase2": [1.] * nb_loras,
        "phase3": [1.] * nb_loras,
        "shared": [False] * nb_loras,
    }


def parse_loras_multipliers(loras_multipliers, nb_loras, num_inference_steps, merge_slist = None, nb_phases = 2, model_switch_step = None, model_switch_step2 = None, model_switch_phase = 1, lora_multiplier_branches = None):
    if isinstance(loras_multipliers, str):
        pos = _find_bar(loras_multipliers)
        if pos >= 0 and _find_bar(loras_multipliers[pos + 1:]) >= 0: return "", "", "There can be only one '|' character in Loras Multipliers Sequence"

    if model_switch_step is None:
        model_switch_step = num_inference_steps
    if model_switch_step2 is None:
        model_switch_step2 = num_inference_steps
    def is_float(element: any) -> bool:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
    def parse_slist(mult, lora_no, phase_no):
        if "," in mult:
            multlist = mult.split(",")
            slist = []
            for smult in multlist:
                if not is_float(smult):
                    return None, f"Lora sub value no {lora_no + 1} ({smult}) in Multiplier definition '{multlist}' is invalid in Phase {phase_no + 1}"
                slist.append(float(smult))
            return slist, ""
        if not is_float(mult):
            return None, f"Lora Multiplier no {lora_no + 1} ({mult}) is invalid"
        return float(mult), ""

    def set_phase_slist(target_slists, lora_no, phase_no, slist, shared_phases):
        if shared_phases:
            target_slists["phase1"][lora_no] = target_slists["phase2"][lora_no] = target_slists["phase3"][lora_no] = slist
            target_slists["shared"][lora_no] = True
        else:
            target_slists[f"phase{phase_no + 1}"][lora_no] = slist

    branches = [str(branch).strip() for branch in (lora_multiplier_branches or []) if str(branch).strip()]
    base_slists = _new_slists_dict(nb_loras)
    slists_dict = {"model_switch_step": model_switch_step, "model_switch_step2": model_switch_step2, **base_slists}
    for branch in branches:
        slists_dict[branch] = _new_slists_dict(nb_loras)

    if isinstance(loras_multipliers, list) or len(loras_multipliers) > 0:
        list_mult_choices_list = preparse_loras_multipliers(loras_multipliers)[:nb_loras]
        for i, mult in enumerate(list_mult_choices_list):
            if isinstance(mult, str):
                mult = mult.strip()
                phase_mult = mult.split(";")
                shared_phases = len(phase_mult) <=1
                if not shared_phases and len(phase_mult) != nb_phases :
                    if len(phase_mult) > nb_phases:
                        return "", "", f"if the ';' syntax is used for one Lora multiplier, there should be at most {nb_phases} phases for this multiplier"
                    phase_mult = (phase_mult[:1] + phase_mult) if model_switch_phase == 2 else (phase_mult + phase_mult[-1:])
                for phase_no, mult in enumerate(phase_mult):
                    if len(branches) > 0 and ":" in mult:
                        branch_mults = mult.split(":")
                        if len(branch_mults) != len(branches):
                            return "", "", f"Lora Multiplier no {i + 1} ({mult}) should define {len(branches)} branch values separated by ':'"
                        for branch, branch_mult in zip(branches, branch_mults):
                            slist, error = parse_slist(branch_mult, i, phase_no)
                            if len(error) > 0:
                                return "", "", error
                            set_phase_slist(slists_dict[branch], i, phase_no, slist, shared_phases)
                    else:
                        slist, error = parse_slist(mult, i, phase_no)
                        if len(error) > 0:
                            return "", "", error
                        for target_slists in ([slists_dict[branch] for branch in branches] if branches else [slists_dict]):
                            set_phase_slist(target_slists, i, phase_no, slist, shared_phases)
            else:
                for target_slists in ([slists_dict[branch] for branch in branches] if branches else [slists_dict]):
                    set_phase_slist(target_slists, i, 0, float(mult), True)

    if merge_slist is not None:
        if branches:
            for branch in branches:
                branch_slists = slists_dict[branch]
                merge_branch = merge_slist[branch]
                for key in ("phase1", "phase2", "phase3", "shared"):
                    branch_slists[key] = merge_branch[key] + branch_slists[key]
        else:
            for key in ("phase1", "phase2", "phase3", "shared"):
                slists_dict[key] = merge_slist[key] + slists_dict[key]

    if branches:
        first_branch = slists_dict[branches[0]]
        for key in ("phase1", "phase2", "phase3", "shared"):
            slists_dict[key] = first_branch[key]

    phase1 = slists_dict["phase1"]
    loras_list_mult_choices_nums = [ expand_slist(slists_dict, i, num_inference_steps, model_switch_step, model_switch_step2 )  for i in range(len(phase1)) ]
    loras_list_mult_choices_nums = [ slist[0] if isinstance(slist, list) else slist for slist in loras_list_mult_choices_nums ]
    
    return  loras_list_mult_choices_nums, slists_dict, ""

def update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step = None, phase_switch_step2 = None):
    from mmgp import offload
    sz = len(slists_dict["phase1"])
    slists = [ expand_slist(slists_dict, i, num_inference_steps, phase_switch_step, phase_switch_step2 ) for i in range(sz)  ]
    nos = [str(l) for l in range(sz)]
    offload.activate_loras(trans, nos, slists ) 



def get_model_switch_steps(timesteps, guide_phases, model_switch_phase, switch_threshold, switch2_threshold ):
    total_num_steps = len(timesteps)
    model_switch_step = model_switch_step2 = None
    for i, t in enumerate(timesteps):
        if guide_phases >=2 and model_switch_step is None and t <= switch_threshold: model_switch_step = i
        if guide_phases >=3 and model_switch_step2 is None and t <= switch2_threshold: model_switch_step2 = i                    
    if model_switch_step is None: model_switch_step = total_num_steps
    if model_switch_step2 is None: model_switch_step2 = total_num_steps
    phases_description = ""
    if guide_phases > 1:
        phases_description = "Denoising Steps: "        
        phases_description +=  f" Phase 1 = None" if model_switch_step == 0 else f" Phase 1 = 1:{ min(model_switch_step,total_num_steps) }"
        if model_switch_step < total_num_steps:                    
            phases_description += f", Phase 2 = None" if model_switch_step == model_switch_step2 else f", Phase 2 = {model_switch_step +1}:{ min(model_switch_step2,total_num_steps) }"
            if guide_phases > 2 and model_switch_step2 < total_num_steps:  
                phases_description += f", Phase 3 = {model_switch_step2 +1}:{ total_num_steps}"
    return model_switch_step, model_switch_step2, phases_description



from typing import List, Tuple, Dict, Callable

_ALWD = set(":;,.0123456789")

# ---------------- core parsing helpers ----------------

def _find_bar(s: str) -> int:
    com = False
    for i, ch in enumerate(s):
        if ch in ('\n', '\r'):
            com = False
        elif ch == '#':
            com = True
        elif ch == '|' and not com:
            return i
    return -1

def _spans(text: str) -> List[Tuple[int, int]]:
    res, com, in_tok, st = [], False, False, 0
    for i, ch in enumerate(text):
        if ch in ('\n', '\r'):
            if in_tok: res.append((st, i)); in_tok = False
            com = False
        elif ch == '#':
            if in_tok: res.append((st, i)); in_tok = False
            com = True
        elif not com:
            if ch in _ALWD:
                if not in_tok: in_tok, st = True, i
            else:
                if in_tok: res.append((st, i)); in_tok = False
    if in_tok: res.append((st, len(text)))
    return res

def _choose_sep(text: str, spans: List[Tuple[int, int]]) -> str:
    if len(spans) >= 2:
        a, b = spans[-2][1], spans[-1][0]
        return '\n' if ('\n' in text[a:b] or '\r' in text[a:b]) else ' '
    return '\n' if ('\n' in text or '\r' in text) else ' '

def _ends_in_comment_line(text: str) -> bool:
    ln = text.rfind('\n')
    seg = text[ln + 1:] if ln != -1 else text
    return '#' in seg

def _append_tokens(text: str, k: int, sep: str) -> str:
    if k <= 0: return text
    t = text
    if _ends_in_comment_line(t) and (not t.endswith('\n')): t += '\n'
    parts = []
    if t and not t[-1].isspace(): parts.append(sep)
    parts.append('1')
    for _ in range(k - 1):
        parts.append(sep); parts.append('1')
    return t + ''.join(parts)

def _erase_span_and_one_sep(text: str, st: int, en: int) -> str:
    n = len(text)
    r = en
    while r < n and text[r] in (' ', '\t'): r += 1
    if r > en: return text[:st] + text[r:]
    l = st
    while l > 0 and text[l-1] in (' ', '\t'): l -= 1
    if l < st: return text[:l] + text[en:]
    return text[:st] + text[en:]

def _trim_last_tokens(text: str, spans: List[Tuple[int, int]], drop: int) -> str:
    if drop <= 0: return text
    new_text = text
    for st, en in reversed(spans[-drop:]):
        new_text = _erase_span_and_one_sep(new_text, st, en)
    while new_text and new_text[-1] in (' ', '\t'):
        new_text = new_text[:-1]
    return new_text

def _enforce_count(text: str, target: int) -> str:
    sp = _spans(text); cur = len(sp)
    if cur == target: return text
    if cur > target:  return _trim_last_tokens(text, sp, cur - target)
    sep = _choose_sep(text, sp)
    return _append_tokens(text, target - cur, sep)

def _strip_bars_outside_comments(s: str) -> str:
    com, out = False, []
    for ch in s:
        if ch in ('\n', '\r'): com = False; out.append(ch)
        elif ch == '#':        com = True;  out.append(ch)
        elif ch == '|' and not com: continue
        else: out.append(ch)
    return ''.join(out)

def _replace_tokens(text: str, repl: Dict[int, str]) -> str:
    if not repl: return text
    sp = _spans(text)
    for idx in sorted(repl.keys(), reverse=True):
        if 0 <= idx < len(sp):
            st, en = sp[idx]
            text = text[:st] + repl[idx] + text[en:]
    return text

def _drop_tokens_by_indices(text: str, idxs: List[int]) -> str:
    if not idxs: return text
    out = text
    for idx in sorted(set(idxs), reverse=True):
        sp = _spans(out)  # recompute spans after each deletion
        if 0 <= idx < len(sp):
            st, en = sp[idx]
            out = _erase_span_and_one_sep(out, st, en)
    return out

# ---------------- identity for dedupe ----------------

def _default_path_key(p: str) -> str:
    s = p.strip().replace('\\', '/')
    while '//' in s: s = s.replace('//', '/')
    if len(s) > 1 and s.endswith('/'): s = s[:-1]
    return s

# ---------------- new-set splitter (FIX) ----------------

def _select_new_side(
    loras_new: List[str],
    mult_new: str,
    mode: str,  # "merge before" | "merge after"
) -> Tuple[List[str], str]:
    """
    Split mult_new on '|' (outside comments) and split loras_new accordingly.
    Return ONLY the side relevant to `mode`. Extras loras (if any) are appended to the selected side.
    """
    bi = _find_bar(mult_new)
    if bi == -1:
        return loras_new, _strip_bars_outside_comments(mult_new)

    left, right = mult_new[:bi], mult_new[bi + 1:]
    nL, nR = len(_spans(left)), len(_spans(right))
    L = len(loras_new)

    # Primary allocation by token counts
    b_count = min(nL, L)
    rem     = max(0, L - b_count)
    a_count = min(nR, rem)
    extras  = max(0, L - (b_count + a_count))

    if mode == "merge before":
        # take BEFORE loras + extras
        l_sel = loras_new[:b_count] + (loras_new[b_count + a_count : b_count + a_count + extras] if extras else [])
        m_sel = left
    else:
        # take AFTER loras + extras
        start_after = b_count
        l_sel = loras_new[start_after:start_after + a_count] + (loras_new[start_after + a_count : start_after + a_count + extras] if extras else [])
        m_sel = right

    return l_sel, _strip_bars_outside_comments(m_sel)

# ---------------- public API ----------------

def merge_loras_settings(
    loras_old: List[str],
    mult_old: str,
    loras_new: List[str],
    mult_new: str,
    mode: str = "merge before",
    path_key: Callable[[str], str] = _default_path_key,
) -> Tuple[List[str], str]:
    """
    Merge settings with full formatting/comment preservation and correct handling of `mult_new` with '|'.
    Accelerator entries belong BEFORE: merging BEFORE moves matching entries out of AFTER;
    merging AFTER updates matching BEFORE multipliers without duplicating or moving their entries.
    """
    assert mode in ("merge before", "merge after")
    mult_old= mult_old.strip()
    mult_new= mult_new.strip()

    # Old split & alignment
    bi_old = _find_bar(mult_old)
    before_old, after_old = (mult_old[:bi_old], mult_old[bi_old + 1:]) if bi_old != -1 else ("", mult_old)
    orig_had_bar = (bi_old != -1)

    sp_b_old, sp_a_old = _spans(before_old), _spans(after_old)
    n_b_old = len(sp_b_old)
    total_old = len(loras_old)

    if n_b_old <= total_old:
        keep_b = n_b_old
        keep_a = total_old - keep_b
        before_old_aligned = before_old
        after_old_aligned  = _enforce_count(after_old, keep_a)
    else:
        keep_b = total_old
        keep_a = 0
        before_old_aligned = _enforce_count(before_old, keep_b)
        after_old_aligned  = _enforce_count(after_old, 0)

    # NEW: choose the relevant side of the *new* set (fix for '|' in mult_new)
    loras_new_sel, mult_new_sel = _select_new_side(loras_new, mult_new, mode)
    mult_new_aligned = _enforce_count(mult_new_sel, len(loras_new_sel))
    sp_new = _spans(mult_new_aligned)
    new_tokens = [mult_new_aligned[st:en] for st, en in sp_new]

    if mode == "merge after":
        # Preserve BEFORE; replace AFTER (with dedupe/update)
        preserved_loras = loras_old[:keep_b]
        preserved_text  = before_old_aligned
        preserved_spans = _spans(preserved_text)
        pos_by_key: Dict[str, int] = {}
        for i, lp in enumerate(preserved_loras):
            k = path_key(lp)
            if k not in pos_by_key: pos_by_key[k] = i

        repl_map: Dict[int, str] = {}
        drop_idxs: List[int] = []
        for i, lp in enumerate(loras_new_sel):
            j = pos_by_key.get(path_key(lp))
            if j is not None and j < len(preserved_spans):
                repl_map[j] = new_tokens[i] if i < len(new_tokens) else "1"
                drop_idxs.append(i)

        before_text = _replace_tokens(preserved_text, repl_map)
        after_text  = _drop_tokens_by_indices(mult_new_aligned, drop_idxs)
        loras_keep  = [lp for i, lp in enumerate(loras_new_sel) if i not in set(drop_idxs)]
        loras_out   = preserved_loras + loras_keep

    else:
        # Replace BEFORE, moving matching AFTER entries into the accelerator set.
        preserved_loras = loras_old[keep_b:]
        new_keys = {path_key(lp) for lp in loras_new_sel}
        drop_idxs = [i for i, lp in enumerate(preserved_loras) if path_key(lp) in new_keys]
        after_text = _drop_tokens_by_indices(after_old_aligned, drop_idxs)
        before_text = mult_new_aligned
        loras_out = loras_new_sel + [lp for lp in preserved_loras if path_key(lp) not in new_keys]

    # Compose, preserving explicit "before-only" bar when appropriate
    has_before = len(_spans(before_text)) > 0
    has_after  = len(_spans(after_text)) > 0
    if has_before and has_after:
        mult_out = f"{before_text}|{after_text}"
    elif has_before:
        mult_out = before_text + ('|' if (mode == 'merge before' or orig_had_bar) else '')
    else:
        mult_out = after_text

    return loras_out, mult_out

# ---------------- extractor ----------------

def extract_loras_side(
    loras: List[str],
    mult: str,
    which: str = "before",
) -> Tuple[List[str], str]:
    assert which in ("before", "after")
    bi = _find_bar(mult)
    before_txt, after_txt = (mult[:bi], mult[bi + 1:]) if bi != -1 else ("", mult)

    sp_b = _spans(before_txt)
    n_b  = len(sp_b)
    total = len(loras)

    if n_b <= total:
        keep_b = n_b
        keep_a = total - keep_b
    else:
        keep_b = total
        keep_a = 0

    if which == "before":
        return loras[:keep_b], _enforce_count(before_txt, keep_b)
    else:
        return loras[keep_b:keep_b + keep_a], _enforce_count(after_txt, keep_a)
