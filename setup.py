import os
import sys
import json
import subprocess
import argparse
import shutil
import platform

from installation.catalog import kernel_variant, load_config as load_install_config, os_key, resolve_command
from installation.environments import inspect_interpreter, interpreter_path, managed_record
from installation.hardware import profile_for_name
from installation.recommendations import recommend_settings

CONFIG_PATH = "setup_config.json"
ENVS_FILE = "envs.json"
IS_WIN = os.name == 'nt'

_PY_PATH = f'"{os.path.join("{dir}", "Scripts", "python.exe")}"' if IS_WIN else f'"{os.path.join("{dir}", "bin", "python")}"'

ENV_TEMPLATES = {
    "uv": {
        "create": "uv venv --seed --python {ver} \"{dir}\"",
        "run": _PY_PATH,
        "install": f"uv pip install --index-strategy unsafe-best-match --python {_PY_PATH}"
    },
    "venv": {
        "create": "\"{sys_py}\" -m venv \"{dir}\"",
        "run": _PY_PATH,
        "install": f"{_PY_PATH} -m pip install"
    },
    "conda": {
        "create": "conda create -y -p \"{dir}\" python={ver}",
        "run": os.path.join("{dir}", "python.exe"),
        "install": "conda run -p \"{dir}\" pip install"
    },
    "none": {
        "create": "",
        "run": "python" if IS_WIN else "python3",
        "install": "pip install"
    }
}

class EnvsManager:
    def __init__(self):
        self.data = {"active": None, "envs": {}}
        self.load()

    def load(self):
        if os.path.exists(ENVS_FILE):
            try:
                with open(ENVS_FILE, 'r') as f:
                    self.data = json.load(f)
            except:
                print(f"[!] Warning: {ENVS_FILE} corrupted. Starting fresh.")

    def save(self):
        with open(ENVS_FILE, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_active(self):
        return self.data.get("active")

    def set_active(self, name):
        if name in self.data["envs"]:
            self.data["active"] = name
            self.save()
            print(f"[*] '{name}' is now the active environment.")
        else:
            print(f"[!] Environment '{name}' not found.")

    def add_env(self, name, type, path):
        if path:
            cwd = os.getcwd()
            abs_path = os.path.abspath(path)
            try:
                rel_path = os.path.relpath(abs_path, cwd)
                if rel_path.startswith("..") or rel_path == ".":
                    final_path = abs_path
                else:
                    final_path = os.path.join(".", rel_path)
            except ValueError:
                final_path = abs_path
        else:
            final_path = ""

        self.data["envs"][name] = {"type": type, "path": final_path}

        if not self.data["active"]:
            self.data["active"] = name
        self.save()

    def remove_env(self, name):
        if name in self.data["envs"]:
            entry = self.data["envs"][name]
            path = entry["path"]

            if os.path.exists(path) and entry["type"] != "none":
                try:
                    print(f"[*] Deleting directory: {path}")
                    if entry["type"] == "conda":
                         run_cmd(f"conda env remove -p \"{path}\" -y")
                    else:
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"[!] Error removing directory: {e}")

            del self.data["envs"][name]
            if self.data["active"] == name:
                self.data["active"] = None
                keys = list(self.data["envs"].keys())
                if keys:
                    self.data["active"] = keys[0]
                    print(f"[*] Active environment switched to '{keys[0]}'.")
                else:
                    print("[*] No environments left.")
            self.save()

    def list_envs(self):
        return self.data["envs"]

    def resolve_target_env(self):
        """Intelligently determine which env to use for operations."""
        envs = self.list_envs()
        if not envs:
            print("[!] No environments found. Please run install first.")
            sys.exit(1)

        active = self.get_active()

        if len(envs) == 1:
            return list(envs.keys())[0]

        print("\nMultiple environments detected:")
        keys = list(envs.keys())
        for i, k in enumerate(keys):
            marker = "*" if k == active else " "
            print(f"{i+1}. [{marker}] {k} ({envs[k]['type']})")

        print(f"Default: {active}")
        choice = input("Select environment (Number) or Press Enter for Default: ").strip()

        if choice == "":
            return active
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except:
            pass
        return active

def load_config():
    try:
        return load_install_config(CONFIG_PATH)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

def get_gpu_info():
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL
            )
            for line in out.split("\n"):
                if "Chip" in line:
                    name = line.split(":", 1)[1].strip()
                    return name, "APPLE"
        except:
            pass
        return "Apple Silicon (MPS)", "APPLE"

    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding='utf-8',
            stderr=subprocess.DEVNULL
        ).strip()
        return name, "NVIDIA"
    except: pass

    if IS_WIN:
        amd_gpus = get_amd_gpus()
        if amd_gpus: return amd_gpus[0][0], "AMD"
        # wmic is not installed on recent Windows 11 builds
        try:
            names = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            ).split('\n')
            names = [n.strip() for n in names if n.strip()]
            for name in names:
                if "Radeon" in name or "AMD" in name: return name, "AMD"
            if names: return names[0], "INTEL"
        except: pass
    else:
        try:
            name = subprocess.check_output(
                "lspci | grep -i vga",
                shell=True,
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            )
            if "NVIDIA" in name: return name, "NVIDIA"
            if "AMD" in name or "Advanced Micro Devices" in name: return name, "AMD"
        except: pass

    return "Unknown", "UNKNOWN"

def get_amd_gpus():
    """Return (board name, gfx target) for each AMD GPU reported by the driver's clinfo."""
    try:
        out = subprocess.check_output(["clinfo"], encoding='utf-8', errors='replace', stderr=subprocess.DEVNULL)
    except: return []
    gpus = []
    is_gpu, board = False, ""
    for line in out.split('\n'):
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "Device Type":
            is_gpu, board = "CL_DEVICE_TYPE_GPU" in value, ""
        elif is_gpu and key == "Board name":
            board = value
        elif is_gpu and key == "Name" and value.startswith("gfx"):
            gfx = value.split(":")[0]
            gpus.append((board or gfx, gfx))
            is_gpu = False
    return gpus

def get_amd_device_extras():
    targets = []
    for _, gfx in get_amd_gpus():
        if gfx not in targets: targets.append(gfx)
    if targets:
        print(f"[*] Detected AMD GPU target(s): {', '.join(targets)}")
    else:
        targets = [t.strip() for t in input("Could not detect the AMD GPU target. Enter it (e.g. gfx1201, see docs/AMD-INSTALLATION.md): ").split(",") if t.strip()]
    return ",".join(f"device-{t}" for t in targets)

def resolve_torch_cmd(config, torch_k):
    cmd = resolve_cmd(config['components']['torch'][torch_k]['cmd'])
    if "{device}" in cmd:
        cmd = cmd.replace("{device}", get_amd_device_extras())
    return cmd

def get_profile_key(gpu_name, vendor):
    return profile_for_name(gpu_name, vendor)

def get_os_key():
    return os_key()

def resolve_cmd(cmd_entry):
    return resolve_command(cmd_entry, get_os_key())

def run_cmd(cmd, env_vars=None):
    if not cmd: return

    if "&&" in cmd and not IS_WIN:
        print(f"\n>>> Running (Shell): {cmd}")
        custom_env = os.environ.copy()
        if env_vars: custom_env.update(env_vars)
        subprocess.run(cmd, shell=True, check=True, env=custom_env)
        return

    print(f"\n>>> Running: {cmd}")
    custom_env = os.environ.copy()
    if env_vars:
        for k, v in env_vars.items():
            print(f"    [ENV SET] {k}={v}")
            custom_env[k] = v

    subprocess.run(cmd, shell=True, check=True, env=custom_env)

def run_pip_component(pip, cmd, env_path=None):
    if not cmd: return
    env_vars = None
    if not IS_WIN and env_path:
        # Build-time generators may invoke bare python instead of sys.executable.
        env_vars = {"PATH": os.path.join(os.path.abspath(env_path), "bin") + os.pathsep + os.environ.get("PATH", "")}
    run_cmd(cmd.format(pip=pip) if "{pip}" in cmd else f"{pip} {cmd}", env_vars=env_vars)

def install_plugin_requirements(pip_cmd):
    plugins_dir = "plugins"
    if os.path.exists(plugins_dir) and os.path.isdir(plugins_dir):
        for entry in os.listdir(plugins_dir):
            plugin_req = os.path.join(plugins_dir, entry, "requirements.txt")
            if os.path.isfile(plugin_req):
                print(f"\n[*] Installing requirements for plugin '{entry}'...")
                run_cmd(f"{pip_cmd} -r \"{plugin_req}\"")

def get_env_details(name, env_data):
    env_type = env_data["type"]
    dir_name = env_data["path"]
    entry = ENV_TEMPLATES[env_type]

    py_exec = entry['run'] if env_type == "none" else interpreter_path(dir_name, env_type)
    if env_type == "none":
        py_exec = shutil.which(py_exec)
        if py_exec is None:
            return {'error': 'System interpreter not found', 'type': env_type, 'path': dir_name}
    inspection = inspect_interpreter(py_exec)
    if inspection["status"] != "inspected":
        return {'error': inspection["error"], 'type': env_type, 'path': dir_name}
    packages = inspection["packages"]
    data = {key: packages.get(distribution) or "Missing" for key, distribution in (("torch", "torch"), ("sageattention", "sageattention"), ("spas_sage_attn", "spas-sage-attn"), ("flash_attn", "flash-attn"))}
    data["triton"] = packages["triton"] or packages["triton-windows"] or "Missing"
    return {**data, "python": inspection["python"], "path": dir_name, "type": env_type}

def show_status():
    manager = EnvsManager()
    print("\n" + "="*90)
    print(f"{'INSTALLED ENVIRONMENTS & VERSIONS':^90}")
    print("="*90)

    envs = manager.list_envs()
    active = manager.get_active()

    if not envs:
        print("   No environments installed.")
        print("="*90)
        return

    print(f"{'NAME':<15} | {'TYPE':<5} | {'PYTHON':<8} | {'TORCH':<15} | {'TRITON':<9} | {'SAGE':<10} | {'SPARGE':<10} | {'FLASH':<10}")
    print("-" * 90)

    for name, data in envs.items():
        details = get_env_details(name, data)
        marker = "*" if name == active else " "
        display_name = f"[{marker}] {name}"

        if 'error' in details:
            print(f"{display_name:<15} | {data['type']:<5} | [Error reading environment]")
            continue

        print(f"{display_name:<15} | {data['type']:<5} | "
              f"{details.get('python','?'):<8} | "
              f"{details.get('torch','?'):<15} | "
              f"{details.get('triton','?'):<9} | "
              f"{details.get('sageattention','?'):<10} | "
              f"{details.get('spas_sage_attn','?'):<10} | "
              f"{details.get('flash_attn','?'):<10}")

    print("-" * 90)
    print(f" * = Active Environment")
    print("="*90 + "\n")

def install_logic(env_name, env_type, env_path, py_k, torch_k, triton_k, sage_k, sparge_k, flash_k, kernel_list, config):
    template = ENV_TEMPLATES[env_type]
    target_py_ver = config['components']['python'][py_k]['ver']

    print(f"\n[1/3] Preparing Environment: {env_name} ({env_type})...")

    if env_type != "none":
        if env_type == "venv":
            py_ver_short = ".".join(target_py_ver.split(".")[:2])

            if IS_WIN:
                create_cmd = f"py -{py_ver_short} -m venv \"{env_path}\""
            else:
                create_cmd = f"python{py_ver_short} -m venv \"{env_path}\""
        else:
            create_cmd = template["create"].format(
                ver=target_py_ver,
                dir=env_path,
                sys_py=sys.executable
            )

    if create_cmd:
        run_cmd(create_cmd)

    pip = template["install"].format(dir=env_path)

    print(f"\n[2/3] Installing Torch: {config['components']['torch'][torch_k]['label']}...")
    torch_cmd = resolve_torch_cmd(config, torch_k)
    run_cmd(f"{pip} {torch_cmd}")

    print(f"\n[3/3] Installing Requirements & Extras...")
    run_cmd(f"{pip} -r requirements.txt")

    if triton_k:
        cmd = resolve_cmd(config['components']['triton'][triton_k]['cmd'])
        if cmd: run_cmd(f"{pip} {cmd}")

    if sage_k:
        cmd = resolve_cmd(config['components']['sage'][sage_k]['cmd'])
        cmd = cmd.replace("{py}", "cp" + py_k.replace(".", ""))  # wheels built for each Python version
        if cmd.startswith("http") or cmd.startswith("sageattention"):
            run_cmd(f"{pip} {cmd}")
        else:
            if env_type == "venv" or env_type == "uv":
                act = f". \"{env_path}/bin/activate\" && " if not IS_WIN else ""
                run_cmd(f"{act}{cmd}")
            elif env_type == "conda":
                pass

    if sparge_k:
        cmd = resolve_cmd(config['components']['sparge'][sparge_k]['cmd'])
        if cmd: run_pip_component(pip, cmd, env_path=env_path if env_type != "none" else None)

    if flash_k:
        cmd = resolve_cmd(config['components']['flash'][flash_k]['cmd'])
        if cmd: run_cmd(f"{pip} {cmd}")

    for k in kernel_list:
        amd_targets = [arch for arch in ("gfx1200", "gfx1201") if f"device-{arch}" in torch_cmd]
        k = kernel_variant(k, torch_k, py_k, amd_targets)
        if k is None:
            continue
        if k in config['components']['kernels']:
            cmd = resolve_cmd(config['components']['kernels'][k]['cmd'])
            if cmd: run_cmd(f"{pip} {cmd}")

    install_plugin_requirements(pip)

def menu(title, options, recommended_key=None):
    print(f"\n--- {title} ---")
    keys = list(options.keys())
    for i, k in enumerate(keys):
        rec = " [RECOMMENDED FOR YOUR GPU]" if k == recommended_key else ""
        print(f"{i+1}. {options[k]['label']}{rec}")
    choice = input(f"Select option (Enter for Recommended): ")
    if choice == "" and recommended_key: return recommended_key
    try: return keys[int(choice)-1]
    except: return recommended_key

def recommended_triton(profile, torch_key):
    if not profile['triton'] or profile['triton'] == 'v33':
        return profile['triton']
    return {'cu128': 'v34', 'cu130': 'v37', 'rocm10': 'v38'}.get(torch_key)

def do_install_interactive(env_type, config, detected_key):
    manager = EnvsManager()
    create_wgp_config(detected_key, config)

    default_name = f"env_{env_type}" if env_type != "none" else "system"
    print(f"\n--- Configuration for {env_type} ---")
    name = input(f"Enter a name for this environment (Default: {default_name}): ").strip()
    if not name: name = default_name

    cwd = os.getcwd()
    path = os.path.join(cwd, name) if env_type != "none" else ""

    if name in manager.list_envs():
        print(f"\n[!] Warning: Environment '{name}' already exists in registry.")
        choice = input("Do you want to overwrite it? (This will delete the old folder) [y/N]: ").lower()
        if choice != 'y': return
        manager.remove_env(name)
    elif os.path.exists(path) and env_type != "none":
        print(f"\n[!] Warning: Directory '{path}' exists but is not registered.")
        choice = input("Do you want to overwrite this directory? [y/N]: ").lower()
        if choice != 'y': return
        try: shutil.rmtree(path)
        except: pass

    print("\n--- Select Install Mode ---")
    print("1. Autoselect (Based on your GPU)")
    print("2. Manual Selection")

    mode = input("Select option (1-2) [Default: 1]: ").strip()

    if mode == "2":
        base = config['gpu_profiles'][detected_key]
        py_k = menu("Python Version", config['components']['python'], base['python'])
        torch_k = menu("Torch Version", config['components']['torch'], base['torch'])
        triton_k = menu("Triton", config['components']['triton'], recommended_triton(base, torch_k))
        sage_k = menu("Sage Attention", config['components']['sage'], base['sage'])
        sparge_k = menu("Sparge Attention", config['components']['sparge'], base.get('sparge'))
        flash_k = menu("Flash Attention", config['components']['flash'], base['flash'])
        kernels = base['kernels']

        install_logic(name, env_type, path, py_k, torch_k, triton_k, sage_k, sparge_k, flash_k, kernels, config)

    else:
        p = config['gpu_profiles'][detected_key]
        install_logic(name, env_type, path, p['python'], p['torch'], p['triton'], p['sage'], p.get('sparge'), p.get('flash'), p['kernels'], config)

    manager.add_env(name, env_type, path)

    if len(manager.list_envs()) > 1:
        choice = input(f"\nDo you want to make '{name}' the active environment? [Y/n]: ").lower()
        if choice != 'n':
            manager.set_active(name)
    else:
        print(f"\n[*] '{name}' is the only environment, setting as active.")
        manager.set_active(name)

def do_install_auto(env_type, config, detected_key):
    manager = EnvsManager()
    create_wgp_config(detected_key, config)

    name = f"env_{env_type}" if env_type != "none" else "system"
    cwd = os.getcwd()
    path = os.path.join(cwd, name) if env_type != "none" else ""

    if name in manager.list_envs():
        manager.remove_env(name)
    elif os.path.exists(path) and env_type != "none":
        try: shutil.rmtree(path)
        except: pass

    print(f"\n[*] Starting Automatic Install (Hardware Profile: {detected_key})...")
    p = config['gpu_profiles'][detected_key]

    install_logic(name, env_type, path, p['python'], p['torch'], p['triton'], p['sage'], p.get('sparge'), p.get('flash'), p['kernels'], config)

    manager.add_env(name, env_type, path)
    manager.set_active(name)
    print(f"\n[*] Automatic Install Complete! '{name}' is now active.")

def open_terminal():
    manager = EnvsManager()
    env_name = manager.get_active()

    if not env_name:
        print("[!] No active environment. Please select or install one first.")
        input("Press Enter...")
        return

    env_data = manager.list_envs().get(env_name)
    if not env_data:
        print(f"[!] Could not find environment data for '{env_name}'.")
        return

    e_type = env_data["type"]
    e_path = env_data["path"]

    print(f"\n[*] Spawning interactive terminal for '{env_name}'...")
    print(f"[*] (Type 'exit' when you are done to return to the menu)\n")

    if IS_WIN:
        if e_type in ["venv", "uv"]:
            act_bat = os.path.join(e_path, 'Scripts', 'activate.bat')
            subprocess.run(f'cmd.exe /K "{act_bat}"')
        elif e_type == "conda":
            conda_bat = "conda.bat"
            if not shutil.which("conda"):
                user = os.environ.get("USERPROFILE", "")
                paths = [
                    os.path.join(user, "Miniconda3", "condabin", "conda.bat"),
                    os.path.join(user, "Anaconda3", "condabin", "conda.bat"),
                    r"C:\ProgramData\Miniconda3\condabin\conda.bat"
                ]
                for p in paths:
                    if os.path.exists(p):
                        conda_bat = p
                        break
            subprocess.run(f'cmd.exe /K "{conda_bat}" activate "{e_path}"')
        else:
            subprocess.run('cmd.exe /K')
    else:
        rc_cmd = "if [ -f ~/.bashrc ]; then source ~/.bashrc; fi\n"
        if e_type in ["venv", "uv"]:
            rc_cmd += f"source '{os.path.join(e_path, 'bin', 'activate')}'\n"
        elif e_type == "conda":
            rc_cmd += (
                "if command -v conda >/dev/null 2>&1; then eval \"$(conda shell.bash hook)\"; "
                "else for base in \"$HOME/miniconda3\" \"$HOME/anaconda3\" \"/opt/miniconda3\" \"/opt/anaconda3\"; do "
                "if [ -f \"$base/etc/profile.d/conda.sh\" ]; then source \"$base/etc/profile.d/conda.sh\"; break; fi; done; fi\n"
                f"conda activate '{e_path}'\n"
            )

        linux_shell_cmd = f"bash --rcfile <(cat << 'EOF_WAN2GP'\n{rc_cmd}EOF_WAN2GP\n)"
        subprocess.run(linux_shell_cmd, shell=True, executable='/bin/bash')

def switch_git_branch():
    if not os.path.exists(".git"):
        print("[!] Not a git repository. Cannot switch branches.")
        input("Press Enter...")
        return
        
    print("\n[*] Fetching latest branches from remote...")
    try:
        subprocess.run(["git", "fetch", "--prune"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.check_output(["git", "branch", "-a"], encoding='utf-8')
    except Exception as e:
        print(f"[!] Git error: {e}")
        input("Press Enter...")
        return
        
    branches = set()
    active_branch = ""
    for line in out.splitlines():
        line = line.strip()
        if not line: continue
        if "->" in line: continue
        
        is_active = line.startswith("*")
        b = line.lstrip("* ").strip()
        
        if b.startswith("remotes/origin/"):
            b = b[len("remotes/origin/"):]
        elif b.startswith("remotes/"):
            b = "/".join(b.split("/")[2:])
            
        if b and b != "HEAD":
            branches.add(b)
        if is_active:
            active_branch = b
            
    branch_list = sorted(list(branches))
    if not branch_list:
        print("[!] No branches found.")
        input("Press Enter...")
        return
        
    print("\n--- Available Branches ---")
    for i, b in enumerate(branch_list):
        marker = "*" if b == active_branch else " "
        print(f"{i+1}. [{marker}] {b}")
        
    val = input(f"\nEnter number or name of branch (Default: {active_branch}): ").strip()
    if not val:
        return
        
    target = val
    if val.isdigit():
        idx = int(val) - 1
        if 0 <= idx < len(branch_list):
            target = branch_list[idx]
            
    if target == active_branch:
        print(f"[*] Already on '{target}'.")
        input("Press Enter...")
        return

    print(f"[*] Switching to '{target}'...")
    try:
        subprocess.run(["git", "checkout", target], check=True, capture_output=True, text=True)
        print(f"[*] Successfully switched to '{target}'.")
        
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.lower() if e.stderr else ""
        
        if "overwritten" in err_msg or "please commit your changes or stash them" in err_msg:
            print(f"\n[!] Failed to switch to '{target}' due to uncommitted changes.")
            print("How would you like to handle this?")
            print("  1. Carry changes over to new branch (Not recommended)")
            print("  2. Stash changes")
            print("  3. Discard modifications")
            print("  4. Cancel")
            opt = input("\nSelect option (1-4): ").strip()
            
            if opt == "1":
                try:
                    print("\n[*] Stashing changes...")
                    subprocess.run(["git", "stash", "push", "-m", f"Auto-stash before carrying to {target}"], check=True)
                    subprocess.run(["git", "checkout", target], check=True)
                    print(f"[*] Switched to '{target}'. Reapplying changes...")
                    pop_res = subprocess.run(["git", "stash", "pop"])
                    if pop_res.returncode != 0:
                        print("\n[!] Note: There were merge conflicts when reapplying your changes.")
                        print("    The changes are still saved in your stash, but you will need to resolve conflicts in the files.")
                    else:
                        print("[*] Successfully carried changes over.")
                except Exception as inner_e:
                    print(f"[!] Failed to carry changes: {inner_e}")
            elif opt == "2":
                try:
                    print("\n[*] Stashing changes...")
                    subprocess.run(["git", "stash", "push", "-m", f"Auto-stash before switching to {target}"], check=True)
                    subprocess.run(["git", "checkout", target], check=True)
                    print(f"[*] Successfully set changes aside and switched to '{target}'.")
                except Exception as inner_e:
                    print(f"[!] Failed to stash and switch: {inner_e}")
            elif opt == "3":
                try:
                    print("\n[*] Discarding tracked changes...")
                    subprocess.run(["git", "checkout", "-f", target], check=True)
                    print(f"[*] Successfully switched to '{target}'.")
                except Exception as inner_e:
                    print(f"[!] Failed to force switch: {inner_e}")
            else:
                print("[*] Cancelled.")
        else:
            print(f"[!] Error switching branch:\n{e.stderr.strip() if e.stderr else str(e)}")
            
    except Exception as e:
        print(f"[!] Error: {e}")
        
    input("Press Enter...")

def manage_git_stashes():
    if not os.path.exists(".git"):
        print("[!] Not a git repository.")
        input("Press Enter...")
        return

    try:
        out = subprocess.check_output(["git", "stash", "list"], encoding='utf-8')
    except Exception as e:
        print(f"[!] Git error: {e}")
        input("Press Enter...")
        return

    stashes = [line for line in out.splitlines() if line.strip()]
    if not stashes:
        print("[*] No stashed changes found. You're all clean!")
        input("Press Enter...")
        return

    print("\n--- Saved Stashes ---")
    for i, s in enumerate(stashes):
        print(f"{i+1}. {s}")

    print("\nOptions:")
    print("  A. Apply a stash (Restore changes but keep the stash)")
    print("  P. Pop a stash (Restore changes and delete the stash)")
    print("  D. Delete a stash")
    print("  C. Cancel")

    choice = input("\nSelect action (A/P/D/C): ").strip().upper()
    if choice not in ["A", "P", "D"]:
        return

    idx_str = input(f"Enter stash number (1-{len(stashes)}): ").strip()
    if not idx_str.isdigit() or not (1 <= int(idx_str) <= len(stashes)):
        print("[!] Invalid stash number.")
        input("Press Enter...")
        return

    stash_ref = f"stash@{{{int(idx_str)-1}}}"

    try:
        if choice == "A":
            print(f"\n[*] Applying {stash_ref}...")
            subprocess.run(["git", "stash", "apply", stash_ref])
        elif choice == "P":
            print(f"\n[*] Popping {stash_ref}...")
            subprocess.run(["git", "stash", "pop", stash_ref])
        elif choice == "D":
            conf = input(f"Are you sure you want to delete {stash_ref}? (y/n): ").strip().lower()
            if conf == 'y':
                subprocess.run(["git", "stash", "drop", stash_ref])
                print(f"[*] Dropped {stash_ref}.")
    except Exception as e:
        print(f"\n[!] Command returned an error or warning (you may need to resolve file conflicts manually).")

    input("\nPress Enter...")

def do_manage():
    manager = EnvsManager()
    while True:
        os.system('cls' if IS_WIN else 'clear')
        print("==========================================================================================")
        print(f"{'ENVIRONMENT MANAGER':^90}")
        print("==========================================================================================")
        envs = manager.list_envs()
        active = manager.get_active()
        keys = list(envs.keys())
        
        if not envs:
            print(" No environments installed.")
        else:
            for name in keys:
                data = envs[name]
                status = "(Active)" if name == active else ""
                print(f" - {name:<15} [{data['type']}] {status}")
        
        print("------------------------------------------------------------------------------------------")
        print("1. Set Active Environment")
        print("2. Delete Environment")
        print("3. Add Existing Environment")
        print("4. List Environment Details")
        print("5. Open Terminal in Active Environment")
        print("6. Switch Git Branch")
        print("7. Manage Git Stashes")
        print("8. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == "1":
            if not keys:
                input("No environments to activate. Press Enter...")
                continue
            print("\nAvailable Environments:")
            for i, k in enumerate(keys):
                print(f"  {i+1}. {k}")
            val = input("\nEnter name or number of environment to activate: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(keys):
                name = keys[int(val)-1]
            else:
                name = val
            manager.set_active(name)
            input("Press Enter...")
            
        elif choice == "2":
            if not keys:
                input("No environments to delete. Press Enter...")
                continue
            print("\nAvailable Environments:")
            for i, k in enumerate(keys):
                print(f"  {i+1}. {k}")
            val = input("\nEnter name or number of environment to DELETE: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(keys):
                name = keys[int(val)-1]
            else:
                name = val
                
            if name in envs:
                conf = input(f"Are you sure you want to delete '{name}' and its files? (y/n): ")
                if conf.lower() == 'y':
                    manager.remove_env(name)
                    input("Deleted. Press Enter...")
            else:
                print(f"[!] Environment '{name}' not found.")
                input("Press Enter...")
                
        elif choice == "3":
            path = input("Enter the path to the existing environment folder: ").strip()
            if not os.path.exists(path):
                print("[!] Error: Path does not exist.")
            else:
                name = input("Enter a nickname for this environment: ").strip()
                if not name: name = os.path.basename(path.rstrip(os.sep))
                
                print("\nSelect Environment Type:")
                print("1. venv")
                print("2. uv")
                print("3. conda")
                t_choice = input("Choice (Default 1): ")
                e_type = "uv" if t_choice == "2" else "conda" if t_choice == "3" else "venv"
                manager.add_env(name, e_type, os.path.abspath(path))
                print(f"[*] Registered '{name}' at {os.path.abspath(path)}")
            input("Press Enter...")
            
        elif choice == "4":
            show_status()
            input("Press Enter...")
            
        elif choice == "5":
            open_terminal()
            
        elif choice == "6":
            switch_git_branch()
            
        elif choice == "7":
            manage_git_stashes()
            
        elif choice == "8":
            break

def do_upgrade(config, profile_key=None):
    manager = EnvsManager()
    print("\n" + "="*90)
    print(f"{'WAN2GP MANUAL COMPONENT UPGRADE':^90}")
    print("="*90)

    env_name = manager.resolve_target_env()
    env_data = manager.list_envs()[env_name]

    gpu_name, vendor = get_gpu_info()
    profile_key = profile_key or get_profile_key(gpu_name, vendor)
    if profile_key not in config['gpu_profiles']:
        raise ValueError("Hardware profile is unknown. Run setup.py upgrade --profile with an explicit catalog profile.")
    rec = config['gpu_profiles'][profile_key]
    if get_os_key() not in rec.get("platforms", ("win", "linux", "macos")):
        raise ValueError(f"Hardware profile {profile_key} is not supported on {get_os_key()}")

    py_k = menu("Python Version", config['components']['python'], rec['python'])
    torch_k = menu("Torch Version", config['components']['torch'], rec['torch'])
    triton_k = menu("Triton", config['components']['triton'], recommended_triton(rec, torch_k))
    sage_k = menu("Sage Attention", config['components']['sage'], rec['sage'])
    sparge_k = menu("Sparge Attention", config['components']['sparge'], rec.get('sparge'))
    flash_k = menu("Flash Attention", config['components']['flash'], rec['flash'])

    install_logic(env_name, env_data['type'], env_data['path'], py_k, torch_k, triton_k, sage_k, sparge_k, flash_k, rec['kernels'], config)

def get_system_specs():
    ram_gb = 0
    vram_gb = 0

    if IS_WIN:
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                encoding='utf-8', stderr=subprocess.DEVNULL
            ).strip()
            if out:
                ram_gb = int(out) / (1024**3)
        except:
            try:
                out = subprocess.check_output(
                    "wmic computersystem get TotalPhysicalMemory /value",
                    shell=True, encoding='utf-8', stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    if "TotalPhysicalMemory=" in line:
                        ram_gb = int(line.split('=')[1]) / (1024**3)
                        break
            except:
                pass
    else:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        kb_val = float(line.split()[1])
                        ram_gb = kb_val / (1024**2)
                        break
        except: pass

    if ram_gb == 0:
        print("[!] Warning: Could not detect System RAM. Defaulting to 16GB.")
        ram_gb = 16

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8', stderr=subprocess.DEVNULL
        ).strip()
        vram_gb = float(out.split('\n')[0]) / 1024
    except:
        vram_gb = get_adapter_vram_gb()
        if not vram_gb:
            print("[!] Warning: Could not detect VRAM. Defaulting to 8GB.")
            vram_gb = 8

    return ram_gb, vram_gb

def get_adapter_vram_gb():
    """Largest dedicated VRAM among the display adapters, for GPUs nvidia-smi does not cover."""
    sizes = []
    if IS_WIN:
        import winreg
        # Win32_VideoController.AdapterRAM is a 32-bit field that caps at 4GB
        adapters = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, adapters) as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    try:
                        with winreg.OpenKey(root, winreg.EnumKey(root, i)) as key:
                            size = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                        sizes.append(int.from_bytes(size, "little") if isinstance(size, bytes) else int(size))
                    except OSError: pass
        except OSError: pass
    else:
        drm = "/sys/class/drm"
        for card in os.listdir(drm) if os.path.isdir(drm) else []:
            try:
                with open(os.path.join(drm, card, "device", "mem_info_vram_total")) as f:
                    sizes.append(int(f.read()))
            except: pass
    return max(sizes, default=0) / (1024**3)

def create_wgp_config(profile_key, config_data):
    WGP_CONFIG_FILE = "wgp_config.json"

    if os.path.exists(WGP_CONFIG_FILE):
        return

    print("\n[*] Auto-generating wgp_config.json based on hardware...")

    ram, vram = get_system_specs()
    print(f"    Detected: {int(ram)}GB RAM / {int(vram)}GB VRAM")

    prof_settings = config_data['gpu_profiles'].get(profile_key, {})

    attn_mode = prof_settings.get("attention", "")
    if not attn_mode:
        if "50" in profile_key or "40" in profile_key or "30" in profile_key:
            attn_mode = "sage2"
        elif "20" in profile_key:
            attn_mode = "sage"

    config_out = recommend_settings({}, ram_gb=ram, vram_gb=vram, attention=attn_mode)["patch"]

    try:
        with open(WGP_CONFIG_FILE, 'w') as f:
            json.dump(config_out, f, indent=4)
        print(f"    Created config with Video/Image Profile {config_out['video_profile']}, Audio Profile {config_out['audio_profile']}, Attention: '{attn_mode}', Compile: '{config_out['compile']}'")
    except Exception as e:
        print(f"[!] Error writing config: {e}")

def inject_system_paths():
    if not IS_WIN:
        return

    paths = []
    user = os.environ.get("USERPROFILE", "")
    local_app = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")

    for base in [os.path.join(user, "Miniconda3"), os.path.join(user, "Anaconda3"), r"C:\ProgramData\Miniconda3"]:
        c_bin = os.path.join(base, "condabin")
        if os.path.exists(c_bin):
            paths.extend([c_bin, os.path.join(base, "Scripts"), base])
            break

    if user: paths.append(os.path.join(user, ".local", "bin"))
    if appdata: paths.append(os.path.join(appdata, "uv", "bin"))

    if local_app:
        paths.extend([
            os.path.join(local_app, "Programs", "Python", "PyManager"),
            os.path.join(local_app, "Programs", "Python", "Python311", "Scripts")
        ])

    current_path = os.environ.get("PATH", "")
    for p in paths:
        if p and os.path.exists(p) and p not in current_path:
            current_path = f"{p};{current_path}"

    os.environ["PATH"] = current_path

def repair_git_repo():
    print("[*] Repairing WAN2GP repository...")
    if not os.path.exists(".git"):
        run_cmd("git init")

    try:
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/deepbeepmeep/Wan2GP.git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

    run_cmd("git fetch origin")

    try:
        subprocess.run(["git", "rev-parse", "--verify", "origin/main"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        default_branch = "main"
    except subprocess.CalledProcessError:
        default_branch = "master"

    print(f"[*] Force resetting local files to match origin/{default_branch}...")
    run_cmd(f"git reset --hard origin/{default_branch}")
    run_cmd(f"git branch -M {default_branch}")
    run_cmd(f"git branch --set-upstream-to=origin/{default_branch} {default_branch}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manager":
        from installation.cli import main as installation_main
        sys.exit(installation_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        from installation.cli import main as installation_main
        sys.exit(installation_main(sys.argv[1:]))
    inject_system_paths()
    parser = argparse.ArgumentParser(epilog="Read-only Manager planning: python setup.py plan --help")
    parser.add_argument("mode", choices=["install", "update", "upgrade", "status", "manage", "get_env_info"])
    parser.add_argument("--env", default="venv", help="Type of env for install (venv, uv, conda, none)")
    parser.add_argument("--auto", action="store_true", help="Run 1-click automatic install")
    parser.add_argument("--profile", help="Explicit hardware profile from setup_config.json")
    args = parser.parse_args()
    cfg = load_config()
    if args.profile is not None and args.profile not in cfg['gpu_profiles']:
        parser.error(f"Unknown profile {args.profile!r}; choose from {', '.join(cfg['gpu_profiles'])}")
    record = managed_record(os.getcwd())
    if record and args.mode not in ("status", "get_env_info"):
        parser.error(f"Managed installation detected at {record}. Legacy mutations are disabled; use setup.py manager prepare/apply/recover.")

    if args.mode == "get_env_info":
        if record:
            from installation.state import InstallationLock
            from installation.transactions import ready_record
            try:
                with InstallationLock(record.parent.parent):
                    ready_record(record.parent.parent)
            except (OSError, ValueError, RuntimeError) as exc:
                parser.error(str(exc))
        manager = EnvsManager()
        active = manager.get_active()

        if not active or not manager.list_envs().get(active):
            sys.exit(1)

        env_data = manager.list_envs()[active]
        print(f"ENV_INFO|{env_data['type']}|{env_data['path']}")
        sys.exit(0)

    if args.mode == "status":
        show_status()
        sys.exit(0)

    if args.mode == "manage":
        do_manage()
        sys.exit(0)

    if args.mode == "install":
        gpu_name, vendor = get_gpu_info()
        profile_key = args.profile or get_profile_key(gpu_name, vendor)
        if profile_key is None:
            parser.error(f"No automatic profile for {gpu_name!r}; select --profile from {', '.join(cfg['gpu_profiles'])}")
        if get_os_key() not in cfg['gpu_profiles'][profile_key].get("platforms", ("win", "linux", "macos")):
            parser.error(f"Hardware profile {profile_key} is not supported on {get_os_key()}")
        print(f"Hardware Detected: {gpu_name} ({vendor})")
        if args.auto:
            do_install_auto(args.env, cfg, profile_key)
        else:
            do_install_interactive(args.env, cfg, profile_key)

    elif args.mode == "update":
        manager = EnvsManager()
        env_name = manager.resolve_target_env()
        env_data = manager.list_envs()[env_name]

        needs_install = False

        if not os.path.exists(".git"):
            print("[*] No .git folder found.")
            repair_git_repo()
            needs_install = True
        else:
            print("[*] Checking for updates...")
            try:
                old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding='utf-8', stderr=subprocess.DEVNULL).strip()
            except:
                old_head = ""

            try:
                subprocess.run(["git", "pull"], check=True)
                new_head = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding='utf-8', stderr=subprocess.DEVNULL).strip()

                if old_head != new_head or not old_head:
                    needs_install = True

            except subprocess.CalledProcessError:
                print("\n[!] 'git pull' failed.")
                print("[*] Attempting automatic recovery...")
                repair_git_repo()
                needs_install = True

        if needs_install:
            print("\n[*] Updates found. Installing/Verifying requirements...")
            install_fmt = ENV_TEMPLATES[env_data['type']]['install']
            pip_cmd = install_fmt.format(dir=env_data['path'])
            run_cmd(f"{pip_cmd} -r requirements.txt")
        else:
            print("\n[*] Code is already up to date. Skipping requirements installation.")

    elif args.mode == "upgrade":
        try:
            do_upgrade(cfg, args.profile)
        except ValueError as exc:
            parser.error(str(exc))
