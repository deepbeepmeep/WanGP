#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
RunPod Diagnostic & GPU Discovery Tool for Wan2GP
Tests RunPod API connectivity, queries GPU specs/availability, and verifies
serverless configuration WITHOUT spending any credits.
"""

import os
import sys
from pathlib import Path

# Load .env if present
env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("\"'")
            if k and k not in os.environ:
                os.environ[k] = v

def main():
    print("=" * 64)
    print("    🔍 RunPod Serverless & GPU Diagnostic (Zero Credits)   ")
    print("=" * 64)

    try:
        import runpod
    except ImportError:
        print("[!] runpod SDK is not installed. Install via: pip install runpod")
        sys.exit(1)

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()

    if not api_key:
        print("\n[!] RUNPOD_API_KEY is not set.")
        print("    You can create a free account at https://runpod.io, get an API key,")
        print("    and set it in your .env or export RUNPOD_API_KEY=your_key.")
        print("    Note: Querying GPUs and testing setup costs $0.00.\n")
        sys.exit(1)

    runpod.api_key = api_key
    print(f"\n[✓] Using API Key: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")

    # 1. Test User / Account
    print("\n[1/3] Verifying Account Information...")
    try:
        user = runpod.get_user()
        if user:
            print(f" [✓] Connected as: {user.get('email', 'User')}")
            print(f"     Account Balance: ${user.get('creditBalance', 0.0):.2f}")
    except Exception as e:
        print(f" [!] Account lookup error: {e}")

    # 2. Query GPU Types & Specs
    print("\n[2/3] Fetching Available GPU Hardware Catalog from RunPod...")
    try:
        gpus = runpod.get_gpus()
        if gpus and isinstance(gpus, list):
            print(f" [✓] Successfully retrieved {len(gpus)} GPU types from RunPod.")
            print("\n  Top Recommended GPUs for Wan2GP:")
            print("  -------------------------------------------------------------")
            print("  GPU Model                 VRAM     Best Use Case in Wan2GP")
            print("  -------------------------------------------------------------")
            
            # Sort by VRAM descending
            gpus_sorted = sorted(gpus, key=lambda x: x.get("memoryInGb", 0), reverse=True)
            for g in gpus_sorted:
                name = g.get("displayName", g.get("id", "Unknown"))
                vram = g.get("memoryInGb", 0)
                
                # Highlight GPUs particularly relevant for Wan2GP
                if any(k in name.upper() for k in ["4090", "L40", "A100", "H100", "3090", "L4"]):
                    if "4090" in name:
                        use_case = "Cheapest (~$0.44/hr) - Wan 1.3B, Quantized 14B"
                    elif "L40" in name:
                        use_case = "Sweet Spot (~$0.74/hr) - 48GB VRAM native 14B"
                    elif "A100" in name and vram >= 80:
                        use_case = "High Bandwidth (~$1.89/hr) - 80GB SXM Ultra-Fast"
                    elif "H100" in name:
                        use_case = "Top Speed (~$3.50/hr) - Native FP8 Transformer Engine"
                    else:
                        use_case = f"General inference ({vram} GB)"
                    
                    print(f"  {name:<25} {vram:>2} GB   {use_case}")
            print("  -------------------------------------------------------------")
    except Exception as e:
        print(f" [!] Failed to retrieve GPU catalog: {e}")

    # 3. Check Serverless Endpoint
    print("\n[3/3] Checking Serverless Endpoints...")
    if endpoint_id:
        print(f" [✓] Configured Endpoint ID: {endpoint_id}")
        try:
            endpoints = runpod.get_endpoints()
            found = False
            for ep in (endpoints or []):
                if ep.get("id") == endpoint_id:
                    found = True
                    print(f"     Endpoint Name: {ep.get('name')}")
                    print(f"     Idle Timeout: {ep.get('idleTimeout')}s")
                    print(f"     Workers (Min/Max): {ep.get('workersMin')}/{ep.get('workersMax')}")
                    break
            if not found:
                print(f"     Note: Endpoint '{endpoint_id}' was not found in endpoint list or is private.")
        except Exception as e:
            print(f"     Warning querying endpoint list: {e}")
    else:
        print(" [!] No RUNPOD_ENDPOINT_ID set in .env yet.")
        print("     To deploy a serverless worker, configure an endpoint in RunPod console.")

    print("\n" + "=" * 64)
    print(" Diagnostic Complete! (0 credits used)")
    print("=" * 64)

if __name__ == "__main__":
    main()
