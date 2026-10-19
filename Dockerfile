FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Build arg for GPU architectures - specify which CUDA compute capabilities to compile for
# Common values:
#   7.0  - Tesla V100
#   7.5  - RTX 2060, 2070, 2080, Titan RTX
#   8.0  - A100, A800 (Ampere data center)
#   8.6  - RTX 3060, 3070, 3080, 3090 (Ampere consumer)
#   8.9  - RTX 4070, 4080, 4090 (Ada Lovelace)
#   9.0  - H100, H800 (Hopper data center)
#   12.0 - RTX 5070, 5080, 5090 (Blackwell) - Note: sm_120 architecture
#
# Examples:
#   RTX 3060: --build-arg CUDA_ARCHITECTURES="8.6"
#   RTX 4090: --build-arg CUDA_ARCHITECTURES="8.9"
#   Multiple: --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9"
#
# Note: Including 8.9 or 9.0 may cause compilation issues on some setups
# Default includes 8.0 and 8.6 for broad Ampere compatibility
ARG CUDA_ARCHITECTURES="8.0;8.6"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y \
    python3 python3-pip git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg && \
    apt clean

WORKDIR /workspace

COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# First install torch with the versions we want, so that stuff in requirements.txt doesn't pull in the generic versions
# If you change CUDA 12.8 here, you also need to change the FROM docker image at the top
RUN pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install requirements if exists
RUN pip install -r requirements.txt

# Install SageAttention from git (patch GPU detection)
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="8"

COPY <<EOF /tmp/patch_setup.py
import os
with open('setup.py', 'r') as f:
    content = f.read()

# Get architectures from environment variable
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
arch_set = '{' + ', '.join([f'"{arch}"' for arch in arch_list.split(';')]) + '}'

# Replace the GPU detection section
old_section = '''compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")'''

new_section = 'compute_capabilities = ' + arch_set + '''
print(f"Manually set compute capabilities: {compute_capabilities}")'''

content = content.replace(old_section, new_section)

with open('setup.py', 'w') as f:
    f.write(content)
EOF

RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    python3 /tmp/patch_setup.py && \
    pip install --no-build-isolation .

RUN useradd -u 1000 -ms /bin/bash user

RUN chown -R user:user /workspace

RUN mkdir /home/user/.cache && \
    chown -R user:user /home/user/.cache

COPY entrypoint.sh /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
