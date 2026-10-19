# MiniMax H3 FL2VA Video Generation Model

## Overview
**FL2VA** = First-Last 2-Video Architecture
- Video generation model with first + last frame conditioning
- Generates up to 15 seconds of 2K video with synced audio
- Cloud-based API model (NOT local GGUF like MiniMax-M2.7)

## Architecture
- **DiT (Diffusion Transformer)**: Main generation model
- **Dual VAE**: Separate video VAE + audio VAE for high quality
- **Qwen3-VL-32B**: Vision-language encoder for text/image understanding
- **Conditioning**: First frame + Last frame + Text prompt

## Model Files Needed
```
MiniMax-H3/
├── MiniMax-H3-FL2VA-int8_convrot.safetensors  (DiT transformer)
├── qwen3-vl-32b-int8_convrot.safetensors      (text encoder)
├── MiniMax-H3-video_vae.safetensors           (video VAE)
└── MiniMax-H3-audio_vae.safetensors           (audio VAE)
```

## Workflow Structure
1. **LoadImage** → First frame
2. **LoadImage** → Last frame  
3. **FL2VAFirstFrameCondition** → Combine frames → `keyframes`
4. **FL2VATarget** → Set duration (15s) + resolution (832×480, 4:3)
5. **FL2VAEncode** → Text prompt + keyframes → `conditioning`
6. **DualSigmaSampler** → Generate video+audio latent (50 steps, 12 sigma)
7. **DecodeAV** → Latent → Frames + Audio
8. **CreateVideo** → Frames + Audio → MP4 (24fps, 8-bit)
9. **SaveVideo** → Output file

## Key Parameters
- **Duration**: 15 seconds
- **Resolution**: 832×480 (4:3 aspect ratio)
- **Steps**: 50
- **Sigma Points**: 12
- **FPS**: 24
- **Sampler**: Euler

## Example Prompt
```
"0-2s: medium wide shot, the boy raises hand to adjust aviator cap, dusk wilderness background at high altitude
2-5s: camera steadily pushes to facial close-up, arms lean forward gazing afar, clothes and hair sway softly, light slowly covers the face."
```

## Comparison with MiniMax-M2.7
| Feature | MiniMax H3 FL2VA | MiniMax-M2.7 |
|---------|------------------|--------------|
| Type | Cloud API | Local GGUF model |
| Video | ✅ Yes, 15s, 2K | ❌ No |
| Audio | ✅ Synced | ❌ No |
| Context | 128K-256K | ~8K-32K |
| Speed | ~2-5 min/clip | GPU dependent |
| Tool Calling | ✅ Native | ❌ No |

## References
- Workflow example: `MiniMax H3 FL2VA first-last frame to Video.json`
- Documentation: https://vrfi1sk8a0.feishu.cn/wiki/FIWjwgL33ipnkekzk30crmKUnIh
