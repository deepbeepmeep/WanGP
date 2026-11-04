import random
import os
import torch
from PIL import Image
import subprocess
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
# from shared.utils.multitalk_utils import save_video_ffmpeg
from transformers import Wav2Vec2FeatureExtractor
from .wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import math
from shared.utils import files_locator as fl 
from shared.utils.subprocess_stdin import child_kwargs

def custom_init(device, wav2vec):    
    from mmgp import offload
    wav2vec_folder = wav2vec 
    audio_encoder = offload.fast_load_transformers_model(
        os.path.join(wav2vec_folder, "pytorch_model.bin"),
        modelClass=Wav2Vec2Model,
        defaultConfigPath=os.path.join(wav2vec_folder, "config.json"),
        writable_tensors=False,
    ).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_folder, local_files_only=True
    )
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

 
def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu', fps = 25):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * fps

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True, **child_kwargs())
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000, duration = 0):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, duration=duration, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

 
def audio_prepare_multi(left_path, right_path, audio_type = "add", sample_rate=16000, duration = 0, pad = 0, min_audio_duration = 0):
    if not (left_path==None or right_path==None):
        human_speech_array1 = audio_prepare_single(left_path, duration = duration)
        human_speech_array2 = audio_prepare_single(right_path, duration = duration)
    else:
        audio_type='para'
        if left_path==None:
            human_speech_array2 = audio_prepare_single(right_path, duration = duration)
            human_speech_array1 = np.zeros(human_speech_array2.shape[0])
        elif right_path==None:
            human_speech_array1 = audio_prepare_single(left_path, duration = duration)
            human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
        if len(new_human_speech1) != len(new_human_speech2):
            if len(new_human_speech1) < len(new_human_speech2):
                new_human_speech1 = np.pad(new_human_speech1, (0, len(new_human_speech2) - len(new_human_speech1)))
            else:
                new_human_speech2 = np.pad(new_human_speech2, (0, len(new_human_speech1) - len(new_human_speech2)))
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])


    duration_changed = False
    if min_audio_duration  > 0:
        min_samples =  math.ceil( min_audio_duration * sample_rate)
        if len(new_human_speech1) < min_samples:
            new_human_speech1 = np.concatenate([new_human_speech1, np.zeros(min_samples -len(new_human_speech1)) ]) 
            duration_changed = True
        if len(new_human_speech2) < min_samples:
            new_human_speech2 = np.concatenate([new_human_speech2, np.zeros(min_samples -len(new_human_speech2)) ]) 
            duration_changed = True

    #dont include the padding on the summed audio which is used to build the output audio track
    sum_human_speechs = new_human_speech1 + new_human_speech2

    if pad  > 0:
        duration_changed = True
        new_human_speech1 = np.concatenate([np.zeros(pad), new_human_speech1])
        new_human_speech2 = np.concatenate([np.zeros(pad), new_human_speech2])

    return new_human_speech1, new_human_speech2, sum_human_speechs, duration_changed


def get_full_audio_embeddings(audio_guide1 = None, audio_guide2 = None, combination_type ="add", num_frames =  0, fps = 25, sr = 16000, padded_frames_for_embeddings = 0, min_audio_duration = 0, return_sum_only = False):
    from models.wan.multitalk.assets import query_download_def
    from shared.utils.download import process_files_def_if_needed

    process_files_def_if_needed(query_download_def())
    wav2vec_feature_extractor, audio_encoder= custom_init('cpu', fl.locate_folder("chinese-wav2vec2-base"))
    # wav2vec_feature_extractor, audio_encoder= custom_init('cpu', "ckpts/wav2vec")
    pad = int(padded_frames_for_embeddings/ fps * sr)
    new_human_speech1, new_human_speech2, sum_human_speechs, duration_changed = audio_prepare_multi(audio_guide1, audio_guide2, combination_type, duration= num_frames / fps, pad = pad, min_audio_duration = min_audio_duration )
    if return_sum_only:
        full_audio_embs = None
    else:
        audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, sr=sr, fps= fps)
        audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, sr=sr, fps= fps)
        full_audio_embs = []
        if audio_guide1 != None: full_audio_embs.append(audio_embedding_1)
        if audio_guide2 != None: full_audio_embs.append(audio_embedding_2)
        if audio_guide2 == None and not duration_changed: sum_human_speechs = None
    return full_audio_embs, sum_human_speechs


def get_window_audio_embeddings(full_audio_embs, audio_start_idx=0, clip_length = 81, vae_scale = 4, audio_window = 5):
    if full_audio_embs == None: return None
    HUMAN_NUMBER = len(full_audio_embs)
    audio_end_idx = audio_start_idx + clip_length
    indices = (torch.arange(2 * 2 + 1) - 2) * 1 

    audio_embs = []
    # split audio with window size
    for human_idx in range(HUMAN_NUMBER):   
        center_indices = torch.arange(
            audio_start_idx,
            audio_end_idx,
            1
        ).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1).to(full_audio_embs[human_idx].device)
        audio_emb = full_audio_embs[human_idx][center_indices][None,...] #.to(self.device)
        audio_embs.append(audio_emb)
    audio_embs = torch.concat(audio_embs, dim=0) #.to(self.param_dtype)

    # audio_cond = audio.to(device=x.device, dtype=x.dtype)
    audio_cond = audio_embs
    first_frame_audio_emb_s = audio_cond[:, :1, ...] 
    latter_frame_audio_emb = audio_cond[:, 1:, ...] 
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=vae_scale) 
    middle_index = audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
 
    return [first_frame_audio_emb_s, latter_frame_audio_emb_s]

def resize_and_centercrop(cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        
        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        
        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous() 
            # crop
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) 
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :] 

        return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t

def parse_speakers_locations(speakers_locations):
    bbox = {}
    if speakers_locations is None or len(speakers_locations) == 0:
        return None, ""
    speakers = speakers_locations.split(" ")
    if len(speakers) !=2:
        error= "Two speakers locations should be defined"
        return "", error
    
    for i, speaker in enumerate(speakers):
        location = speaker.strip().split(":")
        if len(location) not in (2,4):
            error = f"Invalid Speaker Location '{location}'. A Speaker Location should be defined in the format Left:Right or usuing a BBox Left:Top:Right:Bottom"
            return "", error
        try:
            good = False
            location_float = [ float(val) for val in location]
            good = all( 0 <= val <= 100 for val in location_float)
        except:
            pass
        if not good:
            error = f"Invalid Speaker Location '{location}'. Each number should be between 0 and 100."
            return "", error
        if len(location_float) == 2:
            location_float = [location_float[0], 0, location_float[1], 100]
        bbox[f"human{i}"] = location_float
    return bbox, ""


# construct human mask
def get_target_masks(HUMAN_NUMBER, lat_h, lat_w, src_h, src_w, face_scale = 0.05, bbox = None):
    human_masks = []
    if HUMAN_NUMBER==1:
        background_mask = torch.ones([src_h, src_w])
        human_mask1 = torch.ones([src_h, src_w])
        human_mask2 = torch.ones([src_h, src_w])
        human_masks = [human_mask1, human_mask2, background_mask]
    elif HUMAN_NUMBER==2:
        if bbox != None:
            assert len(bbox) == HUMAN_NUMBER, f"The number of target bbox should be the same with cond_audio"
            background_mask = torch.zeros([src_h, src_w])
            for _, person_bbox in bbox.items():
                y_min, x_min, y_max, x_max = person_bbox
                x_min, y_min, x_max, y_max = max(x_min,5), max(y_min, 5), min(x_max,95), min(y_max,95)                
                x_min, y_min, x_max, y_max =  int(src_h * x_min / 100), int(src_w * y_min / 100), int(src_h * x_max / 100), int(src_w * y_max / 100)
                human_mask = torch.zeros([src_h, src_w])
                human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                background_mask += human_mask
                human_masks.append(human_mask)
        else:
            x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
            background_mask = torch.zeros([src_h, src_w])
            background_mask = torch.zeros([src_h, src_w])
            human_mask1 = torch.zeros([src_h, src_w])
            human_mask2 = torch.zeros([src_h, src_w])
            lefty_min, lefty_max = int((src_w//2) * face_scale), int((src_w//2) * (1 - face_scale))
            righty_min, righty_max = int((src_w//2) * face_scale + (src_w//2)), int((src_w//2) * (1 - face_scale) + (src_w//2))
            human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
            human_mask2[x_min:x_max, righty_min:righty_max] = 1
            background_mask += human_mask1
            background_mask += human_mask2
            human_masks = [human_mask1, human_mask2]
        background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
        human_masks.append(background_mask)
    # toto = Image.fromarray(human_masks[2].mul_(255).unsqueeze(-1).repeat(1,1,3).to(torch.uint8).cpu().numpy())
    ref_target_masks = torch.stack(human_masks, dim=0) #.to(self.device)
    # resize and centercrop for ref_target_masks 
    # ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))
    N_h, N_w = lat_h // 2, lat_w // 2
    token_ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(N_h, N_w), mode='nearest').squeeze() 
    token_ref_target_masks = (token_ref_target_masks > 0) 
    token_ref_target_masks = token_ref_target_masks.float() #.to(self.device)

    token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 

    return token_ref_target_masks
