"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import torch
from diffusers import AutoencoderDC
import torchaudio
import torchvision.transforms as transforms
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from tqdm import tqdm
from mmgp import offload, transfer

try:
    from .music_vocoder import ADaMoSHiFiGANV1
except ImportError:
    from music_vocoder import ADaMoSHiFiGANV1


class MusicDCAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        source_sample_rate=None,
        dcae_weights_path=None,
        dcae_config_path=None,
        vocoder_weights_path=None,
        vocoder_config_path=None,
        device=None,
        dtype=torch.bfloat16,
    ):
        super(MusicDCAE, self).__init__()

        self._dtype = dtype

        self.dcae = offload.fast_load_transformers_model(
            dcae_weights_path,
            modelClass=AutoencoderDC,
            defaultConfigPath=dcae_config_path,
            writable_tensors=False,
            default_dtype=self._dtype,
            ignore_unused_weights=True,
        )
        self.dcae.eval()

        self.vocoder = offload.fast_load_transformers_model(
            vocoder_weights_path,
            modelClass=ADaMoSHiFiGANV1,
            defaultConfigPath=vocoder_config_path,
            writable_tensors=False,
            default_dtype=self._dtype,
            ignore_unused_weights=True,
        )
        self.vocoder.eval()
        self.vocoder._offload_hooks = ["decode"]

        if source_sample_rate is None:
            source_sample_rate = 48000

        self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)

        self.transform = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        return audio, sr

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        # audios: N x 2 x T, 48kHz
        device = audios.device
        dtype = audios.dtype

        if sr is None:
            sr = 48000
            resampler = self.resampler
        else:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(device).to(dtype)

        audio = resampler(audios)

        max_audio_len = audio.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audio = torch.nn.functional.pad(
                audio, (0, 8 * 512 - max_audio_len % (8 * 512))
            )

        mels = self.forward_mel(audio)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latent_lengths = (
            audio_lengths / sr * 44100 / 512 / self.time_dimention_multiple
        ).long()
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents, latent_lengths

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            mels = self.dcae.decoder(latent.unsqueeze(0))
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value

            # wav = self.vocoder.decode(mels[0]).squeeze(1)
            # decode waveform for each channels to reduce vram footprint
            wav_ch1 = self.vocoder.decode(mels[:,0,:,:]).squeeze(1).cpu()
            wav_ch2 = self.vocoder.decode(mels[:,1,:,:]).squeeze(1).cpu()
            wav = torch.cat([wav_ch1, wav_ch2],dim=0)

            if sr is not None:
                resampler = (
                    torchaudio.transforms.Resample(44100, sr)
                )
                wav = resampler(wav.cpu().float())
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [
                wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)
            ]
        return sr, pred_wavs

    @torch.no_grad()
    def decode_overlap(self, latents, audio_lengths=None, sr=None):
        """
        Decodes latents into waveforms using an overlapped DCAE and Vocoder.
        """
        print("Using Overlapped DCAE and Vocoder")

        device = latents.device
        MODEL_INTERNAL_SR = 44100
        DCAE_LATENT_TO_MEL_STRIDE = 8
        VOCODER_AUDIO_SAMPLES_PER_MEL_FRAME = 512

        pred_wavs = []
        final_output_sr = sr if sr is not None else MODEL_INTERNAL_SR

        # --- DCAE Parameters ---
        # dcae_win_len_latent: Window length in the latent domain for DCAE processing
        dcae_win_len_latent = 512 
        # dcae_mel_win_len: Expected mel window length from DCAE decoder output (latent_win * stride)
        dcae_mel_win_len = dcae_win_len_latent * 8
        # dcae_anchor_offset: Offset from anchor point to actual start of latent window slice
        dcae_anchor_offset = dcae_win_len_latent // 4
        # dcae_anchor_hop: Hop size for anchor points in latent domain
        dcae_anchor_hop = dcae_win_len_latent // 2
        # dcae_mel_overlap_len: Overlap length in the mel domain to be trimmed/blended
        dcae_mel_overlap_len = dcae_mel_win_len // 4

        # --- Vocoder Parameters ---
        # vocoder_win_len_audio: Audio samples per vocoder processing window
        vocoder_win_len_audio = 512 * 512 # Example: 262144 samples
        # vocoder_overlap_len_audio: Audio samples for overlap between vocoder windows
        vocoder_overlap_len_audio = 1024  
        # vocoder_hop_len_audio: Hop size in audio samples for vocoder processing
        vocoder_hop_len_audio = vocoder_win_len_audio - 2 * vocoder_overlap_len_audio
        # vocoder_input_mel_frames_per_block: Number of mel frames fed to vocoder in one go
        vocoder_input_mel_frames_per_block = vocoder_win_len_audio // VOCODER_AUDIO_SAMPLES_PER_MEL_FRAME
        
        crossfade_len_audio = 128 # Audio samples for crossfading vocoder outputs
        cf_win_tail = torch.linspace(1, 0, crossfade_len_audio, device=device).unsqueeze(0).unsqueeze(0)
        cf_win_head = torch.linspace(0, 1, crossfade_len_audio, device=device).unsqueeze(0).unsqueeze(0)

        for latent_idx, latent_item in enumerate(latents):
            latent_item = latent_item.to(device)
            current_latent = (latent_item / self.scale_factor + self.shift_factor).unsqueeze(0) # (1, C, H, W_latent)
            latent_len = current_latent.shape[3]

            # 1. DCAE: Latent to Mel Spectrogram (Overlapped)
            mels_segments = []
            if latent_len == 0:
                pass # No mel segments to generate
            else:
                # Determine anchor points for DCAE windows
                # An anchor marks a reference point for a window slice.
                # Window slice: current_latent[..., anchor - offset : anchor - offset + win_len]
                # First anchor ensures window starts at 0. Last anchor ensures tail is covered.
                dcae_anchors = list(range(dcae_anchor_offset, latent_len - dcae_anchor_offset, dcae_anchor_hop))
                if not dcae_anchors: # If latent is too short for the range, use one anchor
                    dcae_anchors = [dcae_anchor_offset]
                
                for i, anchor in enumerate(dcae_anchors):
                    win_start_idx = max(0, anchor - dcae_anchor_offset)
                    win_end_idx = min(latent_len, win_start_idx + dcae_win_len_latent)
                    
                    dcae_input_segment = current_latent[:, :, :, win_start_idx:win_end_idx]
                    if dcae_input_segment.shape[3] == 0: continue

                    mel_output_full = self.dcae.decoder(dcae_input_segment) # (1, C, H_mel, W_mel_fixed_from_dcae)

                    is_first = (i == 0)
                    is_last = (i == len(dcae_anchors) - 1)

                    if is_first and is_last: # Only one segment
                        # Use mel corresponding to actual input latent length
                        true_mel_content_len = dcae_input_segment.shape[3] * DCAE_LATENT_TO_MEL_STRIDE
                        mel_to_keep = mel_output_full[:, :, :, :min(true_mel_content_len, mel_output_full.shape[3])]
                    elif is_first: # First segment, trim end overlap
                        mel_to_keep = mel_output_full[:, :, :, :-dcae_mel_overlap_len]
                    elif is_last: # Last segment, trim start overlap
                        # And ensure we only take content relevant to the (potentially partial) last latent window
                        # The mel_output_full is fixed length. The useful part starts after overlap.
                        # The length of the useful part depends on how much of dcae_input_segment was actual content.
                        # For simplicity in overlap-add, typically trim fixed overlap.
                        # If dcae_input_segment was shorter than dcae_win_len_latent, mel_output_full might contain padding effects.
                        # Standard OLA keeps the corresponding tail.
                        mel_to_keep = mel_output_full[:, :, :, dcae_mel_overlap_len:]
                    else: # Middle segment, trim both overlaps
                        mel_to_keep = mel_output_full[:, :, :, dcae_mel_overlap_len:-dcae_mel_overlap_len]
                    
                    if mel_to_keep.shape[3] > 0:
                        mels_segments.append(mel_to_keep)
            
            if not mels_segments:
                num_mel_channels = current_latent.shape[1]
                mel_height = self.dcae.decoder_output_mel_height
                concatenated_mels = torch.empty(
                    (1, num_mel_channels, mel_height, 0),
                    device=current_latent.device, dtype=current_latent.dtype
                )
            else:
                concatenated_mels = torch.cat(mels_segments, dim=3)

            # Denormalize mels
            concatenated_mels = concatenated_mels * 0.5 + 0.5
            concatenated_mels = concatenated_mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            
            mel_total_frames = concatenated_mels.shape[3]

            # 2. Vocoder: Mel Spectrogram to Waveform (Overlapped)
            if mel_total_frames == 0:
                # Assuming mono or stereo output based on mel channels (typically mono for vocoder from single mel)
                num_audio_channels = 1 # Or determine from vocoder capabilities / mel channels
                final_wav = torch.zeros((num_audio_channels, 0), device=device, dtype=torch.float32)
            else:
                # Initial vocoder window
                # Vocoder expects (C_mel, H_mel, W_mel_block)
                mel_block = concatenated_mels[0, :, :, :vocoder_input_mel_frames_per_block].to(device)
                
                # Pad mel_block if it's shorter than vocoder_input_mel_frames_per_block (e.g. very short audio)
                if 0 < mel_block.shape[2] < vocoder_input_mel_frames_per_block:
                    pad_len = vocoder_input_mel_frames_per_block - mel_block.shape[2]
                    mel_block = torch.nn.functional.pad(mel_block, (0, pad_len), mode='constant', value=0) # Pad last dim
                
                current_audio_output = self.vocoder.decode(mel_block) # (C_audio, 1, Samples)
                current_audio_output = current_audio_output[:, :, :-vocoder_overlap_len_audio] # Remove end overlap

                # p_audio_samples tracks the start of the *next* audio segment to generate (in conceptual total audio samples)
                p_audio_samples = vocoder_hop_len_audio 
                conceptual_total_audio_len_native_sr = mel_total_frames * VOCODER_AUDIO_SAMPLES_PER_MEL_FRAME
                
                pbar_total = 1 + max(0, (conceptual_total_audio_len_native_sr - (vocoder_win_len_audio - vocoder_overlap_len_audio))) // vocoder_hop_len_audio
                
                # Use tqdm if you want a progress bar for the vocoder part
                # with tqdm(total=pbar_total, desc=f"Vocoder {latent_idx+1}/{len(latents)}", leave=False) as pbar:
                # pbar.update(1) # For initial window
                # The loop for subsequent windows
                while p_audio_samples < conceptual_total_audio_len_native_sr:
                    mel_frame_start = p_audio_samples // VOCODER_AUDIO_SAMPLES_PER_MEL_FRAME
                    mel_frame_end = mel_frame_start + vocoder_input_mel_frames_per_block
                    
                    if mel_frame_start >= mel_total_frames: break # No more mel frames

                    mel_block = concatenated_mels[0, :, :, mel_frame_start:min(mel_frame_end, mel_total_frames)].to(device)
                    
                    if mel_block.shape[2] == 0: break # Should not happen if mel_frame_start is valid

                    # Pad if current mel_block is too short (end of sequence)
                    if mel_block.shape[2] < vocoder_input_mel_frames_per_block:
                        pad_len = vocoder_input_mel_frames_per_block - mel_block.shape[2]
                        mel_block = torch.nn.functional.pad(mel_block, (0, pad_len), mode='constant', value=0)

                    new_audio_win = self.vocoder.decode(mel_block) # (C_audio, 1, Samples)

                    # Crossfade
                    # Determine actual crossfade length based on available audio
                    actual_cf_len = min(crossfade_len_audio, current_audio_output.shape[2], new_audio_win.shape[2] - (vocoder_overlap_len_audio - crossfade_len_audio))
                    if actual_cf_len > 0: # Ensure valid slice lengths for crossfade
                        tail_part = current_audio_output[:, :, -actual_cf_len:]
                        head_part = new_audio_win[:, :, vocoder_overlap_len_audio - actual_cf_len : vocoder_overlap_len_audio]
                        
                        crossfaded_segment = tail_part * cf_win_tail[:,:,:actual_cf_len] + \
                                             head_part * cf_win_head[:,:,:actual_cf_len]
                        
                        current_audio_output = torch.cat([current_audio_output[:, :, :-actual_cf_len], crossfaded_segment], dim=2)

                    # Append non-overlapping part of new_audio_win
                    is_final_append = (p_audio_samples + vocoder_hop_len_audio >= conceptual_total_audio_len_native_sr)
                    if is_final_append:
                        segment_to_append = new_audio_win[:, :, vocoder_overlap_len_audio:]
                    else:
                        segment_to_append = new_audio_win[:, :, vocoder_overlap_len_audio:-vocoder_overlap_len_audio]
                    
                    current_audio_output = torch.cat([current_audio_output, segment_to_append], dim=2)
                    
                    p_audio_samples += vocoder_hop_len_audio
                    # pbar.update(1) # if using tqdm
                
                final_wav = current_audio_output.squeeze(1) # (C_audio, Samples)

            # 3. Resampling (if necessary)
            if final_output_sr != MODEL_INTERNAL_SR and final_wav.numel() > 0:
                # Resample expects CPU tensor if using torchaudio.transforms on older versions or for some backends
                resampler = torchaudio.transforms.Resample(
                    MODEL_INTERNAL_SR, final_output_sr, dtype=final_wav.dtype
                )
                final_wav = resampler(final_wav.cpu()).to(device) # Move back to device if needed later
            
            pred_wavs.append(final_wav)

        # 4. Final Truncation
        processed_pred_wavs = []
        for i, wav in enumerate(pred_wavs):
            # Calculate expected length based on original latent, at the FINAL output sample rate
            _num_latent_frames = latents[i].shape[-1] # Use original latent item for shape
            _num_mel_frames = _num_latent_frames * DCAE_LATENT_TO_MEL_STRIDE
            _conceptual_native_audio_len = _num_mel_frames * VOCODER_AUDIO_SAMPLES_PER_MEL_FRAME
            max_possible_len = int(_conceptual_native_audio_len * final_output_sr / MODEL_INTERNAL_SR)

            current_wav_len = wav.shape[1]
            
            if audio_lengths is not None:
                # User-provided length is the primary target, capped by actual and max possible
                target_len = min(audio_lengths[i], current_wav_len, max_possible_len)
            else:
                # No user length, use max possible capped by actual
                target_len = min(max_possible_len, current_wav_len)
            
            processed_pred_wavs.append(transfer.move(wav[:, :max(0, target_len)], "cpu")) # Ensure length is non-negative

        return final_output_sr, processed_pred_wavs

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(
            audios=audios, audio_lengths=audio_lengths, sr=sr
        )
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths


if __name__ == "__main__":

    audio, sr = torchaudio.load("test.wav")
    audio_lengths = torch.tensor([audio.shape[1]])
    audios = audio.unsqueeze(0)

    # test encode only
    model = MusicDCAE()
    # latents, latent_lengths = model.encode(audios, audio_lengths)
    # print("latents shape: ", latents.shape)
    # print("latent_lengths: ", latent_lengths)

    # test encode and decode
    sr, pred_wavs, latents, latent_lengths = model(audios, audio_lengths, sr)
    print("reconstructed wavs: ", pred_wavs[0].shape)
    print("latents shape: ", latents.shape)
    print("latent_lengths: ", latent_lengths)
    print("sr: ", sr)
    torchaudio.save("test_reconstructed.wav", pred_wavs[0], sr)
    print("test_reconstructed.wav")
