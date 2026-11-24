from __future__ import annotations

__all__ = ["ScailPoseProcessor"]

import numpy as np
import torch
from PIL import Image, ImageOps

from preprocessing.dwpose.native import pose_models
from shared.utils import files_locator as fl
from .nlf import load_multiperson_nlf_eager

from .scail_pose_align3d import solve_new_camera_params_central, solve_new_camera_params_down
from .scail_pose_dwpose import DWposeDetector
from .scail_pose_nlf import (
    collect_smpl_poses,
    intrinsic_matrix_from_field_of_view,
    process_data_to_COCO_format,
    process_video_multi_nlf,
    process_video_nlf,
    recollect_dwposes,
    recollect_nlf,
    render_multi_nlf_as_images,
    render_nlf_as_images,
    scale_faces,
    shift_dwpose_according_to_nlf,
)
from .scail_pose_multi import change_poses_to_limit_num, get_largest_bbox_indices


def _pil_resample(name: str):
    try:
        return getattr(Image.Resampling, name)
    except AttributeError:  # Pillow<9
        return getattr(Image, name)


_RESAMPLE_LANCZOS = _pil_resample("LANCZOS")
_RESAMPLE_NEAREST = _pil_resample("NEAREST")


def _tensor_frame_to_uint8_rgb(frame_chw: torch.Tensor) -> np.ndarray:
    frame = frame_chw.detach()
    if frame.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(frame.shape)}")

    if frame.shape[0] == 1:
        frame = frame.expand(3, -1, -1)
    if frame.shape[0] >= 4:
        frame = frame[:3]

    frame = frame.to(dtype=torch.float32)
    if frame.numel() == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    if float(frame.min()) < 0.0:
        frame = (frame + 1.0) * 127.5
    elif float(frame.max()) <= 1.0:
        frame = frame * 255.0

    frame = frame.clamp(0, 255).to(torch.uint8)
    return frame.permute(1, 2, 0).cpu().numpy()


def _tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    m = mask.detach()
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        raise ValueError(f"Expected HW (or 1HW) mask tensor, got shape {tuple(mask.shape)}")

    m = m.to(dtype=torch.float32)
    if m.numel() == 0:
        return Image.fromarray(np.zeros((0, 0), dtype=np.uint8), mode="L")

    if float(m.min()) < 0.0:
        m = (m + 1.0) * 0.5
    elif float(m.max()) > 1.0:
        m = m / 255.0

    m = m.clamp(0.0, 1.0)
    m8 = (m * 255.0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(m8, mode="L")


class ScailPoseProcessor:
    """
    WanGP wrapper around the official SCAIL-Pose preprocessing pipeline:
    - DWpose (YOLOX+RTMPose) for 2D keypoints + bbox.
    - NLFPose for 3D pose.
    - Taichi cylinder renderer + 2D overlay.
    """

    def __init__(self, gpu_id: int = 0, *, multi_person: bool = False, max_people: int = 2):
        if not torch.cuda.is_available():
            raise RuntimeError("SCAIL pose preprocessing requires CUDA (GPU-only).")

        self.gpu_id = int(gpu_id)
        self.multi_person = bool(multi_person)
        self.max_people = max(1, int(max_people))

        from preprocessing.dwpose.assets import query_download_def
        from shared.utils.download import process_files_def_if_needed

        process_files_def_if_needed(query_download_def())
        if self.multi_person:
            from preprocessing.matanyone.utils.model_assets import ensure_selected_matanyone_assets
            ensure_selected_matanyone_assets({"matanyone_version": "v1"})

        # Some upstream helpers use bare `.cuda()` / default device semantics.
        torch.cuda.set_device(self.gpu_id)
        self.detector = DWposeDetector(use_batch=False).to(self.gpu_id)

        eager_ckpt = fl.locate_file("pose/nlf_l_multi_0.3.2.eager.safetensors")
        self.model_nlf = load_multiperson_nlf_eager(
            checkpoint_path=eager_ckpt,
            device=f'cuda:{self.gpu_id}'
        )

        # Lazy-loaded for multi-person mode.
        self._sam_segmenter = None
        self._matanyone_model = None
        self._matanyone_version = None

    def unload(self):
        """Release NLF model from memory (RAM and VRAM)."""
        import gc

        if hasattr(self, 'model_nlf') and self.model_nlf is not None:
            # Move model to CPU first to free VRAM
            self.model_nlf.to('cpu')
            # Delete the model
            del self.model_nlf
            self.model_nlf = None

        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.to('cpu')
            del self.detector
            self.detector = None

        if getattr(self, "_sam_segmenter", None) is not None:
            try:
                self._sam_segmenter.model.to("cpu")
            except Exception:
                pass
            self._sam_segmenter = None

        if getattr(self, "_matanyone_model", None) is not None:
            try:
                self._matanyone_model.to("cpu")
            except Exception:
                pass
            self._matanyone_model = None
            self._matanyone_version = None

        # Force garbage collection and empty CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_sam_loaded(self) -> None:
        if self._sam_segmenter is not None:
            return
        from preprocessing.matanyone.tools.base_segmenter import BaseSegmenter

        # `BaseSegmenter` loads weights from `ckpts/mask/*safetensors` via `files_locator`.
        self._sam_segmenter = BaseSegmenter(SAM_checkpoint=None, model_type="vit_h", device=f"cuda:{self.gpu_id}")

    def _ensure_matanyone_loaded(self) -> None:
        from preprocessing.matanyone.utils.model_assets import get_selected_matanyone_version, load_selected_matanyone_model

        scail_matanyone_config = {"matanyone_version": "v1"}
        selected_version = get_selected_matanyone_version(scail_matanyone_config)
        if self._matanyone_model is not None and self._matanyone_version == selected_version:
            return
        if self._matanyone_model is not None:
            self._matanyone_model.to("cpu")
            self._matanyone_model = None
            torch.cuda.empty_cache()

        self._matanyone_model, self._matanyone_version, _ = load_selected_matanyone_model(scail_matanyone_config)
        self._matanyone_model = self._matanyone_model.eval()
        self._matanyone_model.to(f"cuda:{self.gpu_id}")

    def _extract_and_render_multi(
        self,
        vr_frames_np: list[np.ndarray],
        poses_list,
        det_results_list,
    ) -> torch.Tensor:
        if not vr_frames_np:
            return torch.empty((0,), dtype=torch.float32)

        first_bboxes = det_results_list[0] if det_results_list else None
        if not first_bboxes:
            return torch.empty((0,), dtype=torch.float32)

        indices = get_largest_bbox_indices(first_bboxes, self.max_people)
        if not indices:
            return torch.empty((0,), dtype=torch.float32)

        height, width = vr_frames_np[0].shape[:2]
        pose0 = poses_list[0]

        # 1) Build SAM prompts from DWpose (bbox + a few stable keypoints).
        considered_points = {0, 1, 14, 15}
        box_list_px: list[np.ndarray] = []
        points_list_px: list[np.ndarray] = []
        for idx in indices:
            x1, y1, x2, y2 = first_bboxes[idx]
            box_px = np.array([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)

            candidate = np.asarray(pose0["bodies"]["candidate"][idx], dtype=np.float32)
            subset = np.asarray(pose0["bodies"]["subset"][idx])
            subset_mod = subset.copy()
            for k in range(len(subset_mod)):
                if k not in considered_points:
                    subset_mod[k] = -1

            pts = candidate[subset_mod != -1]
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            pts = pts[(pts[:, 0] >= 0.0) & (pts[:, 1] >= 0.0)]
            if pts.shape[0] == 0:
                pts = np.array([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float32)

            pts_px = pts.copy()
            pts_px[:, 0] *= width
            pts_px[:, 1] *= height

            box_list_px.append(box_px)
            points_list_px.append(pts_px)

        # 2) Initial per-person masks on the first frame (SAM).
        self._ensure_sam_loaded()
        self._sam_segmenter.reset_image()
        self._sam_segmenter.set_image(vr_frames_np[0])

        init_masks: list[np.ndarray] = []
        for box_px, pts_px in zip(box_list_px, points_list_px):
            labels = np.ones((pts_px.shape[0],), dtype=np.int32)
            masks, scores, _logits = self._sam_segmenter.predictor.predict(
                point_coords=pts_px,
                point_labels=labels,
                box=box_px,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                init_masks.append(np.zeros((height, width), dtype=np.uint8))
                continue
            best = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
            init_masks.append(masks[best].astype(np.uint8) * 255)

        # Unload SAM to free VRAM before MatAnyone
        if self._sam_segmenter is not None:
            self._sam_segmenter.reset_image()
            self._sam_segmenter.model.to("cpu")
            del self._sam_segmenter
            self._sam_segmenter = None
            torch.cuda.empty_cache()

        # 3) Track/segment each person across the video (MatAnyone).
        self._ensure_matanyone_loaded()
        from preprocessing.matanyone.matanyone.inference.inference_core import InferenceCore
        from preprocessing.matanyone.matanyone_wrapper import matanyone as matanyone_run

        vr_frames_list = []
        for mask0 in init_masks:
            processor = InferenceCore(self._matanyone_model, cfg=self._matanyone_model.cfg)
            masked_frames, _alpha = matanyone_run(processor, vr_frames_np, mask0)
            vr_frames_list.append(torch.from_numpy(np.stack(masked_frames, axis=0)))
            del processor

        # Unload MatAnyone to free VRAM before NLF
        if self._matanyone_model is not None:
            self._matanyone_model.to("cpu")
            del self._matanyone_model
            self._matanyone_model = None
            self._matanyone_version = None
            torch.cuda.empty_cache()

        # 4) Run NLF per segmented person-video.
        nlf_results = process_video_multi_nlf(self.model_nlf, vr_frames_list)

        # 5) Limit DWpose overlays to the same number of persons for visualization.
        poses_vis, det_vis = change_poses_to_limit_num(list(poses_list), list(det_results_list), num_bboxes=len(indices))
        frames_np_rgba = render_multi_nlf_as_images(nlf_results, poses_vis, reshape_pool=None, intrinsic_matrix=None)

        frames_rgb = np.stack([f[:, :, :3] for f in frames_np_rgba], axis=0).astype(np.float32)
        frames_t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)  # T,C,H,W
        frames_t = frames_t / 127.5 - 1.0
        return frames_t.permute(1, 0, 2, 3).contiguous()

    def extract_and_render(self, *args, **kwargs) -> torch.Tensor:
        with pose_models():
            return self._extract_and_render(*args, **kwargs)

    def _extract_and_render(
        self,
        video_frames: torch.Tensor,
        ref_image: Image.Image,
        mask_frames: torch.Tensor | None = None,
        align_pose: bool = True,
    ) -> torch.Tensor:
        torch.cuda.set_device(self.gpu_id)

        if video_frames is None or not isinstance(video_frames, torch.Tensor):
            raise TypeError("video_frames must be a torch.Tensor")
        if video_frames.dim() != 4:
            raise ValueError(f"Expected video tensor (C,T,H,W), got {tuple(video_frames.shape)}")

        if ref_image is None or not isinstance(ref_image, Image.Image):
            raise TypeError("ref_image must be a PIL.Image")

        out_w, out_h = ref_image.size
        target_size = (int(out_w), int(out_h))

        c, t, h, w = video_frames.shape
        if t == 0:
            return torch.empty((0,), dtype=torch.float32)

        pil_frames: list[Image.Image] = []
        vr_frames_np: list[np.ndarray] = []

        for frame_idx in range(t):
            frame_np = _tensor_frame_to_uint8_rgb(video_frames[:, frame_idx])
            frame_pil = Image.fromarray(frame_np, mode="RGB")

            # Output dimensions must match the ref image; also ensures internal pose extraction
            # resolution is <= output resolution.
            frame_pil = ImageOps.fit(frame_pil, target_size, method=_RESAMPLE_LANCZOS, centering=(0.5, 0.5))

            if mask_frames is not None:
                mask_pil = _tensor_mask_to_pil(mask_frames[:, frame_idx] if mask_frames.dim() == 4 else mask_frames[frame_idx])
                mask_pil = ImageOps.fit(mask_pil, target_size, method=_RESAMPLE_NEAREST, centering=(0.5, 0.5))
                mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
                rgb_arr = np.array(frame_pil, dtype=np.float32)
                rgb_arr = (rgb_arr * mask_arr[:, :, None]).clip(0, 255).astype(np.uint8)
                frame_pil = Image.fromarray(rgb_arr, mode="RGB")

            pil_frames.append(frame_pil)
            vr_frames_np.append(np.array(frame_pil, dtype=np.uint8))

        detector_return_list = [self.detector(pil_frame) for pil_frame in pil_frames]
        poses, _scores, det_results = zip(*detector_return_list)

        if self.multi_person:
            return self._extract_and_render_multi(vr_frames_np, list(poses), list(det_results))

        vr_frames = torch.from_numpy(np.stack(vr_frames_np, axis=0))  # (T,H,W,3) uint8
        nlf_results = process_video_nlf(self.model_nlf, vr_frames, det_results)

        # If nothing was detected across the whole clip, return empty tensor (caller handles it).
        first_pose_idx = None
        for i, item in enumerate(nlf_results):
            if len(item.get("nlfpose", [])) == 0:
                continue
            first = item["nlfpose"][0]
            if first is None:
                continue
            try:
                has_any = len(first) > 0
            except TypeError:
                has_any = False
            if has_any:
                first_pose_idx = i
                break
        if first_pose_idx is None:
            return torch.empty((0,), dtype=torch.float32)

        target_H, target_W = out_h, out_w
        ori_camera_pose = intrinsic_matrix_from_field_of_view([target_H, target_W])
        ori_focal = ori_camera_pose[0, 0]

        if align_pose:
            # Reference image pose + NLF (single frame)
            ref_rgb = ref_image.convert("RGB")
            ref_rgb = ImageOps.fit(ref_rgb, target_size, method=_RESAMPLE_LANCZOS, centering=(0.5, 0.5))
            pose_ref, _score_ref, det_result_ref = self.detector(ref_rgb)
            if det_result_ref is None or len(det_result_ref) == 0:
                align_pose = False
            else:
                vr_ref = torch.from_numpy(np.array(ref_rgb, dtype=np.uint8)).unsqueeze(0)
                nlf_results_ref = process_video_nlf(self.model_nlf, vr_ref, [det_result_ref])

                pose_3d_first_driving_frame = nlf_results[first_pose_idx]["nlfpose"][0][0].cpu().numpy()
                pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)

                poses_2d_ref = pose_ref["bodies"]["candidate"][0][:14]
                poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * target_W
                poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * target_H

                poses_2d_subset = pose_ref["bodies"]["subset"][0][:14]
                pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

                valid_upper_indices = []
                valid_lower_indices = []
                upper_body_indices = [0, 2, 3, 5, 6]
                lower_body_indices = [9, 10, 12, 13]
                for j in range(len(poses_2d_subset)):
                    if poses_2d_subset[j] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[j]) != 0:
                        if j in upper_body_indices:
                            valid_upper_indices.append(j)
                        if j in lower_body_indices:
                            valid_lower_indices.append(j)

                if len(valid_lower_indices) >= 4:
                    valid_indices = [1] + valid_lower_indices
                    new_camera_intrinsics, scale_m = solve_new_camera_params_down(
                        pose_3d_coco_first_driving_frame[valid_indices], ori_focal, [target_H, target_W], poses_2d_ref[valid_indices]
                    )
                else:
                    valid_indices = [1] + valid_upper_indices
                    new_camera_intrinsics, scale_m = solve_new_camera_params_central(
                        pose_3d_coco_first_driving_frame[valid_indices], ori_focal, [target_H, target_W], poses_2d_ref[valid_indices]
                    )

                poses_list = list(poses)
                _ = scale_faces(poses_list, [pose_ref])
                nlf_results = recollect_nlf(nlf_results)
                poses_list = recollect_dwposes(poses_list)
                shift_dwpose_according_to_nlf(
                    collect_smpl_poses(nlf_results), poses_list, ori_camera_pose, new_camera_intrinsics, target_H, target_W
                )
                frames_np_rgba = render_nlf_as_images(nlf_results, poses_list, reshape_pool=None, intrinsic_matrix=new_camera_intrinsics)

                frames_rgb = np.stack([f[:, :, :3] for f in frames_np_rgba], axis=0).astype(np.float32)
                frames_t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)  # T,C,H,W
                frames_t = frames_t / 127.5 - 1.0
                return frames_t.permute(1, 0, 2, 3).contiguous()

        # Non-aligned path (official fallback)
        nlf_results = recollect_nlf(nlf_results)
        frames_np_rgba = render_nlf_as_images(nlf_results, list(poses), reshape_pool=None, intrinsic_matrix=ori_camera_pose)
        frames_rgb = np.stack([f[:, :, :3] for f in frames_np_rgba], axis=0).astype(np.float32)
        frames_t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)
        frames_t = frames_t / 127.5 - 1.0
        return frames_t.permute(1, 0, 2, 3).contiguous()
