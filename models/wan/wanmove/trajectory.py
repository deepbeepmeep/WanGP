import os
import sys
sys.path.append(os.path.abspath('.'))
import math
import random
from io import BytesIO

import imageio.v3 as iio
import numpy as np

import torch
from PIL import Image, ImageDraw
from torchvision import transforms



SKIP_ZERO = False

def get_pos_emb(
    pos_k: torch.Tensor,
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate batch position embeddings.
    
    Args:
        pos_k (torch.Tensor): A 1D tensor containing positions for which to generate embeddings.
        pos_emb_dim (int): The dimension of position embeddings.
        theta_func (callable): Function to compute thetas based on position and embedding dimensions.
        device (torch.device): Device to store the position embeddings.
        dtype (torch.dtype): Desired data type for computations.
    
    Returns:
        torch.Tensor: The position embeddings with shape (batch_size, pos_emb_dim).
    """
    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device, dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
    # Expand denominator to match the shape needed for broadcasting
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)
    
    thetas = theta_func(denominator_expanded, pos_emb_dim)
    
    # Ensure pos_k is in the correct shape for broadcasting
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

    # Concatenate sine and cosine embeddings along the last dimension
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb

def create_pos_feature_map(
    pred_tracks: torch.Tensor, # [T, N, 2]
    pred_visibility: torch.Tensor, # [T, N]
    downsample_ratios: list[int],
    height: int,
    width: int,
    pos_emb_dim: int,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Create a feature map from the predicted tracks.

    Args:
    - pred_tracks: torch.Tensor, the predicted tracks, [T, N, 2]
    - pred_visibility: torch.Tensor, the predicted visibility, [T, N]
    - downsample_ratios: list[int], the ratios for downsampling time, height, and width
    - height: int, the height of the feature map
    - width: int, the width of the feature map
    - pos_emb_dim: int, the dimension of the position embeddings
    - track_num: int, the number of tracks to use
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - feature_map: torch.Tensor, the feature map, [T', H', W', pos_emb_dim]
    - track_pos: torch.Tensor, the position embeddings, [N, T', 2], 2 = height, width
    """
    
    assert t_down_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."

    t, n, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    feature_map = torch.zeros((t-1) // t_down + 1, height // h_down, width // w_down, pos_emb_dim, device=device, dtype=dtype)
    track_pos = - torch.ones(n, (t-1) // t_down + 1, 2, dtype=torch.long, device=device)
    
    if track_num == -1:
        track_num = n

    tracks_idx = torch.randperm(n, device=device)[:track_num]
    tracks = pred_tracks[:, tracks_idx]
    visibility = pred_visibility[:, tracks_idx] if pred_visibility is not None else None
    tracks_embs = get_pos_emb(torch.randperm(n, device=device)[:track_num], pos_emb_dim, device=device, dtype=dtype)

    cover_count = 0
    if visibility is None:
        cur_visibility = [True] * track_num

    for t_idx in range(0, t, t_down):
        if t_down_strategy == "sample" or t_idx == 0:
            cur_tracks = tracks[t_idx] # [N, 2]
            if visibility is not None:
                cur_visibility = visibility[t_idx] # [N]
        else:
            cur_tracks = tracks[t_idx:t_idx+t_down].mean(dim=0)
            cur_visibility = torch.any(visibility[t_idx:t_idx+t_down], dim=0)
        
        for i in range(track_num):
            if not cur_visibility[i] or cur_tracks[i][0] < 0 or cur_tracks[i][1] < 0 or cur_tracks[i][0] >= width or cur_tracks[i][1] >= height:
                continue
            x, y = cur_tracks[i]
            x, y = int(x // w_down), int(y // h_down)
            feature_map[t_idx // t_down, y, x] += tracks_embs[i]
            track_pos[i, t_idx // t_down, 0], track_pos[i, t_idx // t_down, 1] = y, x

    return feature_map, track_pos


def replace_feature(
    vae_feature: torch.Tensor,  # [B, C', T', H', W']
    track_pos: torch.Tensor,    # [B, N, T', 2]
) -> torch.Tensor:
    b, _, t, h, w = vae_feature.shape
    assert b == track_pos.shape[0], "Batch size mismatch."
    n = track_pos.shape[1]
    
    # Shuffle the trajectory order
    track_pos = track_pos[:, torch.randperm(n, device=track_pos.device), :, :]
    
    # Extract coordinates at time steps ≥ 1 and generate a valid mask
    current_pos = track_pos[:, :, 1:, :]  # [B, N, T-1, 2]
    mask = (current_pos[..., 0] >= 0) & (current_pos[..., 1] >= 0)  # [B, N, T-1]
    
    # Get all valid indices
    valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 3]
    num_valid = valid_indices.shape[0]
    
    if num_valid == 0:
        return vae_feature
    
    # Decompose valid indices into each dimension
    batch_idx = valid_indices[:, 0]
    track_idx = valid_indices[:, 1]
    t_rel = valid_indices[:, 2]
    t_target = t_rel + 1  # Convert to original time step indices
    
    # Extract target position coordinates
    h_target = current_pos[batch_idx, track_idx, t_rel, 0].long()  # Ensure integer indices
    w_target = current_pos[batch_idx, track_idx, t_rel, 1].long()
    
    # Extract source position coordinates (t=0)
    h_source = track_pos[batch_idx, track_idx, 0, 0].long()
    w_source = track_pos[batch_idx, track_idx, 0, 1].long()
    
    # Get source features and assign to target positions
    src_features = vae_feature[batch_idx, :, 0, h_source, w_source]
    vae_feature[batch_idx, :, t_target, h_target, w_target] = src_features
    
    return vae_feature

def get_video_track_video(
    model,
    video_tensor: torch.Tensor, # [T, C, H, W]
    downsample_ratios: list[int],
    pos_emb_dim: int,
    grid_size: int = 32,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the track video from the video tensor.

    Args:
    - model: torch.nn.Module, the model for tracking, CoTracker
    - video_tensor: torch.Tensor, the video tensor, [T, C, H, W]
    - downsample_ratios: list[int], the ratios for downsampling time, height, and width
    - height: int, the height of the feature map
    - width: int, the width of the feature map
    - pos_emb_dim: int, the dimension of the position embeddings
    - grid_size: int, the size of the grid
    - track_num: int, the number of tracks to use
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    -  track_video: torch.Tensor, the track video, [pos_emb_dim, T', H', W']
    -  track_pos: torch.Tensor, the position embeddings, [N, T', 2], 2 = height, width
    -  pred_tracks: the predicted point trajectories
    -  pred_visibility: visibility of the predicted point trajectories
    """

    t, c, height, width = video_tensor.shape
    with (
        torch.autocast(device_type=device.type, dtype=dtype),
        torch.no_grad(),
    ):
        pred_tracks, pred_visibility = model(
            video_tensor.unsqueeze(0),
            grid_size=grid_size,
            backward_tracking=False,
        ) 
    
    track_video, track_pos = create_pos_feature_map(
        pred_tracks[0], pred_visibility[0], downsample_ratios, height, width, pos_emb_dim, track_num, t_down_strategy, device, dtype
    )

    return track_video.permute(3, 0, 1, 2), track_pos, pred_tracks, pred_visibility


# === user input tracks ===

def resize_tracks(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    t_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize tracks to a specified number of frames.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - t_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - resized_tracks: torch.Tensor, the resized tracks, [target_frame_num, N, 2]
    - resized_visibility: torch.Tensor, the resized visibility, [target_frame_num, N]
    """

    assert t_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."
    assert t_strategy in ["sample"], "only support sample strategy."

    def get_xy_from_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the x and y coordinates from the height and width tensor.

        Args:
        - hw_tensor: torch.Tensor, the tensor of height and width, [N, height, width]

        Returns:
        - xy_tensor: torch.Tensor, the tensor of x and y coordinates, [N, 2]
        """

        h, w = hw_tensor.shape[-2:]
        _, y, x = torch.nonzero(hw_tensor, as_tuple=True)
        xy_tensor = torch.stack((x, y), dim=-1)
        assert xy_tensor.shape[0] == hw_tensor.shape[0], "The number of points should be the same."
        return xy_tensor

    def get_average_xy_from_batch_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        for b in range(hw_tensor.shape[0]):
            xy_tensor = get_xy_from_hw(hw_tensor[b])
            if b == 0:
                xy_tensors = xy_tensor
            else:
                xy_tensors += xy_tensor
        xy_tensors /= hw_tensor.shape[0]
        return xy_tensors
    
    # Get the number of frames in the input tracks
    num_frames, num_tracks, _, _ = img_tracks.shape

    new_tracks = torch.zeros(target_frame_num, num_tracks, 2, device=device, dtype=dtype)
    new_visibility = torch.ones(target_frame_num, num_tracks, device=device, dtype=torch.bool)

    new_tracks[0] = get_xy_from_hw(img_tracks[0])
    # -1 for removing the first frame
    num_frames -= 1
    target_frame_num -= 1

    new_frame_idx = 1
    if target_frame_num <= num_frames:
        t_down = num_frames / target_frame_num
        frame_idxs = [int((i - 1) * t_down + 1) for i in range(1, target_frame_num + 1)]
        for i, frame_idx in enumerate(frame_idxs):
            if t_strategy == "sample":
                new_tracks[new_frame_idx] = get_xy_from_hw(img_tracks[frame_idx])
            else:
                next_frame_idx = frame_idxs[i + 1] if i + 1 < len(frame_idxs) else num_frames + 1 # +1 as compensation for the -1
                new_tracks[new_frame_idx] = get_average_xy_from_batch_hw(img_tracks[frame_idx:next_frame_idx])

            new_frame_idx += 1
    else:
        t_repeat = target_frame_num / num_frames
        target_frame_idxs = [int((i - 1) * t_repeat + 1) for i in range(1, num_frames + 1)]
        for i, target_frame_idx in enumerate(target_frame_idxs):
            next_target_frame_idx = target_frame_idxs[i + 1] if i + 1 < len(target_frame_idxs) else target_frame_num + 1
            if t_strategy == "sample":
                new_tracks[target_frame_idx:next_target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
            else:
                if target_frame_idx == next_target_frame_idx:
                    new_tracks[target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
                else:
                    next_new_frame_idx = new_frame_idx + 1 if new_frame_idx + 1 < num_frames else new_frame_idx
                    for j in range(target_frame_idx, next_target_frame_idx):
                        new_tracks[j] = (1 - (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx)) * get_xy_from_hw(img_tracks[new_frame_idx]) + (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx) * get_xy_from_hw(img_tracks[next_new_frame_idx])

            new_frame_idx += 1

    # print(new_tracks[1])
    return new_tracks, new_visibility

def generate_custom_feature_map(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    downsample_ratios: list[int],
    pos_emb_dim: int,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a custom feature map from the tracks.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - downsample_ratios: List[int], the ratios for downsampling time, height, and width
    - pos_emb_dim: int, the dimension of the position embeddings
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - feature_map: torch.Tensor, the feature map, [T', H', W', pos_emb_dim]
    """

    height, width = img_tracks.shape[-2:]
    resized_tracks, resized_visibility = resize_tracks(img_tracks, target_frame_num, t_down_strategy, device, dtype)
    feature_map, track_pos = create_pos_feature_map(
        resized_tracks, 
        resized_visibility, 
        downsample_ratios, 
        height, 
        width, 
        pos_emb_dim, 
        track_num=-1, 
        t_down_strategy=t_down_strategy, 
        device=device, 
        dtype=dtype
    )

    return feature_map, track_pos



# ---------------------------
# Visualize functions
# --------------------------


def draw_overall_gradient_polyline_on_image(image, line_width, points, start_color):
    """
    - image (Image): target image to draw on.
    - line_width (int): initial line width.
    - points (list of tuples): list of points forming the polyline, each point is (x, y).
    - start_color (tuple): starting color of the line (R, G, B).

    Return:
    - Image: original image with the gradient polyline drawn.
    """
    
    def get_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    # Create a new image with the same size as the original
    new_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(new_image, 'RGBA')
    points = points[::-1]

    # Compute total length
    total_length = sum(get_distance(points[i], points[i+1]) for i in range(len(points)-1))

    # Accumulated length
    accumulated_length = 0

    # Draw the gradient polyline
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = get_distance(start_point, end_point)
        steps = int(segment_length)

        for i in range(steps):
            # Current accumulated length
            current_length = accumulated_length + (i / steps) * segment_length

            # Alpha from fully opaque to fully transparent
            alpha = int(255 * (1 - current_length / total_length))
            color = (*start_color, alpha)

            # Interpolated coordinates
            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            # Dynamic line width, decreasing from initial width to 1
            dynamic_line_width = int(line_width * (1 - (current_length / total_length)))
            dynamic_line_width = max(dynamic_line_width, 1)  # minimum width is 1 to avoid 0

            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length

    return new_image
   
def add_weighted(rgb, track):
    rgb = np.array(rgb) # [H, W, C] "RGB"
    track = np.array(track) # [H, W, C] "RGBA"
    
    # Compute weights from the alpha channel
    alpha = track[:, :, 3] / 255.0

    # Expand alpha to 3 channels to match RGB
    alpha = np.stack([alpha] * 3, axis=-1)

    # Blend the two images
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)
    
    return Image.fromarray(blend_img.astype(np.uint8))
        
def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24):
    color_map = [
        (102, 153, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 102, 204),
        (0, 255, 0)
    ]
    circle_size = 12
    line_width = 16
    
    video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy() # (81, 480, 832, 3), uint8
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()
    # print(video.shape, tracks.shape)
    
    output_frames = []
    # Process the video
    for t in range(video.shape[0]):
        # Extract current frame
        frame = video[t]
        frame = Image.fromarray(frame).convert("RGB")
        
        # Draw tracks
        for n in range(tracks.shape[1]):
            if visibility is not None and visibility[t, n] == 0:
                continue
            
            # Track coordinate at current frame
            track_coord = tracks[t, n]
            tracks_coord = tracks[max(t-track_frame, 0):t+1, n]
            
            # Draw a circle
            draw = ImageDraw.Draw(frame)
            draw.ellipse((track_coord[0] - circle_size, track_coord[1] - circle_size, track_coord[0] + circle_size, track_coord[1] + circle_size), fill=color_map[n % len(color_map)])
            # Draw the polyline
            track_image = draw_overall_gradient_polyline_on_image(frame, line_width, tracks_coord, color_map[n % len(color_map)])
            frame = add_weighted(frame, track_image)
        
        # Save current frame
        output_frames.append(frame.convert("RGB"))
        
    return output_frames
