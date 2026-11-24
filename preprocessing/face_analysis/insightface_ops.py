# Pre- and post-processing of the InsightFace python package v0.7.3 (https://github.com/deepinsight/insightface, MIT: LICENSE_INSIGHTFACE):
# utils/face_align.py, model_zoo/scrfd.py (anchor decoding, NMS, letterbox), model_zoo/landmark.py (crop transform).
# WanGP changes: numpy-only, vectorized point transforms, rotation-0-or-any crop matrix computed directly, scikit-image 0.26 estimate API.
import cv2
import numpy as np
from skimage import transform as trans

arcface_dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2) and (image_size % 112 == 0 or image_size % 128 == 0)
    ratio = image_size / (112.0 if image_size % 112 == 0 else 128.0)
    dst = arcface_dst * ratio
    if image_size % 112 != 0:
        dst[:, 0] += 8.0 * ratio
    if hasattr(trans.SimilarityTransform, "from_estimate"):
        return trans.SimilarityTransform.from_estimate(lmk, dst).params[0:2, :]
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]


def norm_crop(img, landmark, image_size=112):
    return cv2.warpAffine(img, estimate_norm(landmark, image_size), (image_size, image_size), borderValue=0.0)


def transform(data, center, output_size, scale, rotation):
    """The crop of `data` around `center`, scaled by `scale` and rotated by `rotation` degrees, and its 2x3 matrix."""
    rot = float(rotation) * np.pi / 180.0
    rotate = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) * scale
    M = np.hstack([rotate, (output_size / 2 - rotate @ np.asarray(center, dtype=np.float64))[:, None]])
    return cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0), M


def trans_points2d(pts, M):
    return (pts.astype(np.float32) @ M[:, :2].T + M[:, 2]).astype(np.float32)


def letterbox(img, input_size):
    """The image resized into the top-left of an input_size (w, h) black canvas, and its scale."""
    im_ratio, model_ratio = float(img.shape[0]) / img.shape[1], float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))
    return det_img, float(new_height) / img.shape[0]


def anchor_centers(height, width, stride, num_anchors=2):
    centers = (np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32) * stride).reshape((-1, 2))
    return np.stack([centers] * num_anchors, axis=1).reshape((-1, 2)) if num_anchors > 1 else centers


def distance2bbox(points, distance):
    return np.stack([points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1], points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]], axis=-1)


def distance2kps(points, distance):
    return np.stack([points[:, i % 2] + distance[:, i] for i in range(distance.shape[1])], axis=-1)


def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= thresh)[0] + 1]
    return keep


def decode_detections(levels, strides, det_scale, threshold, nms_thresh):
    """SCRFD.detect after the network: boxes (n, 5: x0, y0, x1, y1, score) and 5-point keypoints (n, 5, 2) in image pixels.
    `levels` holds per stride the scores, box distances, keypoint distances and feature size (h, w) of the letterboxed input."""
    scores_list, bboxes_list, kpss_list = [], [], []
    for (scores, bbox_preds, kps_preds, (height, width)), stride in zip(levels, strides):
        centers = anchor_centers(height, width, stride)
        pos_inds = np.where(scores >= threshold)[0]
        scores_list.append(scores[pos_inds])
        bboxes_list.append(distance2bbox(centers, bbox_preds * stride)[pos_inds])
        kpss_list.append(distance2kps(centers, kps_preds * stride).reshape((-1, 5, 2))[pos_inds])
    scores = np.vstack(scores_list)
    order = scores.ravel().argsort()[::-1]
    pre_det = np.hstack((np.vstack(bboxes_list) / det_scale, scores)).astype(np.float32, copy=False)[order, :]
    keep = nms(pre_det, nms_thresh)
    return pre_det[keep, :], (np.vstack(kpss_list) / det_scale)[order][keep]
