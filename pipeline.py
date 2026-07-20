"""
Stages
1. Detection + segmentation (YOLO11-seg)
2. Pose (YOLO11-pose): body keypoints
3. Head: derived from the face keypoints + crown.
4. Depth (Depth Anything V2)
5. Scale

Height basis, in priority order:
  full body (feet visible)  ->  crown-to-feet
  else head visible         ->  head-length x anthropometric ratio
  else                      ->  extrapolate from the lowest visible landmark
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO class ids used by the YOLO models.
COCO_PERSON = 0
COCO_CELL_PHONE = 67

# Mirror selfies are often tall and the phone is small; higher-res inference with
# a lower confidence gate is what makes the phone reliably detectable.
DETECT_IMGSZ = 1280
DETECT_CONF = 0.15

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# COCO-17 keypoint indices (yolo pose order).
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

# Total standing height as a multiple of head length (crown->chin). ~7.5 for
# adults. Used for the head-based estimate.
HEAD_TO_STATURE = 7.5
# The sole sits below the ankle joint by roughly this fraction of the shin.
FOOT_OFFSET_SHIN = 0.33

# Fraction of stature from the crown down to a landmark (feet-out-of-frame case).
CROWN_TO_LANDMARK_FRACTION = {"shoulder": 0.18, "hip": 0.52, "knee": 0.72}
LANDMARK_UNCERTAINTY_CM = {"shoulder": 9.0, "hip": 6.0, "knee": 4.0}

_seg_model = None
_pose_model = None
_depth_pipe = None


def _load_models():
    global _seg_model, _pose_model, _depth_pipe
    if _seg_model is None:
        from ultralytics import YOLO
        _seg_model = YOLO("yolo11n-seg.pt")
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO("yolo11n-pose.pt")
    if _depth_pipe is None:
        from transformers import pipeline
        _depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=-1)
    return _seg_model, _pose_model, _depth_pipe


@dataclass
class Result:
    ok: bool
    message: str = ""
    height_cm: float = 0.0
    naive_height_cm: float = 0.0
    uncertainty_cm: float = 0.0
    basis: str = ""
    phone_source: str = ""
    phone_width_px: float = 0.0
    depth_factor: float = 1.0
    phone_depth_applied: bool = False
    annotated: Image.Image = None
    depth_map: Image.Image = None
    notes: list = field(default_factory=list)


def _to_numpy_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _pick_person(seg_result, H, W):
    """Return (mask_bool, box) for the largest person, or None."""
    if seg_result.masks is None:
        return None
    boxes = seg_result.boxes
    masks = seg_result.masks.data.cpu().numpy()
    best = None
    for i in range(len(boxes)):
        if int(boxes.cls[i]) != COCO_PERSON:
            continue
        area = masks[i].sum()
        if best is None or area > best[0]:
            best = (area, i)
    if best is None:
        return None
    i = best[1]
    box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
    return _resize_mask(masks[i], (H, W)), box


def _pick_phone(seg_result, H, W):
    """Return (box, mask_bool, conf) for the most confident phone, or None."""
    boxes = seg_result.boxes
    masks = seg_result.masks.data.cpu().numpy() if seg_result.masks is not None else None
    best = None
    for i in range(len(boxes)):
        if int(boxes.cls[i]) != COCO_CELL_PHONE:
            continue
        conf = float(boxes.conf[i])
        if best is None or conf > best[2]:
            m = _resize_mask(masks[i], (H, W)) if masks is not None else None
            best = (tuple(boxes.xyxy[i].cpu().numpy().tolist()), m, conf)
    return best


def _resize_mask(mask, target_hw):
    h, w = target_hw
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _mask_vertical_extent(mask_bool):
    ys = np.where(mask_bool.any(axis=1))[0]
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max())


def _kp(kpts, name, conf_th=0.4):
    """Return (x, y) for a keypoint above threshold, else None."""
    x, y, c = kpts[KP[name]]
    return (float(x), float(y)) if c >= conf_th else None


def _kp_pair(kpts, left, right, conf_th=0.4):
    """Average of a left/right keypoint pair (whichever pass), or None."""
    pts = [p for p in (_kp(kpts, left, conf_th), _kp(kpts, right, conf_th)) if p]
    if not pts:
        return None
    return (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _select_best_pose(pose_res):
    """
    Return a single robust skeleton for the subject.
    """
    if pose_res.keypoints is None or len(pose_res.keypoints) == 0:
        return None
    kdata = pose_res.keypoints.data.cpu().numpy()  # (N,17,3)
    boxes = pose_res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    anchor = int(areas.argmax())
    merged = kdata[anchor].copy()
    for i in range(len(kdata)):
        if i == anchor or _box_iou(boxes[i], boxes[anchor]) > 0.35:
            better = kdata[i][:, 2] > merged[:, 2]
            merged[better] = kdata[i][better]
    return merged


def analyze(pil_img, phone_height_mm, phone_width_mm, phone_source):
    seg_model, pose_model, depth_pipe = _load_models()

    img_rgb = pil_img.convert("RGB")
    bgr = _to_numpy_bgr(img_rgb)
    H, W = bgr.shape[:2]

    seg_res = seg_model(bgr, imgsz=DETECT_IMGSZ, conf=DETECT_CONF, verbose=False)[0]
    person = _pick_person(seg_res, H, W)
    phone = _pick_phone(seg_res, H, W)

    if person is None:
        return Result(ok=False, message="No person detected in the photo.")
    if phone is None:
        return Result(ok=False, message=(
            "No phone detected. The phone is the scale reference, so it must be "
            "visible. Try a clearer photo where the phone isn't hidden."))

    person_mask, person_box = person
    phone_box, phone_mask, phone_conf = phone

    pose_res = pose_model(bgr, imgsz=DETECT_IMGSZ, verbose=False)[0]
    kpts = _select_best_pose(pose_res)

    # --- phone width from the uncovered top of the phone mask ---
    phone_width_px, occluded_note = _phone_width_px(phone_mask, phone_box)

    # --- head box (always, when face keypoints exist) ---
    head = _head_box(kpts, person_mask) if kpts is not None else None

    # --- crown of the head ---
    if head is not None:
        crown_y = head["crown_y"]
    else:
        ext = _mask_vertical_extent(person_mask)
        crown_y = ext[0] if ext else 0

    # --- stature (px) with the priority: full body -> head -> landmark ---
    basis, stature_px, foot_y, stat_notes = _stature_pixels(
        kpts, head, crown_y, person_mask, H)

    # --- depth: map + local phone-vs-chest correction ---
    depth_np, depth_vis = _run_depth(depth_pipe, img_rgb, (H, W))
    depth_factor, phone_depth_applied, depth_notes = _depth_correction(
        depth_np, phone_box, phone_mask, person_mask, kpts)

    eff_phone_width_px = phone_width_px * depth_factor
    px_per_mm_corrected = eff_phone_width_px / phone_width_mm
    px_per_mm_naive = phone_width_px / phone_width_mm

    height_mm = stature_px / px_per_mm_corrected
    naive_mm = stature_px / px_per_mm_naive

    # --- uncertainty budget ---
    unc = 3.0
    if "generic" in phone_source.lower():
        unc += 3.0
    if basis == "head":
        unc += 10.0
    elif basis in LANDMARK_UNCERTAINTY_CM:
        unc += LANDMARK_UNCERTAINTY_CM[basis]
    if not phone_depth_applied:
        unc += 2.0
    if occluded_note:
        unc += 1.5

    notes = []
    notes += stat_notes + depth_notes
    if occluded_note:
        notes.append(occluded_note)
    if phone_conf < 0.4:
        notes.append(f"Phone detection confidence is low ({phone_conf:.2f}).")

    res = Result(
        ok=True,
        height_cm=height_mm / 10.0,
        naive_height_cm=naive_mm / 10.0,
        uncertainty_cm=unc,
        basis=basis,
        phone_source=phone_source,
        phone_width_px=phone_width_px,
        depth_factor=depth_factor,
        phone_depth_applied=phone_depth_applied,
        depth_map=depth_vis,
        notes=notes,
    )
    res.annotated = _annotate(
        img_rgb, phone_box, person_box, head, crown_y, foot_y, stature_px, kpts, res)
    return res


def _phone_width_px(phone_mask, phone_box):
    """
    Robust phone width in px, from the widest (uncovered) rows of the mask.
    """
    x1, y1, x2, y2 = [int(v) for v in phone_box]
    box_w = x2 - x1
    box_h = y2 - y1
    if phone_mask is None:
        return float(box_w), None
    rows = phone_mask[max(0, y1):y2, :]
    widths = rows.sum(axis=1)
    widths = widths[widths > 0]
    if widths.size == 0:
        return float(box_w), None
    width_px = float(np.percentile(widths, 90))
    # A fully visible vertical phone is ~2.1x taller than wide. If the box is
    # much squarer, the bottom is hidden/tilted -- width is still fine, just note.
    note = None
    if box_h < 1.6 * width_px:
        note = ("The phone's bottom looks hidden or tilted; visible top width was used as the ruler.")
    return width_px, note


def _head_box(kpts, person_mask):
    """
    Build a head box from face keypoints + the crown from the mask.
    Returns dict(box, crown_y, chin_y, head_len_px, cx) or None.
    """
    nose = _kp(kpts, "nose")
    eyes = _kp_pair(kpts, "l_eye", "r_eye")
    ears = (_kp(kpts, "l_ear"), _kp(kpts, "r_ear"))
    if nose is None and eyes is None:
        return None
    anchor = eyes or nose
    cx = anchor[0]

    # head width from ears (fallback: eye distance)
    if ears[0] and ears[1]:
        head_w = abs(ears[0][0] - ears[1][0]) * 1.35
        cx = (ears[0][0] + ears[1][0]) / 2
    else:
        le, re = _kp(kpts, "l_eye"), _kp(kpts, "r_eye")
        if le and re:
            head_w = abs(le[0] - re[0]) * 2.6
        else:
            head_w = 90.0
    head_w = max(head_w, 20.0)

    # crown = topmost mask pixel within the head's horizontal span
    H, W = person_mask.shape
    xl = max(0, int(cx - head_w / 2))
    xr = min(W, int(cx + head_w / 2))
    strip = person_mask[:, xl:xr]
    ys = np.where(strip.any(axis=1))[0]
    crown_y = int(ys.min()) if len(ys) else int(anchor[1] - head_w)
    """
    Head length (crown -> chin). This is inherently noisy: crown-based anchors
    run LONG when there's tall hair, while the ear-to-ear width runs SHORT when
    the head is turned. We average a crown-based estimate with a hair-
    independent width-based one so the two biases partly cancel.
    """
    crown_based = []
    if eyes is not None:
        crown_based.append((eyes[1] - crown_y) / 0.50)
    if nose is not None:
        crown_based.append((nose[1] - crown_y) / 0.66)
    ests = []
    if crown_based:
        ests.append(float(np.mean(crown_based)))
    if ears[0] and ears[1]:
        ests.append(1.55 * abs(ears[0][0] - ears[1][0]))  # crown-chin ~1.5x breadth
    head_len = float(np.mean(ests)) if ests else head_w * 1.35
    head_len = float(np.clip(head_len, 0.9 * head_w, 2.2 * head_w))
    chin_y = crown_y + head_len

    return {
        "box": (cx - head_w / 2, crown_y, cx + head_w / 2, chin_y),
        "crown_y": crown_y,
        "chin_y": chin_y,
        "head_len_px": chin_y - crown_y,
        "cx": cx,
    }


def _stature_pixels(kpts, head, crown_y, person_mask, img_h):
    """Return (basis, stature_px, foot_y, notes) using the priority order."""
    notes = []

    # 1) Full body: reliable ankles -> extend to the sole.
    if kpts is not None:
        ankle = _kp_pair(kpts, "l_ankle", "r_ankle", conf_th=0.4)
        hip = _kp_pair(kpts, "l_hip", "r_hip", conf_th=0.4)
        if ankle is not None and (hip is None or ankle[1] > hip[1]):
            knee = _kp_pair(kpts, "l_knee", "r_knee", conf_th=0.3)
            shin = (ankle[1] - knee[1]) if knee else 0.06 * (ankle[1] - crown_y)
            foot_y = ankle[1] + FOOT_OFFSET_SHIN * max(shin, 0)
            foot_y = min(foot_y, img_h - 1)
            return "full body", foot_y - crown_y, foot_y, notes

    # 2) Head-based: most partial selfies show the whole head.
    if head is not None and head["head_len_px"] > 5:
        stature_px = head["head_len_px"] * HEAD_TO_STATURE
        notes.append(
            "Feet not visible -> height estimated from head size "
            f"(head length x {HEAD_TO_STATURE:g}). Lower-confidence.")
        return "head", stature_px, crown_y + stature_px, notes

    # 3) Lowest visible landmark.
    if kpts is not None:
        for name, keys in (("knee", ["l_knee", "r_knee"]),
                           ("hip", ["l_hip", "r_hip"]),
                           ("shoulder", ["l_shoulder", "r_shoulder"])):
            pt = _kp_pair(kpts, keys[0], keys[1])
            if pt is not None:
                frac = CROWN_TO_LANDMARK_FRACTION[name]
                stature_px = (pt[1] - crown_y) / frac
                notes.append(
                    f"Feet & head unclear -> extrapolated from the {name} "
                    f"(assumes {name} at {frac:.0%} of height). Low-confidence.")
                return name, stature_px, crown_y + stature_px, notes

    # 4) Nothing usable -> silhouette extent.
    ext = _mask_vertical_extent(person_mask)
    bottom = ext[1] if ext else crown_y
    notes.append("No keypoints; assumed the silhouette spans head to feet.")
    return "full body", bottom - crown_y, bottom, notes


def _run_depth(depth_pipe, pil_rgb, target_hw):
    """Return (disparity_map[H,W], colorized_PIL). Higher value = closer."""
    out = depth_pipe(pil_rgb)
    d = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    H, W = target_hw
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
    vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return d, Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))


def _depth_correction(depth_np, phone_box, phone_mask, person_mask, kpts):
    """
    Rescale factor phone->body plane, from LOCAL depth (phone vs chest).

    Sampling the chest directly behind the phone (rather than the whole body)
    is robust to mirrors compressing the whole reflection to "far". Returns
    (factor<=1, applied, notes).
    """
    notes = []
    x1, y1, x2, y2 = [int(v) for v in phone_box]

    if phone_mask is not None and phone_mask.any():
        phone_disp = float(np.median(depth_np[phone_mask]))
    else:
        region = depth_np[max(0, y1):y2, max(0, x1):x2]
        phone_disp = float(np.median(region)) if region.size else 0.0

    # chest band right behind the phone
    chest = _chest_mask(person_mask, phone_box, kpts)
    if chest is not None and chest.any():
        body_disp = float(np.median(depth_np[chest]))
        ref = "chest"
    else:
        bm = person_mask.copy()
        bm[max(0, y1):y2, max(0, x1):x2] = False
        vals = depth_np[bm]
        body_disp = float(np.median(vals)) if vals.size else 0.0
        ref = "body"

    if phone_disp <= 0 or body_disp <= 0:
        return 1.0, False, ["Depth unreadable; no correction applied."]

    factor = body_disp / phone_disp  # = phone_depth / body_depth
    if factor > 1.0:
        notes.append(
            f"Depth read the phone as behind the {ref} (factor {factor:.2f}); "
            "likely a mirror artifact, so no depth correction was applied.")
        return 1.0, False, notes
    """
    Physical bound: a phone held in front of the chest is realistically only
    ~10-25% closer than the body in a mirror selfie. Clamp to avoid a noisy
    depth map (common through mirrors) producing an extreme correction.
    """
    factor = float(np.clip(factor, 0.80, 1.0))
    notes.append(
        f"Depth correction: phone ~{(1-factor)*100:.0f}% closer than the {ref}; "
        f"its width was scaled by {factor:.2f} before measuring.")
    return factor, True, notes


def _chest_mask(person_mask, phone_box, kpts):
    """Boolean mask of the upper torso behind the phone, excluding the phone."""
    if kpts is None:
        return None
    sh = _kp_pair(kpts, "l_shoulder", "r_shoulder")
    hip = _kp_pair(kpts, "l_hip", "r_hip")
    ls, rs = _kp(kpts, "l_shoulder"), _kp(kpts, "r_shoulder")
    if sh is None or ls is None or rs is None:
        return None
    top = int(sh[1])
    bottom = int(hip[1]) if hip else int(sh[1] + 0.3 * person_mask.shape[0])
    bottom = int(top + 0.45 * (bottom - top)) if bottom > top else top + 40
    xl, xr = sorted((int(ls[0]), int(rs[0])))
    band = np.zeros_like(person_mask)
    band[max(0, top):max(top + 1, bottom), max(0, xl):max(xl + 1, xr)] = True
    chest = person_mask & band
    x1, y1, x2, y2 = [int(v) for v in phone_box]
    pad = 8
    chest[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = False
    return chest


def _load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _annotate(pil_rgb, phone_box, person_box, head, crown_y, foot_y,
              stature_px, kpts, result):
    img = pil_rgb.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, int(min(W, H) * 0.004))
    font = _load_font(max(14, int(min(W, H) * 0.030)))
    small = _load_font(max(11, int(min(W, H) * 0.020)))

    # person box (cyan)
    px1, py1, px2, py2 = person_box
    draw.rectangle([px1, py1, px2, py2], outline=(0, 200, 255), width=lw)

    # head box (orange)
    if head is not None:
        hx1, hy1, hx2, hy2 = head["box"]
        draw.rectangle([hx1, hy1, hx2, hy2], outline=(255, 150, 0), width=lw)
        draw.text((hx1, max(0, hy1 - 20)), "head", fill=(255, 150, 0), font=small)

    # phone box (magenta)
    fx1, fy1, fx2, fy2 = phone_box
    draw.rectangle([fx1, fy1, fx2, fy2], outline=(255, 0, 200), width=lw)
    draw.text((fx1, max(0, fy1 - 20)), "phone (width ref)",
              fill=(255, 0, 200), font=small)

    # keypoints (yellow)
    if kpts is not None:
        r = lw + 1
        for x, y, c in kpts:
            if c >= 0.4:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 230, 0))

    # height line (green) crown -> foot, clipped to the image
    cx = (px1 + px2) / 2
    fy = min(max(crown_y + stature_px, foot_y), H - 1) if foot_y else crown_y + stature_px
    fy = min(fy, H - 1)
    draw.line([(cx, crown_y), (cx, fy)], fill=(60, 255, 60), width=lw)
    for yy in (crown_y, fy):
        draw.line([(cx - 18, yy), (cx + 18, yy)], fill=(60, 255, 60), width=lw)
    if result.basis != "full body":
        draw.text((cx + 22, (crown_y + fy) / 2),
                  f"est. via {result.basis}", fill=(60, 255, 60), font=small)

    # label banner: feet+inches first (with +/-), then centimetres (with +/-)
    inch = result.height_cm / 2.54
    ft_whole = int(inch // 12)
    in_rem = round(inch - ft_whole * 12)
    if in_rem == 12:
        ft_whole, in_rem = ft_whole + 1, 0
    unc_in = result.uncertainty_cm / 2.54
    label = f"{ft_whole}'{in_rem}\"  (+/- {unc_in:.0f}\")"
    label2 = (f"{result.height_cm:.0f} cm (+/- {result.uncertainty_cm:.0f})"
              f"  |  basis: {result.basis}")
    draw.rectangle([0, 0, W, int(max(H * 0.075, 44))], fill=(0, 0, 0))
    draw.text((10, 6), label, fill=(255, 255, 255), font=font)
    draw.text((10, 8 + font.size), label2, fill=(180, 255, 180), font=small)
    return img
