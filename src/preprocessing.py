"""
Preprocessing utilities for Sign Language Recognition using Temporal Classification.

Turns a sign-language video into a fixed-size landmark sequence of shape (T, 144):
    1. MediaPipe Holistic landmark extraction (upper-body pose + both hands)
    2. Spatial normalisation (relative to shoulder mid-point, scaled by shoulder width)
    3. Linear interpolation of undetected values
    4. Temporal resampling to exactly T frames

Feature layout (144 values per frame, x/y/z for each landmark):
    [0:18]    pose  (L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist)
    [18:81]   left hand  (21 landmarks)
    [81:144]  right hand (21 landmarks)

Typical use (e.g. in the live demo):
    from src.preprocessing import preprocess_video
    x = preprocess_video("my_clip.mp4")      # shape (1, 40, 144), ready for the model
"""

import numpy as np

# ---------------- Config (must match the notebook used to build the dataset) ----------------
T = 40                       # time steps per sequence
MAX_FRAMES_TO_PROCESS = 64   # frames sampled per video before running MediaPipe
POSE_IDS = [11, 12, 13, 14, 15, 16]   # MediaPipe pose indices used
N_FEATS = (len(POSE_IDS) + 21 + 21) * 3   # 144
POSE_SL, LH_SL, RH_SL = slice(0, 18), slice(18, 81), slice(81, 144)


# ---------------- Landmark extraction ----------------
def landmarks_to_vector(res):
    """MediaPipe Holistic result -> vector of N_FEATS values (NaN where not detected)."""
    v = np.full(N_FEATS, np.nan, dtype=np.float32)
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        v[POSE_SL] = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in POSE_IDS]).ravel()
    if res.left_hand_landmarks:
        v[LH_SL] = np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark]).ravel()
    if res.right_hand_landmarks:
        v[RH_SL] = np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark]).ravel()
    return v


def extract_video_landmarks(path, start=1, end=-1, use_frame_bounds=False):
    """Run MediaPipe Holistic on a video file. Returns (num_frames, N_FEATS) array or None."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    if use_frame_bounds:
        frames = frames[max(int(start) - 1, 0): (int(end) if end and end > 0 else None)]
    if len(frames) == 0:
        return None

    keep = np.linspace(0, len(frames) - 1, min(len(frames), MAX_FRAMES_TO_PROCESS)).astype(int)
    out = np.full((len(keep), N_FEATS), np.nan, dtype=np.float32)
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as holistic:
        for i, k in enumerate(keep):
            rgb = cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)
            out[i] = landmarks_to_vector(holistic.process(rgb))
    return out


def detection_rates(raw):
    """Fraction of frames with the upper body detected, and with at least one hand detected."""
    pose = ~np.isnan(raw[:, POSE_SL]).any(1)
    lh = ~np.isnan(raw[:, LH_SL]).any(1)
    rh = ~np.isnan(raw[:, RH_SL]).any(1)
    return pose.mean(), (lh | rh).mean()


# ---------------- Preprocessing ----------------
def spatial_normalize(raw):
    """Make coordinates relative to the shoulder mid-point and divide by shoulder width."""
    a = raw.reshape(len(raw), -1, 3).astype(np.float64)   # (frames, 48 landmarks, xyz)
    ls, rs = a[:, 0, :], a[:, 1, :]                       # left / right shoulder
    valid = ~(np.isnan(ls).any(1) | np.isnan(rs).any(1))
    if valid.sum() == 0:
        return None
    center = np.median((ls[valid] + rs[valid]) / 2, axis=0)
    width = np.median(np.linalg.norm(ls[valid, :2] - rs[valid, :2], axis=1))
    if width < 1e-6:
        return None
    return ((a - center) / width).reshape(len(raw), -1)


def fill_missing(arr):
    """Linearly interpolate NaNs over time; channels that are always missing become 0."""
    out = arr.copy()
    idx = np.arange(len(arr))
    for j in range(out.shape[1]):
        col = out[:, j]
        ok = ~np.isnan(col)
        if ok.all():
            continue
        out[:, j] = 0.0 if not ok.any() else np.interp(idx, idx[ok], col[ok])
    return out


def resample_time(arr, t_new=T):
    """Resample a (frames, features) array to exactly t_new frames."""
    n = len(arr)
    if n == 1:
        return np.repeat(arr, t_new, axis=0)
    old, new = np.linspace(0, 1, n), np.linspace(0, 1, t_new)
    return np.stack([np.interp(new, old, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def process_raw(raw, t_new=T):
    """Raw landmarks (frames, 144) -> clean fixed-length sequence (t_new, 144), or None."""
    norm = spatial_normalize(raw)
    if norm is None:
        return None
    return resample_time(fill_missing(norm), t_new).astype(np.float32)


def preprocess_video(path):
    """Any video file -> array of shape (1, T, 144), the same format the model is trained on."""
    raw = extract_video_landmarks(path)
    seq = None if raw is None else process_raw(raw)
    return None if seq is None else seq[None, ...]


# ---------------- Augmentation (training data only) ----------------
def augment(seq, rng):
    """Random scale, small rotation, shift, jitter and speed change on a (T, 144) sequence."""
    s = seq.reshape(len(seq), -1, 3).copy()
    s *= rng.uniform(0.9, 1.1)
    th = np.deg2rad(rng.uniform(-10, 10))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s[..., :2] = s[..., :2] @ R.T
    s[..., :2] += rng.normal(0, 0.03, size=(1, 1, 2))
    s += rng.normal(0, 0.01, size=s.shape)
    out = s.reshape(len(seq), -1)
    gamma = rng.uniform(0.8, 1.25)
    old = np.linspace(0, 1, len(out))
    new = np.linspace(0, 1, len(out)) ** gamma
    out = np.stack([np.interp(new, old, out[:, j]) for j in range(out.shape[1])], axis=1)
    return out.astype(np.float32)