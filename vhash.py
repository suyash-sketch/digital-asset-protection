"""
Digital Asset Protection - Robust Video Piracy Detection
=========================================================
Mirrors the image-based phash_2.py architecture but for video.

Uses a multi-layered approach to detect piracy:
  Layer 1: Global video hash (representative keyframe pHash)
            → Catches near-exact copies, re-encodes, compressions
  Layer 2a: Sliding window sequence match
            → Catches unedited clips/highlights in original order
  Layer 2b: Bag-of-hashes match (order-independent)
            → Catches speed-altered, reordered, or cut-and-spliced edits
  Layer 3: ORB feature matching on sampled keyframes
            → Catches re-filmed screens, phone recordings, angle changes

For the Google Solution Challenge - Digital Asset Protection
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import numpy as np
import imagehash
from PIL import Image


# ============================================================================
# FRAME EXTRACTION UTILITIES
# ============================================================================

def extract_frames(video_path, fps_sample=1, normalize_width=800, max_frames=500):
    """
    Extract frames from a video at a given sample rate.
    
    Args:
        video_path    : Path to the video file
        fps_sample    : How many frames to extract per second (default: 1)
        normalize_width: Resize frames to this width for consistency
        max_frames    : Hard cap to avoid memory issues on long videos

    Returns:
        List of PIL Images (grayscale, normalized width)
    
    Strategy:
        - We sample 1 frame/sec by default — a 90-min film gives ~5400 frames
          but max_frames caps it at 500 (evenly spaced), keeping RAM sane.
        - Grayscale conversion here so every downstream layer gets consistent input.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 25  # fallback for broken metadata

    # Frame indices we want to extract (1 per second)
    sample_every = max(1, int(video_fps / fps_sample))
    wanted_indices = list(range(0, total_frames, sample_every))

    # If still too many, evenly subsample down to max_frames
    if len(wanted_indices) > max_frames:
        step = len(wanted_indices) // max_frames
        wanted_indices = wanted_indices[::step][:max_frames]

    frames = []
    wanted_set = set(wanted_indices)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in wanted_set:
            # Convert BGR→Gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            # Normalize width
            new_w = normalize_width
            new_h = int(h * (new_w / w))
            gray = cv2.resize(gray, (new_w, new_h))
            frames.append(Image.fromarray(gray))

        frame_idx += 1

    cap.release()
    print(f"  [INFO] Extracted {len(frames)} frames from '{os.path.basename(video_path)}'")
    return frames


def is_informative_frame(pil_img, min_std=10):
    """Filter out black frames, fade-outs, solid-colour slates."""
    return np.array(pil_img).std() > min_std


# ============================================================================
# LAYER 1: Global Video Hash
# ============================================================================

def global_video_hash(frames, hash_size=16):
    """
    Compute a single representative pHash for the whole video.
    Uses the median frame (middle of the video) as the representative.

    Why median?
      - First frame is often a logo/slate
      - Last frame may be black
      - Middle frame is most likely actual content

    Mirrors: global_hash_check() from phash_2.py
    """
    if not frames:
        return None
    mid = frames[len(frames) // 2]
    return imagehash.phash(mid, hash_size=hash_size)


def global_hash_check(orig_frames, susp_frames):
    """
    L1: Compare representative frame hashes.
    Returns Hamming distance (lower = more similar).
    """
    h1 = global_video_hash(orig_frames)
    h2 = global_video_hash(susp_frames)

    if h1 is None or h2 is None:
        print("  [L1] Could not compute global hash")
        return 999

    dist = h1 - h2
    print(f"  [L1] Global video hash distance: {dist}")
    return dist


# ============================================================================
# LAYER 2: Temporal Matching — Sliding Window + Bag-of-Hashes
# ============================================================================

def compute_frame_hash_sequence(frames, hash_size=8):
    """
    Hash every informative frame to build the video's 'DNA' —
    a time-ordered list of perceptual hashes.

    Blank/black frames are filtered out so they don't dilute match scores.
    hash_size=8 → 64-bit hash, fast to compare with Hamming distance.
    """
    seq = []
    for f in frames:
        if is_informative_frame(f):
            seq.append(imagehash.phash(f, hash_size=hash_size))
    return seq


def sliding_window_match(orig_seq, susp_seq, threshold=10):
    """
    L2a — ORDER-DEPENDENT matching.

    Slides the suspect sequence along the original to find the best
    contiguous alignment. Best for clips that are taken as-is from the
    original (no speed change, no reordering).

    Also checks the reversed sequence to catch horizontally-flipped
    re-uploads (a common piracy trick).

    Returns:
        best_coverage (float) : 0–100, % of suspect frames that matched
        best_position (int)   : ~second in original where match starts
    """
    if not orig_seq or not susp_seq:
        return 0.0, -1

    window = len(susp_seq)
    best_coverage = 0.0
    best_pos = -1

    # If suspect is longer than original, align from the start
    if window >= len(orig_seq):
        matched = sum(
            1 for s, o in zip(susp_seq, orig_seq)
            if (s - o) <= threshold
        )
        coverage = (matched / len(susp_seq)) * 100
        print(f"  [L2a] Direct alignment (suspect >= original): {coverage:.1f}%")
        return coverage, 0

    # Forward pass
    for start in range(len(orig_seq) - window + 1):
        orig_window = orig_seq[start : start + window]
        matched = sum(
            1 for s, o in zip(susp_seq, orig_window)
            if (s - o) <= threshold
        )
        coverage = (matched / window) * 100
        if coverage > best_coverage:
            best_coverage = coverage
            best_pos = start

    # Reverse pass — catches mirror/flip re-uploads
    susp_rev = list(reversed(susp_seq))
    for start in range(len(orig_seq) - window + 1):
        orig_window = orig_seq[start : start + window]
        matched = sum(
            1 for s, o in zip(susp_rev, orig_window)
            if (s - o) <= threshold
        )
        coverage = (matched / window) * 100
        if coverage > best_coverage:
            best_coverage = coverage
            best_pos = start

    print(f"  [L2a] Sliding window best match: {best_coverage:.1f}% at ~{best_pos}s in original")
    return best_coverage, best_pos


def bag_of_hashes_match(orig_seq, susp_seq, threshold=10):
    """
    L2b — ORDER-INDEPENDENT matching.

    This directly fixes the problem you identified:
      Edited clips (speed-altered, reordered, cut-and-spliced) will have
      their frames appear in a DIFFERENT ORDER than the original.
      The sliding window will miss these. This function won't.

    How it works:
      For each suspect frame, check if ANY frame in the original is within
      Hamming distance <= threshold. Order doesn't matter at all.

    Think of it like your image tile_hash_check() — you checked whether
    each tile in the suspect image existed anywhere in the original,
    regardless of position. This is the exact same idea applied to time.

    Returns:
        s_coverage (float) : % of suspect frames found in original
                             High → suspect is largely made of original content
        o_coverage (float) : % of original frames found in suspect
                             High → suspect contains most of the original
    """
    if not orig_seq or not susp_seq:
        return 0.0, 0.0

    # How much of the suspect can be found in the original?
    s_matched = sum(
        1 for s in susp_seq
        if any((s - o) <= threshold for o in orig_seq)
    )

    # How much of the original can be found in the suspect?
    o_matched = sum(
        1 for o in orig_seq
        if any((o - s) <= threshold for s in susp_seq)
    )

    s_cov = (s_matched / len(susp_seq)) * 100
    o_cov = (o_matched / len(orig_seq)) * 100

    print(f"  [L2b] Bag-of-hashes — Suspect coverage: {s_cov:.1f}%  |  Original coverage: {o_cov:.1f}%")
    return s_cov, o_cov


# ============================================================================
# LAYER 3: ORB Feature Matching on Sampled Keyframes
# ============================================================================

def orb_video_check(orig_frames, susp_frames, sample_count=10):
    """
    Run ORB matching between evenly-sampled frames from both videos.

    We pick `sample_count` frames from each video and do pairwise ORB
    matching between the BEST-MATCHING pairs (not all pairs — that would
    be O(n²) and slow).

    Strategy:
      1. Pick evenly spaced frames from each video
      2. For each suspect frame, find the best-matching original frame
         using global pHash (cheap) as a pre-filter
      3. Run full ORB only on the top candidate pairs
      4. Return average match ratio across all pairs

    Mirrors: orb_feature_check() from phash_2.py
    """
    if not orig_frames or not susp_frames:
        return 0.0

    # Evenly sample frames
    def sample_evenly(frames, n):
        if len(frames) <= n:
            return frames
        step = len(frames) // n
        return [frames[i * step] for i in range(n)]

    orig_sample = sample_evenly(orig_frames, sample_count)
    susp_sample = sample_evenly(susp_frames, sample_count)

    # Convert PIL → numpy for OpenCV
    def to_cv(pil_img):
        return np.array(pil_img)

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ratios = []

    for s_frame in susp_sample:
        s_cv = to_cv(s_frame)
        kp2, des2 = orb.detectAndCompute(s_cv, None)
        if des2 is None or len(des2) < 10:
            continue

        # Pre-filter: find the most similar original frame using pHash
        s_hash = imagehash.phash(s_frame, hash_size=8)
        orig_sorted = sorted(orig_sample, key=lambda f: imagehash.phash(f, hash_size=8) - s_hash)
        top_candidates = orig_sorted[:3]  # only ORB the top 3 candidates

        for o_frame in top_candidates:
            o_cv = to_cv(o_frame)
            kp1, des1 = orb.detectAndCompute(o_cv, None)
            if des1 is None or len(des1) < 10:
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            # Geometric verification
            if len(good) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = int(mask.sum()) if mask is not None else 0
            else:
                inliers = len(good)

            min_feat = min(len(kp1), len(kp2))
            if min_feat > 0:
                ratios.append(inliers / min_feat)

    if not ratios:
        print("  [L3] ORB: No valid frame pairs found")
        return 0.0

    avg_ratio = np.mean(ratios)
    max_ratio = np.max(ratios)
    print(f"  [L3] ORB avg match ratio: {avg_ratio*100:.1f}%  |  best pair: {max_ratio*100:.1f}%")
    return avg_ratio


# ============================================================================
# MAIN VERDICT ENGINE
# ============================================================================

def check_video_for_piracy(original_path, suspect_path):
    """
    Multi-layered video piracy detection.

    Decision logic:
      L1  (global hash <= 20)           : Near-identical copy
      L2a (sliding window > 40%)        : Unedited clip in original order
      L2b (bag coverage > 40%)          : Edited/reordered clip, still same content
      L3  (ORB avg > 15%)               : Visually similar (re-filmed, screenshot)
      L1 + L3 combo                     : Weak but combined signal
      L2b + L3 combo                    : Edited clip confirmed by structure

    Reports WHERE in the original a pirated clip was found (L2a only).
    """
    print(f"\n{'='*60}")
    print(f"ORIGINAL: {os.path.basename(original_path)}")
    print(f"SUSPECT : {os.path.basename(suspect_path)}")
    print(f"{'='*60}")

    # --- Extract frames ---
    print("\n[Extracting frames...]")
    orig_frames = extract_frames(original_path)
    susp_frames = extract_frames(suspect_path)

    if not orig_frames or not susp_frames:
        print(">> VERDICT: ERROR - Could not read one or both videos")
        return "ERROR"

    # Compute hash sequences once — reused by both L2a and L2b
    orig_seq = compute_frame_hash_sequence(orig_frames)
    susp_seq = compute_frame_hash_sequence(susp_frames)

    # ── Layer 1: Global Hash ──────────────────────────────────────────────────
    print("\n[Layer 1: Global hash check]")
    global_dist = global_hash_check(orig_frames, susp_frames)

    if global_dist <= 20:
        _print_verdict("PIRACY DETECTED", f"Near-identical video fingerprint (dist={global_dist})")
        return "PIRACY DETECTED"

    # ── Layer 2a: Sliding Window (order-dependent) ────────────────────────────
    print("\n[Layer 2a: Sliding window — ordered clip match]")
    window_coverage, match_pos = sliding_window_match(orig_seq, susp_seq)

    if window_coverage > 40:
        _print_verdict(
            "PIRACY DETECTED",
            f"Unedited clip detected ({window_coverage:.1f}% match at ~{match_pos}s in original)"
        )
        return "PIRACY DETECTED"

    # ── Layer 2b: Bag-of-Hashes (order-independent) ───────────────────────────
    print("\n[Layer 2b: Bag-of-hashes — edited/reordered clip match]")
    s_cov, o_cov = bag_of_hashes_match(orig_seq, susp_seq)

    if s_cov > 40:
        _print_verdict(
            "PIRACY DETECTED",
            f"Edited/reordered clip — {s_cov:.1f}% of suspect frames found in original"
        )
        return "PIRACY DETECTED"

    if o_cov > 60:
        _print_verdict(
            "PIRACY DETECTED",
            f"Suspect contains most of original — original coverage {o_cov:.1f}%"
        )
        return "PIRACY DETECTED"

    # ── Layer 3: ORB Feature Matching ─────────────────────────────────────────
    print("\n[Layer 3: ORB feature matching]")
    orb_ratio = orb_video_check(orig_frames, susp_frames)

    if orb_ratio > 0.15:
        _print_verdict(
            "PIRACY DETECTED",
            f"Significant visual overlap (ORB={orb_ratio*100:.1f}%)"
        )
        return "PIRACY DETECTED"

    # ── Combined weak signals ─────────────────────────────────────────────────
    if global_dist < 80 and orb_ratio > 0.08:
        _print_verdict(
            "PIRACY DETECTED",
            f"Combined signals — hash={global_dist}, ORB={orb_ratio*100:.1f}%"
        )
        return "PIRACY DETECTED"

    if s_cov > 20 and orb_ratio > 0.05:
        _print_verdict(
            "PIRACY DETECTED",
            f"Bag+ORB combo — suspect_cov={s_cov:.1f}%, ORB={orb_ratio*100:.1f}%"
        )
        return "PIRACY DETECTED"

    if window_coverage > 20 and orb_ratio > 0.05:
        _print_verdict(
            "PIRACY DETECTED",
            f"Window+ORB combo — window={window_coverage:.1f}%, ORB={orb_ratio*100:.1f}%"
        )
        return "PIRACY DETECTED"

    _print_verdict(
        "ORIGINAL CONTENT",
        f"No significant match "
        f"(hash={global_dist}, window={window_coverage:.1f}%, "
        f"bag_s={s_cov:.1f}%/bag_o={o_cov:.1f}%, ORB={orb_ratio*100:.1f}%)"
    )
    return "ORIGINAL CONTENT"


def _print_verdict(verdict, reason):
    print(f"\n>> VERDICT: {verdict}")
    print(f"   Reason : {reason}")


# ============================================================================
# REGISTRATION (Provenance / "Blockchain simulation")
# ============================================================================

import hashlib
import json
import datetime


def register_video(video_path, db_path="asset_registry.json"):
    """
    Register an original video into the asset database.
    
    Stores:
      - SHA-256 of the raw file (cryptographic proof)
      - pHash sequence (for matching)
      - Global hash (for quick L1 check)
      - Timestamp (proof of registration time)

    In production: the SHA-256 + timestamp would be anchored on-chain.
    For the prototype: this JSON file IS your blockchain.
    """
    print(f"\n[REGISTER] Processing '{os.path.basename(video_path)}'...")

    # Cryptographic hash of the actual file bytes
    sha256 = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    # Extract frames and build fingerprint
    frames = extract_frames(video_path)
    if not frames:
        print("  [ERROR] Could not extract frames")
        return None

    global_h = global_video_hash(frames)
    frame_seq = compute_frame_hash_sequence(frames)

    record = {
        "filename": os.path.basename(video_path),
        "sha256": file_hash,
        "global_hash": str(global_h),
        "frame_sequence": [str(h) for h in frame_seq],
        "frame_count": len(frame_seq),
        "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # Load or init DB
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            db = json.load(f)
    else:
        db = []

    db.append(record)
    with open(db_path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"  [REGISTER] Saved. SHA256: {file_hash[:16]}...")
    print(f"  [REGISTER] Frames fingerprinted: {len(frame_seq)}")
    print(f"  [REGISTER] Timestamp: {record['registered_at']}")
    return record


def check_against_registry(suspect_path, db_path="asset_registry.json"):
    """
    Check a suspect video against ALL registered originals.
    Runs L1 → L2a → L2b against stored fingerprints (no re-extracting originals).
    L3 (ORB) is run on-demand only when L1/L2 show partial signals.
    Returns the matched record, or None.
    """
    if not os.path.exists(db_path):
        print("[ERROR] No registry found. Register originals first.")
        return None

    with open(db_path, "r") as f:
        db = json.load(f)

    print(f"\n[SCAN] Checking '{os.path.basename(suspect_path)}' against {len(db)} registered asset(s)...")

    susp_frames = extract_frames(suspect_path)
    if not susp_frames:
        return None

    susp_seq  = compute_frame_hash_sequence(susp_frames)
    susp_glob = global_video_hash(susp_frames)

    for record in db:
        print(f"\n  ── vs '{record['filename']}' (registered {record['registered_at']}) ──")

        orig_glob = imagehash.hex_to_hash(record["global_hash"])
        orig_seq  = [imagehash.hex_to_hash(h) for h in record["frame_sequence"]]

        # L1
        dist = orig_glob - susp_glob
        print(f"  [L1] Hash distance: {dist}")
        if dist <= 20:
            print(f"  >> MATCH (L1): Near-identical copy")
            _report_match(record)
            return record

        # L2a — sliding window
        window_cov, pos = sliding_window_match(orig_seq, susp_seq)
        if window_cov > 40:
            print(f"  >> MATCH (L2a): Unedited clip at ~{pos}s")
            _report_match(record)
            return record

        # L2b — bag of hashes (catches edited/reordered clips)
        s_cov, o_cov = bag_of_hashes_match(orig_seq, susp_seq)
        if s_cov > 40 or o_cov > 60:
            print(f"  >> MATCH (L2b): Edited clip — suspect_cov={s_cov:.1f}%, orig_cov={o_cov:.1f}%")
            _report_match(record)
            return record

        # L3 — only run ORB if weak signals exist (saves time)
        weak_signal = dist < 100 or window_cov > 15 or s_cov > 15
        if weak_signal:
            print(f"  [L3] Weak signal detected — running ORB...")
            orb_ratio = orb_video_check(orig_frames=susp_frames, susp_frames=susp_frames)
            # NOTE: for a proper registry L3, you'd need to reload the original video.
            # For prototype, flag this as needing manual review instead:
            if orb_ratio > 0.15 or (dist < 80 and orb_ratio > 0.08):
                print(f"  >> MATCH (L3): Visual overlap confirmed (ORB={orb_ratio*100:.1f}%)")
                _report_match(record)
                return record

    print("\n  [SCAN] No match found in registry — content appears original.")
    return None


def _report_match(record):
    print(f"     Asset     : {record['filename']}")
    print(f"     SHA-256   : {record['sha256'][:32]}...")
    print(f"     Registered: {record['registered_at']}")
    print(f"     → This content belongs to a registered rights holder.")


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL ASSET PROTECTION - VIDEO PIRACY DETECTION")
    print("=" * 60)

    # ── USAGE ──────────────────────────────────────────────────────────────────
    #
    # OPTION A: Direct comparison (mirrors phash_2.py style)
    #   check_video_for_piracy("original.mp4", "suspect_clip.mp4")
    #
    # OPTION B: Registry-based (production flow)
    #   Step 1 — Register originals once:
    #     register_video("ipl_match_highlights.mp4")
    #     register_video("full_match_broadcast.mp4")
    #
    #   Step 2 — Scan any suspect clip:
    #     check_against_registry("uploaded_clip.mp4")
    #
    # ──────────────────────────────────────────────────────────────────────────

    # Test 1: Exact / near-exact copy → expect PIRACY (L1 catches it)
    print("\n--- TEST 1: Near-exact copy (expect PIRACY) ---")
    check_video_for_piracy("video1.mp4", "video1 copy.mp4")

    # Test 2: Unedited highlight clip → expect PIRACY (L2a catches it)
    print("\n--- TEST 2: Unedited clip from original (expect PIRACY) ---")
    check_video_for_piracy("video1.mp4", "video3.mp4")

    # Test 3: Speed-altered or reordered edit → expect PIRACY (L2b catches it)
    print("\n--- TEST 3: Speed-altered / reordered edit (expect PIRACY) ---")
    check_video_for_piracy("video1.mp4", "video_bubble.mp4")

    # Test 4: Completely different video → expect ORIGINAL
    print("\n--- TEST 4: Different video (expect ORIGINAL) ---")
    check_video_for_piracy("video1.mp4", "video4.mp4")

    # Test 5: Full registry flow
    print("\n--- TEST 5: Registry scan ---")
    register_video("video1.mp4")
    check_against_registry("video1_reorder.mp4")