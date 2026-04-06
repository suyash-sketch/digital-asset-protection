"""
Digital Asset Protection - Robust Image Piracy Detection
=========================================================
Uses a multi-layered approach to detect piracy:
  Layer 1: Perceptual hash (fast, catches exact/near-exact copies)
  Layer 2: Tile-based hash matching (catches crops of the SAME image)
  Layer 3: ORB feature matching (catches similar content even from different
           angles, re-photographs, screenshots, etc.)

For the Google Solution Challenge - Digital Asset Protection
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import imagehash
import numpy as np
import cv2


# ============================================================================
# LAYER 1: Global Perceptual Hash
# ============================================================================

def global_hash_check(original_path, suspect_path):
    """
    Quick check using global perceptual hash.
    Catches exact copies, minor resizes, light compression.
    """
    orig = imagehash.phash(Image.open(original_path).convert('L'), hash_size=16)
    susp = imagehash.phash(Image.open(suspect_path).convert('L'), hash_size=16)
    dist = orig - susp
    print(f"  [L1] Global phash distance: {dist}")
    return dist


# ============================================================================
# LAYER 2: Tile-based Hash Matching
# ============================================================================

def is_informative(tile_img, min_std=15):
    """Filters out flat/blank tiles."""
    return np.array(tile_img).std() > min_std


def get_tile_hashes(image_path, tile_size=128, step=32, normalize_width=800):
    """Extract overlapping tile hashes from an image."""
    img = Image.open(image_path).convert('L')
    if normalize_width:
        ratio = normalize_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((normalize_width, new_height), Image.LANCZOS)

    width, height = img.size
    hashes = []

    if width < tile_size or height < tile_size:
        return hashes

    for top in range(0, height - tile_size + 1, step):
        for left in range(0, width - tile_size + 1, step):
            tile = img.crop((left, top, left + tile_size, top + tile_size))
            if is_informative(tile):
                hashes.append(imagehash.phash(tile, hash_size=8))

    return hashes


def tile_hash_check(original_path, suspect_path, threshold=10):
    """
    Tile-based comparison. Catches direct crops of the same image.
    """
    oh = get_tile_hashes(original_path)
    sh = get_tile_hashes(suspect_path)

    if not oh or not sh:
        return 0.0, 0.0

    # Suspect coverage: what % of suspect tiles match original
    s_match = sum(1 for s in sh if any((s - o) <= threshold for o in oh))
    # Original coverage: what % of original tiles match suspect
    o_match = sum(1 for o in oh if any((o - s) <= threshold for s in sh))

    s_cov = (s_match / len(sh)) * 100
    o_cov = (o_match / len(oh)) * 100

    print(f"  [L2] Tile matching - Suspect cov: {s_cov:.1f}%, Orig cov: {o_cov:.1f}%")
    return s_cov, o_cov


# ============================================================================
# LAYER 3: ORB Feature Matching (Key Innovation)
# ============================================================================

def orb_feature_check(original_path, suspect_path, normalize_width=800):
    """
    Uses ORB (Oriented FAST and Rotated BRIEF) to detect structural
    similarity between images. This is CRUCIAL for detecting:
    - Re-photographed content
    - Screenshots of original images
    - Same scene from slightly different angles
    - Cropped + resized + compressed copies

    Returns a match ratio (0.0 to 1.0) representing how much visual
    content is shared between the two images.
    """
    # Load and normalize images
    img1 = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("  [L3] Error: Could not load images for ORB")
        return 0.0

    # Normalize to consistent width
    for idx, img in enumerate([img1, img2]):
        h, w = img.shape
        if w != normalize_width:
            ratio = normalize_width / w
            new_h = int(h * ratio)
            if idx == 0:
                img1 = cv2.resize(img, (normalize_width, new_h))
            else:
                img2 = cv2.resize(img, (normalize_width, new_h))

    # Create ORB detector with many features
    orb = cv2.ORB_create(nfeatures=2000)

    # Detect and compute
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        print("  [L3] Too few features detected")
        return 0.0

    # BFMatcher with Hamming distance (appropriate for ORB binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Use knnMatch for Lowe's ratio test
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test (Lowe's test) - keep only distinctive matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Additional geometric verification using homography
    if len(good_matches) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC to find geometrically consistent matches
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if mask is not None:
            inliers = int(mask.sum())
        else:
            inliers = 0
    else:
        inliers = len(good_matches)

    # Compute match quality metrics
    min_features = min(len(kp1), len(kp2))
    match_ratio = inliers / min_features if min_features > 0 else 0
    raw_match_count = len(good_matches)

    print(f"  [L3] ORB features: {len(kp1)} vs {len(kp2)}")
    print(f"  [L3] Good matches: {raw_match_count}, Inliers: {inliers}")
    print(f"  [L3] Match ratio: {match_ratio:.3f} ({match_ratio*100:.1f}%)")

    return match_ratio


# ============================================================================
# MAIN VERDICT ENGINE
# ============================================================================

def check_for_piracy(original_path, suspect_path):
    """
    Multi-layered piracy detection.

    Decision logic:
    - L1 alone (global hash <= 15): Definite piracy (near-exact copy)
    - L2 alone (tile coverage > 40%): Piracy (direct crop)
    - L3 alone (ORB match > 15%): Piracy (similar content, different framing)
    - L1 + L3 combo: Lower thresholds when both show some similarity
    """
    print(f"\n{'='*60}")
    print(f"ORIGINAL: {original_path.split('/')[-1]}")
    print(f"SUSPECT : {suspect_path.split('/')[-1]}")
    print(f"{'='*60}")

    # --- Layer 1: Global Hash ---
    global_dist = global_hash_check(original_path, suspect_path)

    if global_dist <= 15:
        verdict = "PIRACY DETECTED"
        reason = f"Near-identical fingerprint (dist={global_dist})"
        print(f"\n>> VERDICT: {verdict}")
        print(f"   Reason: {reason}")
        return verdict

    # --- Layer 2: Tile Matching ---
    s_cov, o_cov = tile_hash_check(original_path, suspect_path)

    if s_cov > 40 or o_cov > 40:
        verdict = "PIRACY DETECTED"
        reason = f"Direct crop detected (S={s_cov:.1f}%, O={o_cov:.1f}%)"
        print(f"\n>> VERDICT: {verdict}")
        print(f"   Reason: {reason}")
        return verdict

    # --- Layer 3: ORB Feature Matching ---
    orb_ratio = orb_feature_check(original_path, suspect_path)

    if orb_ratio > 0.15:
        verdict = "PIRACY DETECTED"
        reason = f"Significant feature overlap (ORB={orb_ratio*100:.1f}%)"
        print(f"\n>> VERDICT: {verdict}")
        print(f"   Reason: {reason}")
        return verdict

    # --- Combined weak signals ---
    # If global hash is somewhat close AND ORB shows some matches
    if global_dist < 80 and orb_ratio > 0.08:
        verdict = "PIRACY DETECTED"
        reason = f"Combined signals (hash={global_dist}, ORB={orb_ratio*100:.1f}%)"
        print(f"\n>> VERDICT: {verdict}")
        print(f"   Reason: {reason}")
        return verdict

    # If tile matching shows moderate overlap AND ORB confirms
    if (s_cov > 20 or o_cov > 20) and orb_ratio > 0.05:
        verdict = "PIRACY DETECTED"
        reason = f"Tile+ORB combo (S={s_cov:.1f}%, O={o_cov:.1f}%, ORB={orb_ratio*100:.1f}%)"
        print(f"\n>> VERDICT: {verdict}")
        print(f"   Reason: {reason}")
        return verdict

    verdict = "ORIGINAL CONTENT"
    reason = f"No significant match (hash={global_dist}, tiles=S:{s_cov:.1f}%/O:{o_cov:.1f}%, ORB={orb_ratio*100:.1f}%)"
    print(f"\n>> VERDICT: {verdict}")
    print(f"   Reason: {reason}")
    return verdict


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL ASSET PROTECTION - PIRACY DETECTION")
    print("=" * 60)

    # Test 1: Completely different image (should be ORIGINAL)
    print("\n\n--- TEST 1: Different images (expect ORIGINAL) ---")
    check_for_piracy(
        'C:/Users/Admin/OneDrive/Pictures/Saved Pictures/ipl.png',
        'C:/Users/Admin/OneDrive/Pictures/Saved Pictures/image.jpg'
    )

    # Test 2: Cropped/similar copy (should be PIRACY)
    print("\n\n--- TEST 2: Cropped copy (expect PIRACY) ---")
    check_for_piracy(
        'C:/Users/Admin/OneDrive/Pictures/Saved Pictures/ipl.png',
        'C:/Users/Admin/OneDrive/Pictures/Saved Pictures/ipl1.png'
    )