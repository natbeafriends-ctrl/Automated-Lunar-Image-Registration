"""
Lunar Image Registration Pipeline
==================================
Aligns two lunar surface photos taken at different times (different lighting,
different scale/zoom) using SIFT features + RANSAC outlier rejection +
homography-based warping.

Pipeline:
    1. SIFT keypoint detection (illumination-tolerant shape features)
    2. FLANN-based matching + Lowe's ratio test
    3. RANSAC to find the geometric transform and reject bad matches
    4. Homography + warpPerspective to align image1 onto image2
    5. Overlay visualization + RMSE accuracy metric (if control points given)

Usage:
    python lunar_registration.py photo1.jpg photo2.jpg --out results/

Requires: opencv-contrib-python, numpy, matplotlib
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Phase 1 + 2: Feature detection and matching
# ----------------------------------------------------------------------

def detect_and_match(img1_gray, img2_gray, ratio_thresh=0.75):
    """SIFT keypoints/descriptors on both images, FLANN match, ratio test."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    print(f"[Phase 1] Keypoints -> img1: {len(kp1)}, img2: {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough keypoints detected. Try a different "
                            "image pair or lower SIFT contrast threshold.")

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"[Phase 2] Matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 10:
        print("[WARNING] Fewer than 10 good matches — RANSAC may be unreliable. "
              "Consider a different image pair or lower ratio_thresh.")

    return kp1, kp2, good_matches


# ----------------------------------------------------------------------
# Phase 3: RANSAC outlier rejection + homography
# ----------------------------------------------------------------------

def find_homography_ransac(kp1, kp2, good_matches, reproj_thresh=5.0):
    """Fit a homography with RANSAC, return H, inlier mask, and inlier matches."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)

    if H is None:
        raise RuntimeError("Homography estimation failed — matches too noisy.")

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]

    print(f"[Phase 3] RANSAC inliers: {inlier_mask.sum()} / {len(good_matches)} "
          f"({100 * inlier_mask.sum() / len(good_matches):.1f}%)")

    return H, inlier_mask, inlier_matches


# ----------------------------------------------------------------------
# Phase 4: Warp + overlay
# ----------------------------------------------------------------------

def warp_and_overlay(img1_color, img2_color, H):
    """Warp img1 onto img2's frame and build a blended overlay."""
    h2, w2 = img2_color.shape[:2]
    warped1 = cv2.warpPerspective(img1_color, H, (w2, h2))

    overlay = cv2.addWeighted(warped1, 0.5, img2_color, 0.5, 0)

    # Checkerboard overlay: alternates 40px blocks from each image, makes
    # misalignment along block boundaries very obvious.
    checker = img2_color.copy()
    block = 40
    for y in range(0, h2, block):
        for x in range(0, w2, block):
            if ((x // block) + (y // block)) % 2 == 0:
                checker[y:y + block, x:x + block] = warped1[y:y + block, x:x + block]

    return warped1, overlay, checker


# ----------------------------------------------------------------------
# Phase 4b: Accuracy metric (RMSE on manually chosen control points)
# ----------------------------------------------------------------------

def compute_rmse(H, control_pts_img1, control_pts_img2):
    """
    control_pts_img1 / control_pts_img2: list of (x, y) tuples, manually
    identified as the SAME real-world point in each image.
    Returns per-point error (px) and overall RMSE.
    """
    pts1 = np.float32(control_pts_img1).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts1, H).reshape(-1, 2)
    actual = np.float32(control_pts_img2)

    errors = np.linalg.norm(projected - actual, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    print("[Phase 4b] Control point errors (px):", np.round(errors, 2))
    print(f"[Phase 4b] RMSE: {rmse:.2f} px")
    return errors, rmse


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------

def save_match_visualization(img1_gray, kp1, img2_gray, kp2, matches, path, title):
    vis = cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if vis.ndim == 3 else vis, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_overlay_panel(img1_color, img2_color, warped1, overlay, checker, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (img1_color, "Photo 1 (original)"),
        (img2_color, "Photo 2 (original / reference)"),
        (warped1, "Photo 1 warped to Photo 2's frame"),
        (overlay, "50/50 blended overlay"),
        (checker, "Checkerboard overlay (misalignment check)"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lunar image registration pipeline")
    parser.add_argument("image1", help="Path to photo 1 (to be aligned)")
    parser.add_argument("image2", help="Path to photo 2 (reference)")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    parser.add_argument("--reproj", type=float, default=5.0, help="RANSAC reprojection threshold (px)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img1_color = cv2.imread(args.image1)
    img2_color = cv2.imread(args.image2)
    if img1_color is None or img2_color is None:
        sys.exit("Could not read one or both images — check the paths.")

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    kp1, kp2, good_matches = detect_and_match(img1_gray, img2_gray, args.ratio)
    save_match_visualization(
        img1_gray, kp1, img2_gray, kp2, good_matches,
        os.path.join(args.out, "01_matches_before_ransac.png"),
        f"Matches after ratio test ({len(good_matches)})"
    )

    H, inlier_mask, inlier_matches = find_homography_ransac(kp1, kp2, good_matches, args.reproj)
    save_match_visualization(
        img1_gray, kp1, img2_gray, kp2, inlier_matches,
        os.path.join(args.out, "02_matches_after_ransac.png"),
        f"RANSAC inlier matches ({len(inlier_matches)})"
    )

    print("[Phase 3] Homography matrix:\n", H)
    np.savetxt(os.path.join(args.out, "homography.txt"), H)

    warped1, overlay, checker = warp_and_overlay(img1_color, img2_color, H)
    save_overlay_panel(img1_color, img2_color, warped1, overlay, checker,
                        os.path.join(args.out, "03_overlay_panel.png"))
    cv2.imwrite(os.path.join(args.out, "warped_photo1.png"), warped1)
    cv2.imwrite(os.path.join(args.out, "overlay_blend.png"), overlay)
    cv2.imwrite(os.path.join(args.out, "overlay_checkerboard.png"), checker)

    print(f"\nDone. Outputs saved to: {args.out}/")
    print("Panels: 01_matches_before_ransac.png, 02_matches_after_ransac.png, "
          "03_overlay_panel.png")
    print("\nFor the accuracy score (RMSE), manually pick 5-10 matching points "
          "in both photos and call compute_rmse(H, pts1, pts2) — see the "
          "function docstring in this file.")


if __name__ == "__main__":
    main()
