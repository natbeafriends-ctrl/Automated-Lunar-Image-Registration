"""
Ground-truth RMSE validation for the lunar registration pipeline.

Uses the 35 crater centers from make_test_images.py (same RNG seed = same
craters) as control points, since their true positions in both photo1's
and photo2's frames are exactly known from the generator. This gives a
reproducible accuracy number without manual point-picking.

Run AFTER lunar_registration.py has produced results/homography.txt:

    python make_test_images.py
    python lunar_registration.py photo1_sun_upper_left.png photo2_sun_moderate_shift.png --out results
    python validate_rmse.py
"""

import numpy as np
from make_test_images import rng
from lunar_registration import compute_rmse

SIZE = (600, 800)
H_IMG, W_IMG = SIZE
N_CRATERS = 35

# Ground-truth transform used to generate photo2 from the relit terrain
# (must match make_test_images.py exactly)
ANGLE_DEG, SCALE, TX, TY = 12, 1.15, 40, -25


def true_crater_centers():
    """Re-draw the same craters make_test_images.py used (same seed, same order)."""
    craters = []
    for _ in range(N_CRATERS):
        cx = rng.uniform(0.05 * W_IMG, 0.95 * W_IMG)
        cy = rng.uniform(0.05 * H_IMG, 0.95 * H_IMG)
        r = rng.uniform(15, 55)
        depth = rng.uniform(40, 90)
        craters.append((cx, cy, r, depth))
    return np.array([(c[0], c[1]) for c in craters], dtype=np.float32)


def true_transform_matrix():
    import cv2
    M = cv2.getRotationMatrix2D((W_IMG / 2, H_IMG / 2), ANGLE_DEG, SCALE)
    M[0, 2] += TX
    M[1, 2] += TY
    return M


def main():
    pts1 = true_crater_centers()
    M = true_transform_matrix()
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_true = (M @ pts1_h.T).T

    in_frame = (
        (pts2_true[:, 0] >= 0) & (pts2_true[:, 0] < W_IMG) &
        (pts2_true[:, 1] >= 0) & (pts2_true[:, 1] < H_IMG)
    )
    pts1_f, pts2_f = pts1[in_frame], pts2_true[in_frame]
    print(f"Using {len(pts1_f)} / {len(pts1)} crater centers inside photo2's frame\n")

    H = np.loadtxt("results/homography.txt")
    errors, rmse = compute_rmse(H, pts1_f, pts2_f)
    print(f"\nFINAL: RMSE = {rmse:.2f} px over {len(pts1_f)} ground-truth control points")


if __name__ == "__main__":
    main()
