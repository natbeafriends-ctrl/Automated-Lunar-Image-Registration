# Lunar Surface Image Registration Pipeline

Aligns two photos of the same lunar spot taken at different times (different
sun angle, possibly different scale/rotation).

## Files
- `lunar_registration.py` — the pipeline (SIFT → FLANN match + ratio test →
  RANSAC affine/homography → warp → overlay → RMSE)
- `make_test_images.py` — generates a synthetic test pair (crater terrain,
  same physical texture, two different sun angles, known ground-truth
  rotate+scale+translate) so you can validate the pipeline before using real
  ISRO data
- `photo1_sun_upper_left.png`, `photo2_sun_moderate_shift.png` — the
  generated test pair
- `output/registration_summary.png` — 4-panel proof: all matches → RANSAC
  inliers → alpha overlay → checkerboard overlay

## Run it
```bash
python lunar_registration.py photo1.png photo2.png --out-dir output
```

Options:
- `--transform affine` (default) — rotation+scale+translation only. Use this
  unless you genuinely expect perspective distortion.
- `--transform homography` — full 8-DOF perspective. Needs 20+ well-spread
  inliers or it overfits.
- `--ratio 0.75` — Lowe's ratio test threshold (lower = stricter matches)
- `--reproj 5.0` — RANSAC pixel-distance threshold for inlier acceptance

## Validated result (synthetic pair)
- 4,705 good matches after ratio test → 4,670 RANSAC inliers
- Estimated transform vs. known ground-truth transform: **RMSE = 0.09 px**
  across 24 test points — near-perfect recovery

## Important finding from testing (put this in your slides — judges like it)
Ran the same pipeline on a version of photo2 with the sun angle **fully
flipped** (upper-left → upper-right, a worst-case shadow reversal) instead of
a moderate shift:
- Good matches dropped from 4,705 → 17
- RANSAC inliers dropped to 5
- Homography became unreliable / visibly misaligned in the overlay

**This is real, useful evidence for your presentation**: it proves the exact
failure mode described in the problem statement (extreme lighting breaks
plain SIFT), and justifies mentioning LoFTR as a stretch-goal upgrade for
extreme cases, while showing SIFT+RANSAC is solid for moderate lighting
differences — which covers most same-orbit revisit pairs.

## Using with real images
1. Replace the two PNG paths with your real photo pair.
2. Start with `--transform affine` — most same-altitude orbital revisits are
   well modeled by rotation+scale+translation, and affine is far more robust
   with fewer matches than homography.
3. For an accuracy number: manually pick 5–10 matching points you can verify
   by eye in both images, pass them to `compute_rmse()` in
   `lunar_registration.py` (see the `control_pts1`/`control_pts2` args of
   `run_pipeline()`), and report the RMSE in pixels.
4. If a pair has extreme shadow differences and match count collapses (like
   the flipped-sun test above), that's your cue to try LoFTR instead of SIFT.

## Why affine by default, not homography
Early testing with `findHomography` (8 DOF) on real crater-scale imagery
with only ~10 matches produced a poorly-constrained fit — the extra degrees
of freedom overfit to noisy matches. Switching to `estimateAffinePartial2D`
(4 DOF: rotation, uniform scale, translation) — which matches the actual
physics of same-altitude orbital revisits — reduced ground-truth error from
~70 px to under 1 px on the same match set. Use `--transform homography`
only when you have strong reason to expect real perspective distortion
(very different viewing angles) and enough inliers to constrain it.
