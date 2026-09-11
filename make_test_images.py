"""
Generates two synthetic 'lunar-like' images to test the registration
pipeline end-to-end without needing real ISRO data yet.

Key fix vs v1: surface texture (regolith grain) is a FIXED physical
property of the terrain, rendered once. Photo 2 is derived by re-lighting
the SAME terrain (different sun angle) and then applying a known
similarity transform (rotate+scale+translate) -- exactly like a second
orbital pass over the same patch at a different time/angle.

  - photo1_sun_upper_left.png     : base view, sun from upper-left
  - photo2_sun_moderate_shift.png : same terrain, sun angle shifted,
                                     rotated+scaled+translated
"""

import cv2
import numpy as np

rng = np.random.default_rng(42)


def make_texture(size):
    """Multi-octave grain texture -- a fixed physical property of the terrain."""
    h, w = size
    texture = np.zeros((h, w), dtype=np.float32)
    for scale, amp in [(2, 18), (6, 12), (16, 8), (40, 5)]:
        small = rng.normal(0, 1, size=(max(1, h // scale), max(1, w // scale))).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        texture += up * amp
    return texture


def render_terrain(size, craters, texture, light_dir, base_gray=120, noise_std=4):
    """Render grey terrain + fixed texture, shaded per light_dir (sensor noise added fresh each render)."""
    h, w = size
    img = np.full((h, w), base_gray, dtype=np.float32)

    ly, lx = light_dir
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    for (cx, cy, r, depth) in craters:
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx**2 + dy**2)
        inside = dist < r

        with np.errstate(invalid="ignore", divide="ignore"):
            nz = np.sqrt(np.clip(r**2 - dist**2, 0, None)) / r
        nx = np.where(inside, -dx / r, 0)
        ny = np.where(inside, -dy / r, 0)
        norm = np.sqrt(lx**2 + ly**2) + 1e-6
        lxn, lyn = lx / norm, ly / norm
        lzn = 0.6
        shade = nx * lxn + ny * lyn + nz * lzn
        shade = np.clip(shade, -1, 1)

        crater_shade = shade * depth
        img = np.where(inside, base_gray + crater_shade, img)

        rim_band = (dist > r * 0.85) & (dist < r * 1.05)
        rim_dx, rim_dy = -dx / (dist + 1e-6), -dy / (dist + 1e-6)
        rim_facing = rim_dx * lxn + rim_dy * lyn
        rim_bonus = np.where(rim_band & (rim_facing > 0.3), 35 * rim_facing, 0)
        img = img + rim_bonus

    # fixed physical texture (same every render -- it's the terrain grain)
    img = img + texture

    # fresh sensor noise per shot (this DOES vary between photos, like a real camera)
    sensor_noise = rng.normal(0, noise_std, size=(h, w))
    img = img + sensor_noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    size = (600, 800)
    h, w = size

    n_craters = 35
    craters = []
    for _ in range(n_craters):
        cx = rng.uniform(0.05 * w, 0.95 * w)
        cy = rng.uniform(0.05 * h, 0.95 * h)
        r = rng.uniform(15, 55)
        depth = rng.uniform(40, 90)
        craters.append((cx, cy, r, depth))

    texture = make_texture(size)  # fixed terrain grain, shared by both photos

    # --- Photo 1: sun from upper-left ---
    img1 = render_terrain(size, craters, texture, light_dir=(-1, -1))

    # --- Photo 2: SAME terrain (same craters + same texture), sun angle
    #     shifted moderately, then a similarity transform applied to
    #     simulate a different orbital pass / different zoom ---
    img2_relit = render_terrain(size, craters, texture, light_dir=(-0.3, -1), base_gray=110)

    angle = 12
    scale = 1.15
    tx, ty = 40, -25
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img2 = cv2.warpAffine(img2_relit, M, (w, h), borderValue=(100, 100, 100))

    cv2.imwrite("/home/claude/photo1_sun_upper_left.png", img1)
    cv2.imwrite("/home/claude/photo2_sun_moderate_shift.png", img2)
    print("Saved photo1_sun_upper_left.png and photo2_sun_moderate_shift.png")
    print(f"Ground-truth transform applied to img2: rotate={angle} deg, "
          f"scale={scale}, translate=({tx},{ty})")


if __name__ == "__main__":
    main()
