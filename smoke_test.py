"""Quick end-to-end check: imports load, models download, pipeline runs.

Usage:
    python smoke_test.py                # runs on a blank image (graceful path)
    python smoke_test.py path/to.jpg    # runs on a real photo
"""

import sys

from PIL import Image

import phone_specs
from pipeline import analyze


def main():
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        print(f"Loaded {sys.argv[1]}  size={img.size}")
    else:
        img = Image.new("RGB", (640, 960), (128, 128, 128))
        print("Using a blank 640x960 image (expect a graceful 'no person' result).")

    h_mm, w_mm, source = phone_specs.resolve_phone_size(phone_specs.GENERIC_LABEL)
    print(f"Phone reference: {source} -> {h_mm}x{w_mm} mm")

    print("Running pipeline (first run downloads model weights, be patient)...")
    res = analyze(img, h_mm, w_mm, source)

    print("\n=== RESULT ===")
    print("ok:", res.ok)
    if not res.ok:
        print("message:", res.message)
    else:
        print(f"height: {res.height_cm:.1f} cm  +/- {res.uncertainty_cm:.1f}")
        print(f"naive : {res.naive_height_cm:.1f} cm")
        print(f"basis={res.basis}  phone_width_px={res.phone_width_px:.0f}")
        print(f"depth_factor={res.depth_factor:.3f} applied={res.phone_depth_applied}")
        for n in res.notes:
            print("  -", n)
        out = "smoke_annotated.png"
        res.annotated.save(out)
        res.depth_map.save("smoke_depth.png")
        print(f"saved {out} and smoke_depth.png")
    print("\nPipeline executed without crashing. ✔")


if __name__ == "__main__":
    main()
