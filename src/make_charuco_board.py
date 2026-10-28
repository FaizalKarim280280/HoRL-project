#!/usr/bin/env python
"""Write a printable ChArUco calibration board (PDF + PNG) at exact physical size.

Print it at 100% / "actual size" (no "fit to page"), then measure one black square with a
ruler. If it is not exactly the size printed under the board, pass the measured size to the
calibration scripts with --square-mm. Tape the board flat; a warped board ruins calibration.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from wrist_vision import BoardSpec

PX_PER_MM = 20  # 508 dpi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "assets" / "charuco_board")
    p.add_argument("--squares-x", type=int, default=5)
    p.add_argument("--squares-y", type=int, default=7)
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--marker-mm", type=float, default=22.0)
    args = p.parse_args()

    spec = BoardSpec(args.squares_x, args.squares_y, args.square_mm / 1000, args.marker_mm / 1000)
    board_px = (round(spec.squares_x * args.square_mm * PX_PER_MM), round(spec.squares_y * args.square_mm * PX_PER_MM))
    board_img = spec.board().generateImage(board_px, marginSize=0, borderBits=1)

    margin = 10 * PX_PER_MM
    text_h = 12 * PX_PER_MM
    page = Image.new("L", (board_px[0] + 2 * margin, board_px[1] + 2 * margin + text_h), 255)
    page.paste(Image.fromarray(board_img), (margin, margin))
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("Helvetica", 4 * PX_PER_MM)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (margin, margin + board_px[1] + 3 * PX_PER_MM),
        f"ChArUco {spec.squares_x}x{spec.squares_y} {spec.dictionary} - square {args.square_mm:.1f} mm, "
        f"marker {args.marker_mm:.1f} mm. Print at 100% and measure a square!",
        fill=0,
        font=font,
    )
    dpi = PX_PER_MM * 25.4
    page.save(args.out.with_suffix(".pdf"), resolution=dpi)
    page.save(args.out.with_suffix(".png"), dpi=(dpi, dpi))
    w_mm, h_mm = page.size[0] / PX_PER_MM, page.size[1] / PX_PER_MM
    print(f"Wrote {args.out.with_suffix('.pdf')} and .png ({w_mm:.0f} x {h_mm:.0f} mm page; fits A4 and Letter).")


if __name__ == "__main__":
    main()
