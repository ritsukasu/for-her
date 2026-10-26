import argparse
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = "aliy"
WINDOW_TITLE = "Her Favorite Painting"

BG = np.array([8, 10, 14], dtype=np.float32)
WINDOW_BG = np.array([0, 0, 0], dtype=np.float32)
CANDIDATES = " .'`,:;-~=+*ox#%&@$MW"
BG_FACTOR = 0.2
FG_BOOST = 1.35
BLOCK_FRACTION = 0.55

FRAME_COLORS = {
    "top": (244, 204, 104),
    "left": (232, 186, 88),
    "right": (196, 144, 62),
    "bottom": (176, 128, 52),
}


def load_font(size):
    names = [
        "DejaVuSansMono-Bold.ttf", "consolab.ttf", "LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "Menlo.ttc", "/System/Library/Fonts/Menlo.ttc",
        "DejaVuSansMono.ttf", "consola.ttf", "Consolas.ttf",
        "LiberationMono-Regular.ttf", "cour.ttf", "Courier New.ttf",
    ]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


class Atlas:
    def __init__(self, font_size):
        font = load_font(font_size)
        asc, desc = font.getmetrics()
        self.cw = max(3, int(math.ceil(font.getlength("M"))))
        self.ch = max(5, asc + desc)

        def mask(c):
            im = Image.new("L", (self.cw, self.ch), 0)
            ImageDraw.Draw(im).text((0, 0), c, font=font, fill=255)
            return np.asarray(im, dtype=np.float32) / 255.0

        ramp = sorted(set(CANDIDATES), key=lambda c: mask(c).mean())
        edges = ["-", "\\", "|", "/"]

        def dilate(m, xs, ys):
            out = m.copy()
            for dx in xs:
                for dy in ys:
                    out = np.maximum(out, np.roll(np.roll(m, dy, axis=0), dx, axis=1))
            return np.clip(out * 1.6, 0, 1)

        bold_spec = {
            "-": ((0,), (-1, 0, 1)),
            "=": ((0,), (0, 1)),
            "|": ((-1, 0, 1), (0,)),
            "/": ((0, 1), (0, 1)),
            "\\": ((0, 1), (0, 1)),
            "+": ((-1, 0, 1), (-1, 0, 1)),
        }
        bold_chars = ["B" + c for c in bold_spec]
        bold_masks = [dilate(mask(c), *spec) for c, spec in bold_spec.items()]

        chars = ramp + edges + bold_chars
        self.masks = np.stack([mask(c) for c in ramp + edges] + bold_masks)
        self.n_ramp = len(ramp)
        self.edge_idx = np.array([len(ramp) + i for i in range(4)])
        self.index = {c: i for i, c in enumerate(chars)}


def build_pyramid(img):
    levels = [np.asarray(img, dtype=np.float32)]
    while min(levels[-1].shape[:2]) > 16:
        h, w = levels[-1].shape[:2]
        small = Image.fromarray(levels[-1].astype(np.uint8)).resize(
            (max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
        levels.append(np.asarray(small, dtype=np.float32))
    return levels


def art_cells(levels, atlas, tcols, trows, acols, arows, S):
    ih = levels[0].shape[0]

    offs = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    acc = np.zeros((trows, tcols, 3), dtype=np.float32)

    footprint = ih / (arows * S)
    lvl = int(np.clip(math.floor(math.log2(max(footprint, 1.0))), 0, len(levels) - 1))
    src = levels[lvl]
    sh, sw = src.shape[:2]

    for ox, oy in offs:
        ax = (np.arange(tcols) + 0.5 + ox - tcols / 2) / S
        ay = (np.arange(trows) + 0.5 + oy - trows / 2) / S
        ax, ay = np.meshgrid(ax, ay)
        px = (ax / acols + 0.5) * sw
        py = (ay / arows + 0.5) * sh
        pxi = np.clip(px.astype(int), 0, sw - 1)
        pyi = np.clip(py.astype(int), 0, sh - 1)
        acc += src[pyi, pxi]
    rgb = acc / len(offs)

    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0

    gy, gx = np.gradient(lum)
    gy = gy * 0.5
    mag = np.hypot(gx, gy)
    theta = (np.arctan2(gy, gx) + np.pi / 2) % np.pi
    eb = np.zeros(theta.shape, dtype=int)
    eb[(theta >= np.pi / 8) & (theta < 3 * np.pi / 8)] = 1
    eb[(theta >= 3 * np.pi / 8) & (theta < 5 * np.pi / 8)] = 2
    eb[(theta >= 5 * np.pi / 8) & (theta < 7 * np.pi / 8)] = 3

    idx = np.clip((lum ** 0.8 * (atlas.n_ramp - 1)).round().astype(int), 0, atlas.n_ramp - 1)
    idx = np.where((mag > 0.06) & (lum > 0.12), atlas.edge_idx[eb], idx)

    fg = np.clip(rgb * FG_BOOST + 45, 0, 255)
    bg = rgb * BG_FACTOR
    return idx, bg, fg


def frame_block(atlas, R, C, tr, tc, bold=False):
    ix = {k: atlas.index[("B" if bold else "") + k] for k in "-=|+/\\"}
    ix[" "] = atlas.index[" "]
    idx = np.full((R, C), ix[" "], dtype=int)
    fg = np.zeros((R, C, 3), dtype=np.float32)

    def shade(i, n):
        return 1.0 - 0.38 * (i / max(1, n - 1))

    for i in range(tr):
        ch = ix["-"]
        s = shade(i, tr)
        idx[i, :] = ch
        fg[i, :] = np.array(FRAME_COLORS["top"]) * s
        idx[R - 1 - i, :] = ch
        fg[R - 1 - i, :] = np.array(FRAME_COLORS["bottom"]) * s
    for j in range(tc):
        s = shade(j, tc)
        idx[tr:R - tr, j] = ix["|"]
        fg[tr:R - tr, j] = np.array(FRAME_COLORS["left"]) * s
        idx[tr:R - tr, C - 1 - j] = ix["|"]
        fg[tr:R - tr, C - 1 - j] = np.array(FRAME_COLORS["right"]) * s
    for (r_sl, c_sl, col) in [
        (slice(0, tr), slice(0, tc), FRAME_COLORS["top"]),
        (slice(0, tr), slice(C - tc, C), FRAME_COLORS["right"]),
        (slice(R - tr, R), slice(0, tc), FRAME_COLORS["left"]),
        (slice(R - tr, R), slice(C - tc, C), FRAME_COLORS["bottom"]),
    ]:
        idx[r_sl, c_sl] = ix["+"]
        fg[r_sl, c_sl] = np.array(col) * 0.8
    idx[0, 0] = ix["/"]
    idx[0, C - 1] = ix["\\"]
    idx[R - 1, 0] = ix["\\"]
    idx[R - 1, C - 1] = ix["/"]
    bg = fg * 0.14
    return idx, bg, fg


def rasterize(atlas, idx, bg, fg):
    R, C = idx.shape
    m = atlas.masks[idx][..., None]
    out = bg[:, :, None, None, :] * (1 - m) + fg[:, :, None, None, :] * m
    out = out.transpose(0, 2, 1, 3, 4).reshape(R * atlas.ch, C * atlas.cw, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def timeline(t, zoom_start, seconds):
    k = min(t / seconds, 1.0)
    e = k * k * (3 - 2 * k)
    return zoom_start ** (1 - e), k >= 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="blue.png")
    ap.add_argument("--size", type=int, default=460,
                    help="height in px of the painting + frame block at the end")
    ap.add_argument("--font-size", type=int, default=6,
                    help="smaller = more characters = more detail")
    ap.add_argument("--frame-lines", type=int, default=3,
                    help="how many stacked lines of characters make up each frame side")
    ap.add_argument("--bold-frame", action="store_true",
                    help="draw frame characters with heavier strokes")
    ap.add_argument("--block-fraction", type=float, default=BLOCK_FRACTION,
                    help="painting share of the window at the end (smaller = bigger window)")
    ap.add_argument("--zoom", type=float, default=1.03,
                    help="start zoom as a multiple of the minimum needed for the "
                         "painting to cover the whole window (>= 1)")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--save")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600), Image.LANCZOS)
    ia = img.width / img.height
    levels = build_pyramid(img)
    atlas = Atlas(args.font_size)

    tr = tc = args.frame_lines

    block_rows = max(14 + 2 * tr, round(args.size / atlas.ch))
    rows = block_rows - 2 * tr
    cols = max(8, round(rows * atlas.ch * ia / atlas.cw))
    block_cols = cols + 2 * tc

    total_rows = round(block_rows / args.block_fraction)
    total_cols = round(block_cols / args.block_fraction)
    if (total_rows - block_rows) % 2:
        total_rows += 1
    if (total_cols - block_cols) % 2:
        total_cols += 1

    f_idx, f_bg, f_fg = frame_block(atlas, block_rows, block_cols, tr, tc, args.bold_frame)
    space = atlas.index[" "]

    s_cover = max(total_cols / cols, total_rows / rows)
    s_start = s_cover * max(1.0, args.zoom)

    def draw(t):
        S, done = timeline(t, s_start, args.seconds)
        a_idx, a_bg, a_fg = art_cells(levels, atlas, total_cols, total_rows, cols, rows, S)

        cx = (np.arange(total_cols) + 0.5 - total_cols / 2) / S
        cy = (np.arange(total_rows) + 0.5 - total_rows / 2) / S
        cx, cy = np.meshgrid(cx, cy)

        art_m = (np.abs(cx) < cols / 2) & (np.abs(cy) < rows / 2)
        fcol = np.floor(cx + block_cols / 2).astype(int)
        frow = np.floor(cy + block_rows / 2).astype(int)
        in_blk = (fcol >= 0) & (fcol < block_cols) & (frow >= 0) & (frow < block_rows)
        fcol = np.clip(fcol, 0, block_cols - 1)
        frow = np.clip(frow, 0, block_rows - 1)

        idx = np.where(art_m, a_idx, np.where(in_blk, f_idx[frow, fcol], space))
        bg = np.where(art_m[..., None], a_bg,
                      np.where(in_blk[..., None], f_bg[frow, fcol], WINDOW_BG))
        fg = np.where(art_m[..., None], a_fg,
                      np.where(in_blk[..., None], f_fg[frow, fcol], WINDOW_BG))
        return rasterize(atlas, idx, bg.astype(np.float32), fg.astype(np.float32)), done

    if args.save:
        out, _ = draw(args.seconds + 1)
        Image.fromarray(out).save(args.save)
        print("saved", args.save, out.shape[1], "x", out.shape[0],
              f"({total_cols} cols x {total_rows} rows)")
        return

    import tkinter as tk
    from PIL import ImageTk

    bg_hex = "#%02x%02x%02x" % tuple(int(v) for v in WINDOW_BG)
    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.configure(bg=bg_hex)
    label = tk.Label(root, bd=0, bg=bg_hex)
    label.pack()
    state = {"t0": time.time(), "photo": None, "last": None, "job": None}

    def show(arr):
        state["last"] = arr
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(arr))
        label.configure(image=state["photo"])

    def tick():
        arr, done = draw(time.time() - state["t0"])
        show(arr)
        state["job"] = None if done else root.after(int(1000 / args.fps), tick)

    def replay(_=None):
        if state["job"]:
            root.after_cancel(state["job"])
        state["t0"] = time.time()
        tick()

    def save(_=None):
        path = os.path.join(os.path.dirname(os.path.abspath(args.image)), "garden_ascii.png")
        Image.fromarray(state["last"]).save(path)
        root.title(f"{WINDOW_TITLE} - saved {path}")

    win_w, win_h = total_cols * atlas.cw, total_rows * atlas.ch
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - win_w) // 2)
    y = max(0, (root.winfo_screenheight() - win_h) // 2)
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.resizable(False, False)
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("r", replay)
    root.bind("s", save)
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()