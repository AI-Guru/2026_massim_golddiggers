"""Client-side "screenshot" of the game field: renders /api/field data with the frontend's sprites.

Same look as the web frontend (tree border, faded grass, fog of war outside the agents' 3x3
views, depot never fogged, one gold dot per carried item), but drawn with Pillow from the
structured field, so it is fast and deterministic. Sprites come from the server's /static/assets.
"""
from __future__ import annotations

import io
import urllib.request

from PIL import Image, ImageDraw

SPRITES = ("grass", "tree", "gold", "depot", "mark", "miner0", "miner1")
FALLBACK = {"grass": (36, 51, 31), "tree": (31, 91, 42), "gold": (255, 210, 63), "depot": (138, 90, 43),
            "mark": (211, 51, 51), "miner0": (227, 73, 72), "miner1": (42, 120, 214)}
GROUND = (36, 51, 31)
FOG = (0, 0, 0, 97)                  # 0.38 alpha, as in the frontend


class FieldRenderer:
    def __init__(self, server_url: str, cell: int = 16, border: int = 1):
        self.cell, self.border = cell, border
        self.sprites = {}
        for name in SPRITES:
            try:
                with urllib.request.urlopen(f"{server_url}/static/assets/{name}.png", timeout=10) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGBA")
            except Exception:
                img = Image.new("RGBA", (16, 16), FALLBACK[name] + (255,))
            self.sprites[name] = img.resize((cell, cell), Image.NEAREST)
        self._background = None          # (width, height, obstacles) -> image; obstacles never change

    def _static(self, w: int, h: int, obstacles: frozenset) -> Image.Image:
        key = (w, h, obstacles)
        if self._background and self._background[0] == key:
            return self._background[1]
        c, b = self.cell, self.border
        img = Image.new("RGBA", ((w + 2 * b) * c, (h + 2 * b) * c), GROUND + (255,))
        grass = self.sprites["grass"].copy()
        grass.putalpha(grass.getchannel("A").point(lambda a: int(a * 0.35)))
        for x in range(w + 2 * b):
            for y in range(h + 2 * b):
                img.alpha_composite(grass, (x * c, y * c))
                if x < b or y < b or x >= w + b or y >= h + b:
                    img.alpha_composite(self.sprites["tree"], (x * c, y * c))
        for x, y in obstacles:
            img.alpha_composite(self.sprites["tree"], ((x + b) * c, (y + b) * c))
        self._background = (key, img)
        return img

    def render(self, field: dict, fog: bool = True) -> Image.Image:
        w, h, c, b = field["width"], field["height"], self.cell, self.border
        rows = field["rows"]
        obstacles = frozenset((x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "#")
        img = self._static(w, h, obstacles).copy()
        px = lambda x, y: ((x + b) * c, (y + b) * c)   # noqa: E731
        for m in field["marks"]:
            img.alpha_composite(self.sprites["mark"], px(*m["pos"]))
        for y, r in enumerate(rows):
            for x, ch in enumerate(r):
                if ch == "$":
                    img.alpha_composite(self.sprites["gold"], px(x, y))
        # gold under an agent is hidden in the rows ("A"/"B" wins): nothing to draw there
        if fog:
            visible = {(a["pos"][0] + dx, a["pos"][1] + dy) for a in field["agents"]
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(shade)
            for x in range(w):
                for y in range(h):
                    if (x, y) not in visible:
                        x0, y0 = px(x, y)
                        d.rectangle([x0, y0, x0 + c - 1, y0 + c - 1], fill=FOG)
            img.alpha_composite(shade)
        img.alpha_composite(self.sprites["depot"], px(*field["depot"]))
        team_index = {t: i for i, t in enumerate(field["teams"].values())}
        draw = ImageDraw.Draw(img)
        dot = max(2, c // 6)
        for a in field["agents"]:
            x0, y0 = px(*a["pos"])
            img.alpha_composite(self.sprites[f"miner{team_index[a['team']]}"], (x0, y0))
            for i in range(a["items"]):
                draw.rectangle([x0 + i * (dot + 1), y0, x0 + i * (dot + 1) + dot, y0 + dot], fill=(17, 17, 17))
                draw.rectangle([x0 + i * (dot + 1), y0, x0 + i * (dot + 1) + dot - 1, y0 + dot - 1],
                               fill=(255, 210, 63))
        return img.convert("RGB")
