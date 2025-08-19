import argparse
import math
from typing import Callable, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self, width: float, x: Callable[[float], float], y: Callable[[float], float]):
        self.width = width
        self.x = x
        self.y = y

semi_circle_beta0 = -(1.0/2 + 1.0/4) * math.pi
semi_circle_r = math.sqrt(2)

semi_circle = Solution(
    2.0,
    lambda t: -1.0 + semi_circle_r * math.cos(semi_circle_beta0 + t),
    lambda t:  1.0 + semi_circle_r * math.sin(semi_circle_beta0 + t)
)

def _in_L(hx: float, hy: float, w: float = 1.0, eps: float = 1e-12) -> bool:
    return (hx <= eps and -eps <= hy <= w + eps) or (-w - eps <= hx <= eps and hy >= -eps)

def RunSolution(s: Solution, *, n_cols: int = 100, dteta: float = 0.001) -> Tuple[float, Set[int], float, int, int]:
    teta = 0.0
    pixel_size = s.width / n_cols
    n_rows = int(math.floor(1.0 / pixel_size))
    carved_out_set: Set[int] = set()

    while teta <= math.pi / 2.0 + 1e-12:
        xt = s.x(teta)
        yt = s.y(teta)
        ct = math.cos(teta)
        st = math.sin(teta)

        for i in range(n_rows):
            sofa_y = i * pixel_size + pixel_size * 0.5
            row_base = i * n_cols
            for j in range(n_cols):
                point_id = row_base + j
                if point_id in carved_out_set:
                    continue
                sofa_x = j * pixel_size + pixel_size * 0.5

                hx = xt + ct * sofa_x - st * sofa_y
                hy = yt + st * sofa_x + ct * sofa_y

                if not _in_L(hx, hy, 1.0):
                    carved_out_set.add(point_id)

        teta += dteta

    area = 1.0 * s.width - (pixel_size * pixel_size) * len(carved_out_set)
    return area, carved_out_set, pixel_size, n_rows, n_cols

def RenderRemaining(pixel_size: float, n_rows: int, n_cols: int, carved_out_set: Set[int], *, show: bool = True, save_path: str | None = None):
    img = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for i in range(n_rows):
        row_base = i * n_cols
        for j in range(n_cols):
            pid = row_base + j
            if pid not in carved_out_set:
                img[i, j] = 255
    if show or save_path:
        plt.figure(figsize=(6, 6))
        plt.imshow(img, origin='lower', interpolation='nearest', cmap='gray')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()

r = 2.0 / math.pi

hammersley = Solution(
    2.0 + 2.0 * r,
    lambda t: r * (math.cos(2.0 * t) - 1.0),
    lambda t: r * math.sin(2.0 * t),
)

SOLUTIONS: dict[str, Solution] = {
    "semi_circle": semi_circle,
    "hammersley": hammersley,
}


def choose_solution(name: str) -> tuple[Solution, str]:
    try:
        return SOLUTIONS[name], name
    except KeyError as exc:
        options = ", ".join(sorted(SOLUTIONS))
        raise ValueError(f"Unknown solution '{name}'. Available options: {options}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a moving sofa solution")
    parser.add_argument(
        "solution",
        choices=sorted(SOLUTIONS),
        help="which solution to run",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the result (use --no-render to skip)",
    )
    args = parser.parse_args()

    solution, name = choose_solution(args.solution)
    area, carved, px, rows, cols = RunSolution(solution, n_cols=600, dteta=0.05)
    print(
        f"{name.replace('_', ' ').capitalize()} sofa area ≈ {area:.6f}  [dx={solution.width/cols:.6f}]"
    )
    RenderRemaining(px, rows, cols, carved, show=args.render, save_path=None)

