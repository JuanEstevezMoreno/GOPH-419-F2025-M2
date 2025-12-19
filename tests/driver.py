import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, cast

SplineFunc = Callable[[Any], Any]

# Directory of *this* file (Examples/)
HERE = os.path.dirname(os.path.abspath(__file__))

# Path to src/ so we can import linalg_interp
SRC_PATH = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from linalg_interp import spline_function  # type: ignore[import]


def run() -> None:
    # Build full paths to data files in Examples/
    water_path = os.path.join(HERE, "water_density_vs_temp_usgs.txt")
    air_path   = os.path.join(HERE, "air_density_vs_temp_eng_toolbox.txt")

    # Load data
    water = np.loadtxt(water_path)
    air   = np.loadtxt(air_path)

    xd_w, yd_w = water[:, 0], water[:, 1]
    xd_a, yd_a = air[:, 0], air[:, 1]

    temps_w = np.linspace(xd_w[0], xd_w[-1], 100)
    temps_a = np.linspace(xd_a[0], xd_a[-1], 100)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Column 0: water, column 1: air
    for col, (xd, yd, temps, title) in enumerate([
        (xd_w, yd_w, temps_w, "Water Density"),
        (xd_a, yd_a, temps_a, "Air Density"),
    ]):
        for order in (1, 2, 3):
            # Get spline function and tell Pylance it's callable
            f_any = spline_function(xd, yd, order=order)
            f = cast(SplineFunc, f_any)

            y_interp = f(temps)

            ax = axes[order - 1, col]
            ax.plot(xd, yd, "o", label="data")
            ax.plot(temps, y_interp, "-", label=f"spline order {order}")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Density")
            ax.set_title(f"{title} — order {order}")
            ax.legend()

    plt.tight_layout()
    # Save figure in Examples/ as well
    out_path = os.path.join(HERE, "spline_plots.png")
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    run()