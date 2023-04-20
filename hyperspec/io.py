from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import numpy.typing as npt
import spectral
import xarray as xr
from scipy.ndimage import gaussian_filter

__all__ = ["read_cube", "crop", "read_preview"]

TCropArr = TypeVar("TCropArr", npt.NDArray, xr.DataArray)


def crop(arr: TCropArr, bounds: npt.NDArray[np.int_]) -> TCropArr:
    xmin, xmax = np.sort(bounds, axis=0)[:, 0][[0, -1]]
    ymin, ymax = np.sort(bounds, axis=1)[:, 1][[0, -1]]
    return arr[ymin:ymax, xmin:xmax]


def read_cube(path: Path, bounds: npt.NDArray | None, smooth: float = 0.0) -> xr.DataArray:
    raw = spectral.open_image(str(path))
    if type(raw) != spectral.io.bilfile.BilFile:
        _err = f"Expected BIL hypercube, got {type(raw)}"
        raise ValueError(_err)

    data = np.rot90(raw.asarray(), -1)
    if smooth > 0.0:
        data = gaussian_filter(data, sigma=smooth)

    cube = xr.DataArray(
        data,
        dims=("y", "x", "band"),
        coords={
            "x": np.arange(raw.ncols),
            "y": np.arange(raw.nrows),
            "band": raw.bands.centers,
        },
    )

    if bounds is not None:
        cube = crop(cube, bounds)
    return cube


def read_preview(
    cube_path: Path, bounds: npt.NDArray[np.int_] | None = None, *, smooth: float = 0.0, greyscale: bool = False
) -> npt.NDArray:
    ident = cube_path.name.removeprefix("REFLECTANCE_").removesuffix(".hdr")
    path = cube_path.parents[1] / f"{ident}.png"
    if not path.exists():
        _err = f"Preview image not found at {path}"
        raise FileNotFoundError(_err)
    preview = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if greyscale:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    if smooth > 0.0:
        preview = gaussian_filter(preview, sigma=smooth)
    if bounds is not None:
        preview = crop(preview, bounds)
    return preview
