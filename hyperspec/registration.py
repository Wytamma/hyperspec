import json
from pathlib import Path
from typing import TypeVar

import cv2
import imutils
import numpy as np
import spectral
import typer
import xarray as xr
from scipy.ndimage import gaussian_filter

__all__ = ["register"]

TCropArr = TypeVar("TCropArr", np.ndarray, xr.DataArray)


def crop(arr: TCropArr, bounds: np.ndarray) -> TCropArr:
    xmin, xmax = np.sort(bounds, axis=0)[:, 0][[0, -1]]
    ymin, ymax = np.sort(bounds, axis=1)[:, 1][[0, -1]]
    return arr[ymin:ymax, xmin:xmax]


def read_cube(path: Path, bounds: np.ndarray | None, smooth: float = 0.0) -> xr.DataArray:
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


def read_preview(cube_path: Path, bounds: np.ndarray | None, smooth: float = 0.0) -> np.ndarray:
    ident = cube_path.name.removeprefix("REFLECTANCE_").removesuffix(".hdr")
    path = cube_path.parents[1] / f"{ident}.png"
    if not path.exists():
        _err = f"Preview image not found at {path}"
        raise FileNotFoundError(_err)
    preview = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    if smooth > 0.0:
        preview = gaussian_filter(preview, sigma=smooth)
    if bounds is not None:
        preview = crop(preview, bounds)
    return preview


def _cli(
    dst_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    src_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    crops_path: Path = typer.Argument(..., exists=True),  # noqa: B008
    out_path: Path = typer.Argument(...),  # noqa: B008
    *,
    max_feat: int = 10_000,
    smooth: float = 0.0,
    debug: bool = False,
):
    register(dst_path, src_path, crops_path, max_feat=max_feat, smooth=smooth, debug=debug, save=out_path)


def register(
    dst_path: Path,
    src_path: Path,
    crops_path: Path,
    *,
    max_feat: int = 10_000,
    smooth: float = 0.0,
    debug: bool = False,
    save: Path | None = None,
) -> tuple[xr.DataArray, np.ndarray]:
    capture_id = src_path.parent.parts[-2]

    with open(crops_path) as f:
        crops = json.load(f)

    if capture_id not in crops:
        _err = f"Capture ID {capture_id} not found in crops file"
        raise ValueError(_err)

    crop_bounds = np.around(np.array(crops[capture_id][:4])).astype(int)

    src_preview = read_preview(src_path, bounds=crop_bounds, smooth=smooth)
    dst_preview = read_preview(dst_path, bounds=crop_bounds, smooth=smooth)

    orb = cv2.ORB_create(nfeatures=max_feat, scaleFactor=1.2, scoreType=cv2.ORB_HARRIS_SCORE)
    keypoints_src, descriptors_src = orb.detectAndCompute(src_preview, None)
    keypoints_dst, descriptors_dst = orb.detectAndCompute(dst_preview, None)

    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 6, "table_number": 6, "key_size": 10, "multi_probe_level": 2}, {"checks": 50}
    )
    matches = [m for m, n in matcher.knnMatch(descriptors_src, descriptors_dst, k=2) if m.distance < 0.7 * n.distance]

    if debug:
        matched_vis = cv2.drawMatches(src_preview, keypoints_src, dst_preview, keypoints_dst, matches, None)
        matched_vis = imutils.resize(matched_vis, width=10_00)
        cv2.imshow(f"Matched Keypoints - {capture_id}", matched_vis)
        cv2.waitKey(0)

    pts_src = np.array([keypoints_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_dst = np.array([keypoints_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homog, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    result_preview = cv2.warpPerspective(src_preview, homog, dst_preview.shape[:2][::-1])
    if save is not None:
        cv2.imwrite(f"{save.parent}/{save.stem}-preview.png", result_preview)

    src_cube = read_cube(src_path, bounds=crop_bounds, smooth=smooth)

    result = xr.zeros_like(src_cube)
    for band in result.band:
        result.loc[..., band] = cv2.warpPerspective(src_cube.sel(band=band).values, homog, dst_preview.shape[:2][::-1])
    result = xr.DataArray(result, dims=src_cube.dims, coords=src_cube.coords)

    if save:
        xr.Dataset({capture_id: result}).to_zarr(save.with_suffix(".zarr"), mode="w")

    return result, result_preview


if __name__ == "__main__":
    typer.run(_cli)
