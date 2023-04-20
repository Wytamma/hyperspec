import json
from pathlib import Path
from typing import Any

import cv2
import imutils
import numpy as np
import numpy.typing as npt
import typer
import xarray as xr

from hyperspec.io import read_cube, read_preview

__all__ = ["register"]


def _cli(
    dst_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    src_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    crops_path: Path = typer.Argument(..., exists=True),  # noqa: B008
    out_path: Path = typer.Argument(...),  # noqa: B008
    *,
    smooth: float = 0.0,
    debug: bool = False,
):
    capture_id = src_path.parent.parts[-2]

    with open(crops_path) as f:
        crops = json.load(f)

    if capture_id not in crops:
        _err = f"Capture ID {capture_id} not found in crops file"
        raise ValueError(_err)

    crop_bounds = np.around(np.array(crops[capture_id][:4])).astype(int)

    src_preview = read_preview(src_path, bounds=crop_bounds, smooth=smooth)
    dst_preview = read_preview(dst_path, bounds=crop_bounds, smooth=smooth)
    src_cube = read_cube(src_path, bounds=crop_bounds, smooth=smooth)

    result, result_preview = register(dst_preview, src_preview, src_cube, debug=debug)

    cv2.imwrite(f"{out_path.parent}/{out_path.stem}-preview.png", result_preview)
    xr.Dataset({capture_id: result}).to_zarr(out_path.with_suffix(".zarr"), mode="w")


def register(
    dst_preview: npt.NDArray,
    src_preview: npt.NDArray,
    src_cube: xr.DataArray,
    *,
    orb_create_kwargs: dict[str, Any] | None = None,
    flann_index_kwargs: dict[str, Any] | None = None,
    flann_search_kwargs: dict[str, Any] | None = None,
    debug: bool = False,
) -> tuple[xr.DataArray, npt.NDArray]:
    _orb_create_kwargs = {"nfeatures": 10_000, "scaleFactor": 1.2, "scoreType": cv2.ORB_HARRIS_SCORE}
    _orb_create_kwargs.update(orb_create_kwargs or {})
    orb = cv2.ORB_create(**_orb_create_kwargs)

    keypoints_src, descriptors_src = orb.detectAndCompute(src_preview, None)
    keypoints_dst, descriptors_dst = orb.detectAndCompute(dst_preview, None)

    _flann_index_kwargs = {"algorithm": 6, "table_number": 6, "key_size": 10, "multi_probe_level": 2}
    _flann_index_kwargs.update(flann_index_kwargs or {})
    _flann_search_kwargs = {"checks": 50}
    _flann_search_kwargs.update(flann_search_kwargs or {})
    matcher = cv2.FlannBasedMatcher(_flann_index_kwargs, _flann_search_kwargs)

    matches = [m for m, n in matcher.knnMatch(descriptors_src, descriptors_dst, k=2) if m.distance < 0.7 * n.distance]

    if debug:
        matched_vis = cv2.drawMatches(src_preview, keypoints_src, dst_preview, keypoints_dst, matches, None)
        matched_vis = imutils.resize(matched_vis, width=10_00)
        cv2.imshow("Matched Keypoints", matched_vis)
        cv2.waitKey(0)

    pts_src = np.array([keypoints_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_dst = np.array([keypoints_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homog, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    result_preview = cv2.warpPerspective(src_preview, homog, dst_preview.shape[:2][::-1])

    result = xr.zeros_like(src_cube)
    for band in result.band:
        result.loc[..., band] = cv2.warpPerspective(
            src_cube.sel(band=band).values, homog, dst_preview.shape[:2][::-1], borderValue=-999
        )
    result = xr.DataArray(result, dims=src_cube.dims, coords=src_cube.coords)

    return result, result_preview


if __name__ == "__main__":
    typer.run(_cli)
