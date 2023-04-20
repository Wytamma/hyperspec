import numba
import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn import decomposition as decomp

__all__ = ["pca", "pixelwise_cosine_similarity"]


def pca(cube: xr.DataArray, n_components: int = 3) -> xr.Dataset:
    model = decomp.PCA(n_components=n_components)
    bands = cube.band.values
    X = (  # noqa: N806  <- sklearn norm
        cube.dropna("x", how="all").dropna("y", how="all").values.reshape((-1, bands.size))
    )
    model.fit_transform(X)
    components = xr.Dataset(
        {str(ii): (("band",), component) for ii, component in enumerate(model.components_)},
        coords={"band": bands},
    )
    return components


@numba.jit(nopython=True)
def __cosine_similarity(arr1: npt.NDArray[np.float_], arr2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    size = arr1.shape[0]
    dist = np.zeros(size)
    for ii in range(size):
        u = arr1[ii]
        v = arr2[ii]
        uv = np.average(u * v)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        dist[ii] = max(0, min(1.0 - uv / np.sqrt(uu * vv), 2.0))
    return dist


def pixelwise_cosine_similarity(cube1: npt.NDArray[np.float_], cube2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    arr1 = cube1.reshape((-1, cube1.shape[-1]))
    arr2 = cube2.reshape((-1, cube2.shape[-1]))

    if arr1.shape != arr2.shape:
        _err = f"cube1 and cube2 must have the same shape, but got {arr1.shape} and {arr2.shape}"
        raise ValueError(_err)

    return 1.0 - __cosine_similarity(arr1, arr2).reshape(cube1.shape[:-1])
