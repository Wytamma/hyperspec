import numba
import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn import decomposition as decomp

__all__ = ["pca", "pixelwise_cosine_similarity"]


def pca(cube: xr.DataArray, n_components: int = 3) -> xr.Dataset:
    """
    Computes principal components of a cube.
    Args:
      cube (xr.DataArray): The cube to compute principal components of.
      n_components (int): The number of components to compute.
    Returns:
      xr.Dataset: A dataset containing the principal components.
    Examples:
      >>> cube = xr.DataArray(np.random.rand(3, 3, 3))
      >>> pca(cube, n_components=2)
      <xarray.Dataset>
      Dimensions:  (band: 2)
      Coordinates:
        * band     (band) int64 0 1
      Data variables:
          0        (band) float64 0.541 0.8
          1        (band) float64 0.8 0.541
    """
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
def _cosine_similarity(arr1: npt.NDArray[np.float_], arr2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the vector-wise cosine similarity between two arrays of vectors.

    NOTE: This function calculates just the diagonal component of `scipy.distance.cdist(..., metric='cosine')`.

    Arguments:
        arr1: An array of vectors with shape (N, M) where N is the number of vectors and M is the dimensionality of each
              vector.
        arr2: An array of vectors with the same shape (N, M) as arr1.

    Returns:
        An array of shape (N,) containing the cosine similarity between each pair of vectors.
    """
    size = arr1.shape[0]
    dist = np.zeros(size)
    for ii in range(size):
        u = arr1[ii]
        v = arr2[ii]
        uv = np.average(u * v)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        dist[ii] = max(0, min(1.0 - uv / np.sqrt(uu * vv), 2.0))
    return 1.0 - dist


def pixelwise_cosine_similarity(cube1: npt.NDArray[np.float_], cube2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the cosine similarity between two cubes.
    Args:
      cube1 (npt.NDArray[np.float_]): The first cube.
      cube2 (npt.NDArray[np.float_]): The second cube.
    Returns:
      npt.NDArray[np.float_]: An array of shape (N,) containing the cosine similarity between each pair of vectors.
    Raises:
      ValueError: If cube1 and cube2 have different shapes.
    Examples:
      >>> cube1 = np.random.rand(3, 3, 3)
      >>> cube2 = np.random.rand(3, 3, 3)
      >>> pixelwise_cosine_similarity(cube1, cube2)
      array([[0.988, 0.988, 0.988],
             [0.988, 0.988, 0.988],
             [0.988, 0.988, 0.988]])
    """
    arr1 = cube1.reshape((-1, cube1.shape[-1]))
    arr2 = cube2.reshape((-1, cube2.shape[-1]))

    if arr1.shape != arr2.shape:
        _err = f"cube1 and cube2 must have the same shape, but got {arr1.shape} and {arr2.shape}"
        raise ValueError(_err)

    return _cosine_similarity(arr1, arr2).reshape(cube1.shape[:-1])
