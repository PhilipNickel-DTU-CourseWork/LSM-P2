from argparse import ArgumentParser
import math
from time import perf_counter as time
import numpy as np

from numba import njit


# Create the argument-parser for easier arguments!
parser = ArgumentParser(description="Poisson problem")

parser.add_argument(
    "-N",
    type=int,
    default=100,
    help="Number of divisions along each of the 3 dimensions",
)
parser.add_argument("--iter", type=int, default=20, help="Number of (max) iterations.")
parser.add_argument("-v0", "--value0", type=float, default=0., help="The initial value of the grid u")
parser.add_argument(
    "--tolerance",
    type=float,
    default=1e-8,
    help="The tolerance of the normalized Frobenius norm of the residual for the convergence.",
)
parser.add_argument(
    "--save-slice",
    nargs=3,
    metavar=["axis", "pos", "FILE"],
    default=None,
    help="Store an image of a slice (pos in [-1;1])",
)

# Add your methods
# For each method the corresponding "step_{}" should exist.
methods = ["view"]
parser.add_argument(
    "--method",
    choices=methods,
    default=methods[0],
    help="The chosen method to solve the Poisson problem.",
)


# Grab options!
options = parser.parse_args()
N: int = options.N
method: str = options.method
N_iter: int = options.iter
tolerance: float = options.tolerance


"""
In the below code, we'll have the axes aligned as z, y, x.
"""


# Define the different methods
def step_view(uold: np.ndarray, u: np.ndarray, f: np.ndarray, h: float):
    """Run a single Poisson step using the *view* method"""

    # The stencil weight
    c: float = 1.0 / 6.0
    h2: float = h * h
    N: int = u.shape[0] - 2

    u[1:-1, 1:-1, 1:-1] = # TODO fill out

    return math.sqrt(np.sum((u - uold) ** 2)) / N**3



def plot_slice(filename: str, axis: str, pos, u: np.ndarray) -> None:
    """Create a slice of the grid and store it to a matplotlib file"""
    from matplotlib import pyplot as plt

    # get axis specification
    axis: int = "z2y1x0".index(axis.lower()) // 2

    pos = float(pos)
    N = u.shape[axis]
    idx: int = int(math.floor(N * (pos + 1) / 2))

    # For debugging purposes:
    #print(axis, pos, idx)
    uslice = np.take(u, idx, axis=axis)
    plt.imshow(uslice.T, vmin=-1, vmax=1, extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.savefig(filename)


# Allocate the matrices
h: float = 2.0 / (N-1)
u1: np.ndarray = np.full([N, N, N], options.value0, dtype=np.float64)
u1[[0, -1], :, :] = 0
u1[:, [0, -1], :] = 0
u1[:, :, [0, -1]] = 0
f: np.ndarray = np.zeros_like(u1)

# The boundary conditions are 0 on all edges.
u2: np.ndarray = u1.copy()
# TODO create f
f[...] = 


def validate(u: np.ndarray) -> bool:
    """ Validate the grid result """
    # TODO implement a validation method


# Retrieve the method that we'll use for solving the Poisson problem
step = globals()[f"step_{method}"]

# Preset the *result* array!
# Just in case the first run is not used...
u = u2.view()
t0 = time()
diff = i = -1

# Run the Poisson problem until either of these are true:
# 1. Maximum number of iterations is met.
# 2. The residual falls below the tolerance.
for i in range(N_iter):
    if i == 1:
        t0 = time()

    # Swapping pointers
    if i % 2 == 0:
        uold = u1.view()
        u = u2.view()
    else:
        u = u1.view()
        uold = u2.view()

    diff = step(uold, u, f, h)

    if diff < tolerance:
        # after this point, we know that `u` contains
        # the result, in any case
        # so we can safely break
        print("ended by tolerance check")
        break


t1 = time()

# Determine the actual number of iterations run
iter_run = i + 1
print(f"{diff = }")
print("time = ", t1 - t0)

if options.save_slice:
    axis, pos, filename = options.save_slice
    plot_slice(filename, axis, pos, u)
