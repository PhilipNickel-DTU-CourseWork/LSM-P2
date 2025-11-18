from mpi4py import MPI
import numpy as np
import math
from time import perf_counter as time
from argparse import ArgumentParser


def split_sizes(n, parts):
    """Split n points into `parts` integers as even as possible.
    Returns counts list and starts list (0-based).
    """
    base = n // parts
    rem = n % parts
    counts = [base + (1 if i < rem else 0) for i in range(parts)]
    starts = [sum(counts[:i]) for i in range(parts)]
    return counts, starts

parser = ArgumentParser(description="MPI 3D Poisson solver")
parser.add_argument('-N', type=int, default=100, help='Number of grid points per axis (global)')
parser.add_argument('--iter', type=int, default=2000, help='Max iterations')
parser.add_argument('-v0', '--value0', type=float, default=0.0, help='Initial u value')
parser.add_argument('--tolerance', type=float, default=1e-8, help='Tolerance on normalized step residual')
parser.add_argument('--output', type=str, default='output/u_final.npy', help='Output filename for assembled u (rank 0)')
parser.add_argument('--omega', type=float, default=1.0, help='Relaxation weight (Jacobi-style)')
options = parser.parse_args()

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create Cartesian communicator with 3 dims
dims = MPI.Compute_dims(size, 3)
cart = comm.Create_cart(dims=dims, periods=(False,False,False), reorder=False)
coords = cart.Get_coords(rank)

px, py, pz = dims

# Local grid sizes
N = options.N
# split each axis into px,py,pz parts
nx_counts, nx_starts = split_sizes(N, px)
ny_counts, ny_starts = split_sizes(N, py)
nz_counts, nz_starts = split_sizes(N, pz)

ix, iy, iz = coords
local_nx = nx_counts[ix]
local_ny = ny_counts[iy]
local_nz = nz_counts[iz]
local_x0 = nx_starts[ix]
local_y0 = ny_starts[iy]
local_z0 = nz_starts[iz]

# physical domain [-1,1]^3
h = 2.0 / (N - 1)

# We allocate arrays with ghost layers: +2 in each dimension
shape_local_with_ghosts = (local_nz + 2, local_ny + 2, local_nx + 2)

u_local = np.full(shape_local_with_ghosts, options.value0, dtype=np.float64)

# We'll keep two buffers for Jacobi-like update
u_old = np.copy(u_local)
u_new = np.copy(u_local)

# Create local coordinates for interior points (only for computing f and u_true)
# global indices for interior (0..N-1)
global_x = np.arange(local_x0, local_x0 + local_nx)
global_y = np.arange(local_y0, local_y0 + local_ny)
global_z = np.arange(local_z0, local_z0 + local_nz)

# convert to physical coordinates in [-1,1]
xs_local = -1.0 + (global_x) * h
ys_local = -1.0 + (global_y) * h
zs_local = -1.0 + (global_z) * h

# make broadcastable grids for interior (using ogrid-like shapes)
# shapes: (local_nz,1,1), (1,local_ny,1), (1,1,local_nx)
Zl = zs_local.reshape((local_nz, 1, 1))
Yl = ys_local.reshape((1, local_ny, 1))
Xl = xs_local.reshape((1, 1, local_nx))

# Compute f on interior and set into u_old/u_new (interior indices [1:-1,1:-1,1:-1])
h2 = h * h
f_local = 3 * (math.pi ** 2) * np.sin(math.pi * Xl) * np.sin(math.pi * Yl) * np.sin(math.pi * Zl)

# put initial arrays: u_local and u_old both with initial value and zero BC in ghost layers
# interior slices are indices [1:-1,1:-1,1:-1]

u_old[1:-1, 1:-1, 1:-1] = u_local[1:-1, 1:-1, 1:-1]  # they are the same initially
u_new = u_old.copy()

# Precompute u_true for local interior
u_true_local = np.sin(math.pi * Xl) * np.sin(math.pi * Yl) * np.sin(math.pi * Zl)

# Exchange function using blocking Send/Recv. Use ordering: if rank < nbr then send first else recv first to avoid deadlock.

def exchange_ghosts(u, cart):
    """
    Exchange ghost layers in all 6 directions for a 3D array u with shape (nz+2, ny+2, nx+2)
    using blocking Sendrecv. Dirichlet boundaries (no neighbor) remain 0.
    
    u: ndarray with ghost layers
    cart: MPI Cartesian communicator
    """
    t0 = time()


    nz, ny, nx = u.shape
    nz -= 2
    ny -= 2
    nx -= 2
    rank = cart.Get_rank()

    # X-direction
    src_x, dst_x = cart.Shift(0, 1)

    # Send right, receive left
    if dst_x != MPI.PROC_NULL or src_x != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[1:-1,1:-1,-2])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=dst_x, sendtag=101,
                    recvbuf=recvbuf, source=src_x, recvtag=101)
        if src_x != MPI.PROC_NULL:
            u[1:-1,1:-1,0] = recvbuf
        else:
            u[1:-1,1:-1,0] = 0.0
    
    # Send left, receive right
    if src_x != MPI.PROC_NULL or dst_x != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[1:-1,1:-1,1])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=src_x, sendtag=102,
                    recvbuf=recvbuf, source=dst_x, recvtag=102)
        if dst_x != MPI.PROC_NULL:
            u[1:-1,1:-1,-1] = recvbuf
        else:
            u[1:-1,1:-1,-1] = 0.0

    # Y-direction
    src_y, dst_y = cart.Shift(1, 1)

    # Send right, receive left
    if dst_y != MPI.PROC_NULL or src_y != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[1:-1,-2, 1:-1])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=dst_y, sendtag=201,
                    recvbuf=recvbuf, source=src_y, recvtag=201)
        if src_y != MPI.PROC_NULL:
            u[1:-1,0, 1:-1] = recvbuf
        else:
            u[1:-1,0, 1:-1] = 0.0
    
    # Send left, receive right
    if src_y != MPI.PROC_NULL or dst_y != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[1:-1,1, 1:-1])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=src_y, sendtag=202,
                    recvbuf=recvbuf, source=dst_y, recvtag=202)
        if dst_y != MPI.PROC_NULL:
            u[1:-1,-1, 1:-1] = recvbuf
        else:
            u[1:-1,-1, 1:-1] = 0.0

    # Z-direction
    src_z, dst_z = cart.Shift(2, 1)

    # Send right, receive left
    if dst_z != MPI.PROC_NULL or src_z != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[-2, 1:-1, 1:-1])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=dst_z, sendtag=301,
                    recvbuf=recvbuf, source=src_z, recvtag=301)
        if src_z != MPI.PROC_NULL:
            u[0, 1:-1,1:-1] = recvbuf
        else:
            u[0, 1:-1,1:-1] = 0.0

    # Send left, receive right
    if src_z != MPI.PROC_NULL or dst_z != MPI.PROC_NULL:
        sendbuf = np.ascontiguousarray(u[1, 1:-1,1:-1])
        recvbuf = np.empty_like(sendbuf)
        comm.Sendrecv(sendbuf=sendbuf, dest=src_z, sendtag=302,
                    recvbuf=recvbuf, source=dst_z, recvtag=302)
        if dst_z != MPI.PROC_NULL:
            u[-1, 1:-1,1:-1] = recvbuf
        else:
            u[-1, 1:-1,1:-1] = 0.0

    t1 = time()
    return t1 - t0


# Main iteration loop
max_iter = options.iter
tol = options.tolerance
omega = options.omega

# For logging
local_compute_time = 0.0
local_comm_time = 0.0
local_total_time = 0.0

# storage for convergence history (only on rank 0 we'll keep full history printed)
if rank == 0:
    history = []

# start
comm.Barrier()
start_all = time()

# initial exchange to fill ghost with zeros or neighbor data
comm_time = exchange_ghosts(u_old, cart)
local_comm_time += comm_time

iter_count = 0
for it in range(max_iter):
    iter_t0 = time()
    # compute update using u_old -> u_new (interior only)
    compute_t0 = time()
    # interior indices
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    c = 1.0 / 6.0
    # weighted Jacobi step: same formula used in sequential code (reads from u_old)
    u_new[interior] = omega * c * (
        u_old[0:-2, 1:-1, 1:-1] + u_old[2:, 1:-1, 1:-1] + u_old[1:-1, 0:-2, 1:-1] + u_old[1:-1, 2:, 1:-1] + u_old[1:-1, 1:-1, 0:-2] + u_old[1:-1, 1:-1, 2:] + h2 * f_local
    )
    compute_t1 = time()
    local_compute_time += (compute_t1 - compute_t0)

    # communication: update ghost cells from neighbors using the newly computed buffer if required by algorithm
    # here, we will swap roles: the next iteration will read from u_new, so exchange ghosts for u_new
    comm_t0 = time()
    comm_time = exchange_ghosts(u_new, cart)
    local_comm_time += comm_time
    comm_t1 = time()

    # compute local sums for diff_step (use u_new - u_old)
    local_sum_sq = float(np.sum((u_new[1:-1,1:-1,1:-1] - u_old[1:-1,1:-1,1:-1])**2))
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    diff_step = math.sqrt(global_sum_sq) / (N ** 3)

    # compute diff_true: compare u_new interior with u_true_local
    local_sum_sq_true = float(np.sum((u_new[1:-1,1:-1,1:-1] - u_true_local)**2))
    global_sum_sq_true = comm.allreduce(local_sum_sq_true, op=MPI.SUM)
    diff_true = math.sqrt(global_sum_sq_true) / (N ** 3)

    # swap buffers: prepare for next iteration (we want u_old = u_new)
    u_old, u_new = u_new, u_old
    iter_count += 1
    iter_t1 = time()
    local_total_time += (iter_t1 - iter_t0)

    # logging by rank 0
    # compute local contribution on every rank (force Python float)
    local_norm_u_true_sq = float(np.sum(u_true_local ** 2))

    # reduce across all ranks (all ranks must call this)
    global_norm_u_true_sq = comm.allreduce(local_norm_u_true_sq, op=MPI.SUM)

    # only rank 0 uses it for logging/heuristics
    if rank == 0:
        global_norm_u_true = math.sqrt(global_norm_u_true_sq)
        expected_truncation = (h ** 2) * global_norm_u_true
        history.append((it + 1, diff_step, diff_true, expected_truncation))

    # broadcast convergence decision (all ranks need to know)
    stop = diff_step < tol
    stop = comm.bcast(stop, root=0)
    if stop:
        break
if rank == 0:
    print(history[-1])

# end iterations
end_all = time()

# Gather timings
total_compute = comm.allreduce(local_compute_time, op=MPI.MAX)
total_comm = comm.allreduce(local_comm_time, op=MPI.MAX)
wall_time = end_all - start_all
max_wall = comm.allreduce(wall_time, op=MPI.MAX)

if rank == 0:
    print(f"Iterations run: {iter_count}")
    print(f"Max compute time across ranks: {total_compute:.6f}s")
    print(f"Max comm   time across ranks: {total_comm:.6f}s")
    print(f"Max wall   time across ranks: {max_wall:.6f}s")

comm.Barrier()

# Assemble global u on rank 0
# Each rank sends its interior block (shape local_nz, local_ny, local_nx) to rank 0
local_interior = u_old[1:-1, 1:-1, 1:-1].copy()  # u_old holds last computed solution
print(f"Rank {rank} interior mean: {np.mean(local_interior):.6e}")

if rank == 0:
    u_global = np.empty((N, N, N), dtype=np.float64)
    # place own block
    z0 = local_z0
    z1 = z0 + local_nz
    y0 = local_y0
    y1 = y0 + local_ny
    x0 = local_x0
    x1 = x0 + local_nx
    u_global[z0:z1, y0:y1, x0:x1] = local_interior
    # receive from others
    for r in range(1, size):
        # get coords of rank r
        rc = cart.Get_coords(r)
        rx, ry, rz = rc
        # careful: earlier coords used (ix,iy,iz) mapping; cart coords ordering matches Create_cart
        # compute source starts and counts using same split logic
        sx = nx_starts[rx]
        sy = ny_starts[ry]
        sz = nz_starts[rz]
        cx = nx_counts[rx]
        cy = ny_counts[ry]
        cz = nz_counts[rz]
        buf = np.empty((cz, cy, cx), dtype=np.float64)
        comm.Recv(buf, source=r, tag=99)
        u_global[sz:sz+cz, sy:sy+cy, sx:sx+cx] = buf
    # save
    np.save(options.output, u_global)
    print(f"Saved assembled solution to {options.output}")
else:
    comm.Send(local_interior, dest=0, tag=99)

comm.Barrier()
if rank == 0:
    print("Done.")

