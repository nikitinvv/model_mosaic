# model_mosaic

Simulate an extended-FOV **mosaic X-ray phase-contrast tomography scan** from a
reconstructed volume.  The simulation is a full two-physics forward model —
first **tomographic projection** through the object via a non-uniform-FFT Radon
transform, then **near-field wave propagation** of the resulting complex
transmission function through free space via the **Fresnel transform** — so the
output isn't just a Radon sinogram but the actual intensity that would be
recorded on a detector in a holographic / propagation-based phase-contrast
experiment:

```
     δ (refractive index)  ──R──► proj = R(δ)              (USFFT Radon)
                                    │
                                    ▼
              ψ = exp(1j · (x22 + 1j · β))                  (weak absorption)
                                    │
                                    ▼
              data = |D_prop(ψ)|²                           (Fresnel propagation)
```

Pipeline: layout the tile grid → upsample the input volume → forward-project
via USFFT Radon → propagate via Fresnel → slice the propagated data into
per-tile HDF5 files ready for downstream stitching.

![Mosaic scan schematic](mosaic_schematic.png)

*Above:* `step0_schematic.py --ups 1` — 360° extended-FOV mosaic covering a
⌀28.67 mm sample with a 4.5 × 3.4 mm detector.  4 x-tiles × 9 z-stacks × 2058
angles = 74 088 projections; virtual FOV ⌀33.96 mm.  Left = top view of tile 0
(overlapping its 180° mirror through the axis) and tile 1 (annulus).  Middle =
side view of the 9 z-stacks with uniform 280 µm overlaps.  Right = virtual
sample coverage after a full 360° rotation.  All dimensions scale with `--ups`
while the physical measurements stay constant.

Everything is **CLI-driven**, **MPI-parallel** (`mpi4py` required), and
**GPU-accelerated** via cupy.  The USFFT / Fresnel machinery is vendored under
[`processing/`](processing/); the HDF5 I/O layer lives in [`iohdf5/`](iohdf5/).

---

## Repository layout

```
mosaic_modeling/
├── processing/                     ← compute modules
│   ├── tomo.py                     Radon (GPU-only Tomo)
│   ├── tomo_large.py               Radon (host-chunked TomoLarge)
│   ├── propagation.py              Fresnel propagation
│   └── kernels.py                  cupy elementwise/raw kernels
├── iohdf5/                         ← HDF5 I/O helpers
│   ├── dxchange_hdf5_chunks.py     tomo_initx / tomo_writex / tomo_readx
│   │                                 (spawn-context Pool; CUDA-safe)
│   └── h5_vchunks.py               SHM buffer + iter_vchunks + init+bcast
├── utils.py                        MPI wiring (COMM/RANK/SIZE/barrier/rprint)
├── step0_schematic.py              plan tile layout
├── step00_upsample_extract.py             TIFF → init.h5   (optional prep step)
├── step1_upsample.py               init → big{UPS}x
├── step2_radon.py                  big → proj.h5     (GPU-only Tomo)
├── step2_radon_large.py            big → proj.h5     (host-chunked TomoLarge)
├── step3_fresnel.py                proj → data.h5    (shared regardless of variant)
├── step4_extract.py                data.h5 → mosaic_h5/{z}_{x}.h5
├── test_h5_buffer_io.py            I/O benchmark
└── {tomo,polaris}_{run,test_h5}.sh launchers
```

The pipeline files at the top level are the entry points.  `processing/`
and `iohdf5/` are proper Python packages (each has an `__init__.py`) —
import as `from processing.tomo import Tomo`, `from iohdf5.h5_vchunks import
alloc_shm`, etc.

---

## HDF5 storage: VDS master + banks

Every large dataset is written as a **top-level Virtual DataSet (VDS)
master file + N per-super-chunk bank files** (the pattern from
[test_h5_buffer_io.py](test_h5_buffer_io.py), which itself adapts the
[doe-maxiv](https://gitlab.com/tomograms/doe-maxiv) `tomo_writex` scheme).

For a dataset written with `--nbanks N` and vchunk shape `(C0, C1, C2)`:

```
{path}/big1x.h5                          ← VDS master (rank 0 creates)
{path}/big1x/
    big1x_data_000000.h5                 ← bank file 0
    big1x_data_000001.h5                 ← bank file 1
    ...  (nvchunks × nbanks bank files)
```

Reads open the master and let h5py resolve the VDS transparently.
Writes go through `iohdf5.dxchange_hdf5_chunks.tomo_writex`, which fans
each vchunk buffer across `nbanks` bank files in parallel POSIX
processes.  **No MPI-IO** — each rank writes only to bank files it owns.

**Rank sharding is round-robin over vchunks** (`ivchunks[RANK::SIZE]`).
Rank 0 alone calls `tomo_initx` (VDS master + all empty bank files),
broadcasts the context, and all ranks then iterate their subset.

`iohdf5.dxchange_hdf5_chunks` uses a **spawn-context multiprocessing.Pool**
cached at module level, so it's safe to call after cupy has initialised
CUDA (fork would inherit the CUDA context and break in the child).

---

## Installation

Compiled dependencies: **cupy** (matching your CUDA toolkit), **mpi4py**
(MPI runtime), and **h5py** (POSIX, no need for MPI-IO build).

```bash
conda create -n mosaic python=3.12
conda activate mosaic

# GPU stack — pick the cupy wheel that matches your CUDA runtime.
# Check with: python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
# 12xxx → cupy-cuda12x    13xxx → cupy-cuda13x
pip install cupy-cuda12x

# MPI runtime + mpi4py.
conda install -c conda-forge "openmpi>=4" mpi4py

# HDF5 (plain, non-MPI build is fine — the pipeline uses POSIX writes).
conda install -c conda-forge h5py

# Everything else.
conda install -c conda-forge tifffile matplotlib scipy numpy
```

**Don't install two cupy wheels at once.**  If `import cupy` warns about
`cupy` + `cupy-cuda13x` both present:
```bash
pip uninstall -y cupy cupy-cuda12x cupy-cuda13x
pip install cupy-cuda12x   # or cupy-cuda13x for CUDA 13
```

**On Polaris (ALCF)** use the site's Cray MPICH:
```bash
module load conda; conda activate base
pip install cupy-cuda12x
CC=cc MPICC=cc pip install --no-cache-dir --no-binary=mpi4py mpi4py
pip install h5py
```

Verify:
```bash
python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.getDeviceCount(), 'GPUs')"
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
python -c "import h5py; print(h5py.__version__)"
```

---

## Pipeline overview

```
   ┌───────────────────┐
   │  init.h5  (input) │  /exchange/data (2560, 2744, 2744) f32
   │                   │  (produced by step00_upsample_extract.py, or dropped in)
   └────────┬──────────┘
            │
   step0_schematic.py       ─ plan tile layout, save PNG + positions.txt
            │
   step1_upsample.py        ─ init → big{UPS}x           (VDS + banks)
            │
   step2_radon.py           ─ big{UPS}x → model_big{UPS}x/proj.h5
        OR                       (GPU-only Tomo)
   step2_radon_large.py     ─ same, but host-chunked TomoLarge for UPS ≥ 8
            │
   step3_fresnel.py         ─ proj.h5 → data.h5           (Fresnel; shared)
            │
   step4_extract.py         ─ data.h5 → mosaic_h5/{z}_{x}.h5
```

Every large dataset (init, big, proj, data) is a VDS+banks store.  All
paths live under a single `--path` root; multiple UPS values coexist
because output filenames are UPS-tagged:

```
<path>/
  init.h5                         ← input VDS master + init/*_data_*.h5 banks
  big{UPS}x.h5                    ← step1 output
  big{UPS}x/*_data_*.h5           ←   bank files
  model_big{UPS}x/
    proj.h5                       ← step2 output   (NTHETA, NZ, N) f32
    proj/*_data_*.h5              ←   bank files
    data.h5                       ← step3 output   (NTHETA, NZ, N) f32
    data/*_data_*.h5              ←   bank files
  mosaic_h5/
    0_0.h5  0_1.h5  …             ← step4 per-tile HDF5 (regular chunked)
  mosaic_schematic{UPS}.png       ← step0 layout figure
  mosaic_positions{UPS}.txt       ← step0 x tile origins (px)
```

Shared filesystem (Lustre / GPFS / NFS) assumed — `--path` should
resolve to the same location on every node.

**Lustre striping** — set striping on every dir that will hold bank
files, so newly-created files inherit it:

```bash
for d in <path> <path>/init <path>/big${UPS}x \
         <path>/model_big${UPS}x <path>/model_big${UPS}x/proj \
         <path>/model_big${UPS}x/data; do
    lfs setstripe -c -1 -S 4M "$d"
done
```

`polaris_run.sh` does this automatically.

---

## Quick start

**Small scale** — UPS 1–4, GPU-only Tomo:
```bash
UPS=2
PATH_DATA=/data2/brain_sym_mosaic
N_GPUS=4

python step0_schematic.py --ups $UPS --path $PATH_DATA

mpirun -n $N_GPUS set_affinity_gpu.sh \
    python step1_upsample.py --ups $UPS --path $PATH_DATA

mpirun -n $N_GPUS set_affinity_gpu.sh \
    python step2_radon.py --ups $UPS --path $PATH_DATA \
        --nzchunk 32

mpirun -n $N_GPUS set_affinity_gpu.sh \
    python step3_fresnel.py --ups $UPS --path $PATH_DATA \
        --npropchunk 8

mpirun -n $N_GPUS python step4_extract.py --ups $UPS --path $PATH_DATA
```

Or use the wrapper: `bash tomo_run.sh` (edit the USER KNOBS at the top).

**Large scale** — UPS ≥ 8, host-chunked TomoLarge:
```bash
mpirun -n $N_GPUS set_affinity_gpu.sh \
    python step2_radon_large.py --ups 8 --path $PATH_DATA \
        --nzchunk 1 \
        --chunk-n 686 --chunk-theta 343 --chunk-xy 686
```
(step3_fresnel and step4_extract are identical either way.)

**On Polaris**: `qsub polaris_run.sh` — sets up Lustre striping,
disables HDF5 file locking (essential on parallel FS), and uses
`set_affinity_gpu_polaris.sh` for GPU affinity based on `PMI_LOCAL_RANK`.

---

## Choosing step2_radon vs step2_radon_large

|                 | `step2_radon.py`                   | `step2_radon_large.py`                     |
|-----------------|------------------------------------|--------------------------------------------|
| Tomo backend    | GPU-only (`Tomo`)                  | Host-chunked (`TomoLarge`)                 |
| Peak GPU memory | `NZCHUNK · (2N)² · 8 B`             | `~CHUNK_N²` (much smaller)                 |
| Peak host memory| small (chunk read + result strip)  | `NZCHUNK · (2N)² · 8 B` (moved to host)     |
| Best for        | `UPS ≤ 4` on a 40 GB GPU           | `UPS ≥ 8`, or any `N` too big for the GPU  |
| Extra CLI       | —                                  | `--chunk-n`, `--chunk-theta`, `--chunk-xy` |

Same output format (VDS + banks proj.h5) — step3_fresnel and step4_extract
don't care which one produced proj.h5.

Chunk sizes for the large variant must divide the sizes they slice into:
- `--chunk-n`     divides `N` and `2N`
- `--chunk-theta` divides `NTHETA` (= `3·N/4` by default)
- `--chunk-xy`    divides `2N`

Defaults (686 / 343 / 686) are divisors of `2744` and `2058`, so they work
for **any** integer `--ups ≥ 1`.

---

## Vchunks (super-chunks) + nbanks

Each step's write path is controlled by two knobs:

- `--nbanks N` — bank files per super-chunk (parallel POSIX writers per
  `tomo_writex` call).  Also = worker-process count.  Typical: 4–8.
- `--{init,big,proj,data}-vchunks C0 C1 C2` — shape of the RAM buffer
  per rank.  A vchunk holds one super-chunk of the output; each rank
  fills its own buffer over multiple compute calls, then `tomo_writex`
  fans it across `nbanks` bank files.

**RAM buffer per rank** = `C0 · C1 · C2 · 4` bytes.  Total peak RAM per
node ≈ `ranks_per_node × sum(active vchunks)`.  On a 512 GB node with
4 ranks/GPU that's ~100 GB budget per rank — plenty for UPS 1–4, needs
manual overrides for UPS 8.

Defaults per stage:
| step             | default vchunk shape                 |
|------------------|--------------------------------------|
| step00_upsample_extract | `(CHUNK_Z, OUT_NYX, OUT_NYX)`  |
| step1_upsample   | `(8·UPS, OUT_NYX, OUT_NYX)`          |
| step2_radon(_large) | `(NTHETA, 8·NZCHUNK, N)`          |
| step3_fresnel    | `(8·NPROPCHUNK, NZ, N)`              |

Constraint: `proj-vchunks C0 = NTHETA` (radon writes all θ at once) and
`data-vchunks C1×C2 = NZ×N` (fresnel writes full planes).

---

## Per-stage CLI reference (most-used flags)

Full options: `python step<N>_<name>.py --help`.

### step0_schematic.py
| flag | default | notes |
|------|---------|-------|
| `--ups`  | `1` | drives detector/sample/pixel scaling |
| `--path` | `/data2/brain_sym_mosaic` | where the PNG + `mosaic_positions{UPS}.txt` land |

### step1_upsample.py
| flag | default | notes |
|------|---------|-------|
| `--ups`      | `4` | upsample factor in every axis |
| `--path`     | `/data2/brain_sym_mosaic` | reads `init.h5`, writes `big{UPS}x.h5` |
| `--n-read`   | `2` | background input prefetchers per rank |
| `--nbanks`   | `8` | bank files per super-chunk |
| `--big-vchunks` | `8·UPS OUT_NYX OUT_NYX` | RAM buffer shape (`C0 % UPS == 0` required) |

### step2_radon.py / step2_radon_large.py
| flag | default | notes |
|------|---------|-------|
| `--ups`     | `2` / `8` | matches step1 |
| `--path`    | `/data2/brain_sym_mosaic` | reads `big{UPS}x.h5`, writes `proj.h5` |
| `--ntheta`  | `3·N/4` | override for cheaper prototyping |
| `--nzchunk` | `8` / `1` | z-slices per Radon call (memory ↔ speed) |
| `--nbanks`  | `8` | bank files per super-chunk |
| `--proj-vchunks` | `NTHETA 8·NZCHUNK N` | RAM buffer shape |
| `--chunk-{n,theta,xy}` | `686/343/686` | *large only* — must divide `N`/`NTHETA`/`2N` |

### step3_fresnel.py
| flag | default | notes |
|------|---------|-------|
| `--ups`         | `2` | matches step2 |
| `--path`        | `/data2/brain_sym_mosaic` | reads `proj.h5`, writes `data.h5` |
| `--npropchunk`  | `8` | angles per Fresnel batch |
| `--nbanks`      | `8` | bank files per super-chunk |
| `--data-vchunks`| `8·NPROPCHUNK NZ N` | RAM buffer shape |
| `--beta-ratio`  | `100` | weak-absorption ratio (β = phase/BETA_RATIO) |
| `--phase-scale` | `1.0` | amplify phase to make Fresnel fringes visible |
| `--distance`    | `1.0 m` | sample→detector distance (parallel beam) |
| `--voxelsize`   | `1.4 µm` | voxel = detector pixel |

### step4_extract.py
| flag | default | notes |
|------|---------|-------|
| `--ups`  | `2` | matches step2/3 |
| `--path` | `/data2/brain_sym_mosaic` | reads `data.h5`, writes `mosaic_h5/*.h5` |

---

## Resuming / partial runs

- **Only radon**: run `step2_radon.py` alone; step3 can be run later.
- **Only fresnel** (`proj.h5` already on disk): run `step3_fresnel.py` directly.
- **Change UPS mid-experiment**: everything is UPS-tagged
  (`big{UPS}x.h5`, `model_big{UPS}x/`, `mosaic_schematic{UPS}.png`), so
  runs at different UPS values coexist under the same `--path` without
  collision.

---

## GPU affinity

All GPU-using steps rely on the launcher (`set_affinity_gpu.sh`) to set
`CUDA_VISIBLE_DEVICES` per local rank so cupy sees exactly one GPU per
process.  Do **not** call the scripts under `mpirun` without the
wrapper — otherwise every rank on a node will fight for the same device.

```bash
mpirun -n <NGPU_TOTAL> set_affinity_gpu.sh python step1_upsample.py …
```

On Polaris use [set_affinity_gpu_polaris.sh](set_affinity_gpu_polaris.sh)
(reads `PMI_LOCAL_RANK` for NUMA-correct GPU assignment).

---

## Preparing init.h5

If you don't already have an `init.h5`, [step00_upsample_extract.py](step00_upsample_extract.py)
crops + masks + soft-tapers a 3-D multi-page TIFF from a reconstruction and
writes the result as a VDS+banks store at `{path}/init.h5`:

```bash
mpirun -n 4 set_affinity_gpu.sh python step00_upsample_extract.py \
    --src /local/tomodata3/vnikitin/…/rec_obj_real/0096.tiff \
    --path /data2/brain_sym_mosaic
```

Optional — not part of the numbered pipeline.  If you already have a proper
`init.h5` with `/exchange/data` of shape `(OUT_NZ, OUT_NYX, OUT_NYX)`
float32, skip it.

---

## I/O benchmark

[test_h5_buffer_io.py](test_h5_buffer_io.py) exercises the same
VDS+banks + `tomo_writex` path the pipeline uses, without any compute,
across three stages (seed, upsample-shaped, radon-shaped, fresnel-shaped).
Useful for tuning `nbanks` / `vchunks` on a new machine.

```bash
bash tomo_test_h5.sh        # local machine
qsub polaris_test_h5.sh     # Polaris
```

Prints aggregate throughput (`sum(bytes) / max(rank elapsed)`) plus
per-rank spread after each stage.
