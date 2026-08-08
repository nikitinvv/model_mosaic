#!/bin/bash
#PBS -A 14238
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:30:00
#PBS -q debug-scaling
#PBS -N mosaic_buf_io_mn
#PBS -j oe
#
# Multi-node throughput scaling test for test_h5_buffer_io.py.
#
# test_h5_buffer_io.py is single-node by design (multiprocessing, no MPI),
# so we cannot MPI-shard a single dataset across nodes without rewriting it.
# Instead, we launch ONE independent Python instance per node in parallel,
# each writing to its OWN subdirectory under $PATH_DATA/n_<hostname>/.
# The aggregate throughput to Eagle is then sum(bytes) / max(elapsed).
#
# What this tells you:
#   • If per-node bytes/s stays ~3 GB/s and total scales linearly with
#     nodes, the single-node ceiling is client-side (Slingshot / Lustre
#     client per node) — MPI-sharding the real pipeline would help.
#   • If per-node bytes/s drops as nodes are added, Eagle's back end is
#     saturated (OST / MDT contention) — MPI-sharding won't help beyond
#     that ceiling.
#
# Change `select=N` above to sweep node counts (1, 2, 4, 10, ...).

# ================== USER KNOBS ==================
UPS=1
# Per-node output dirs go under this root.  Each node gets its own subdir
# to avoid Lustre metadata contention on shared files.
PATH_ROOT=/eagle/APS_IRI/vnikitin/iotest_buf_ups${UPS}_mn

NBANKS=8
NTASKS=8

INIT_VCHUNKS="32 2744 2744"
BIG_VCHUNKS="$((32*UPS)) $((2744*UPS)) $((2744*UPS))"
PROJ_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"
DATA_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"
# ================================================

NNODES=$(wc -l < "$PBS_NODEFILE")
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "Jobid: $PBS_JOBID"
echo "Nodes: $NNODES"
echo "$(cat "$PBS_NODEFILE")"

module use /soft/modulefiles
module load conda
conda activate base
CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')

CONDA_ENV_CANDIDATES=(holotomocupy)
VENV_CANDIDATES=(
    "${HOME}/venvs/vvnikitin/bin/activate"
    "${HOME}/venvs/${CONDA_NAME}/bin/activate"
    "/home/vvnikitin/venvs/vvnikitin/bin/activate"
    "/home/vvnikitin/venvs/${CONDA_NAME}/bin/activate"
)
_env_activated=0
for e in "${CONDA_ENV_CANDIDATES[@]}"; do
    if [[ -d "${HOME}/.conda/envs/${e}" ]]; then
        echo "activating conda env: ${e}"
        conda activate "${e}"
        _env_activated=1
        break
    fi
done
if (( ! _env_activated )); then
    for v in "${VENV_CANDIDATES[@]}"; do
        if [[ -f "$v" ]]; then
            echo "activating venv: $v"
            source "$v"
            _env_activated=1
            break
        fi
    done
fi
if (( ! _env_activated )); then
    echo "WARNING: no project env activated; using base conda at ${CONDA_PREFIX}" >&2
fi

cd "${SCRIPT_DIR}"

mkdir -p "${PATH_ROOT}"
lfs setstripe -c -1 -S 4M "${PATH_ROOT}" 2>/dev/null || true

echo "=== UPS=${UPS}  ROOT=${PATH_ROOT}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NODES=${NNODES} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"

LOGDIR="${PATH_ROOT}/logs"
mkdir -p "${LOGDIR}"

# One Python instance per node.  --ppn 1 puts one rank on each node; the
# rank's $PMI_RANK is unique across the whole job, so we use it to name
# each node's private data + log directory.  --cpu-bind none because the
# Python multiprocessing pool inside will fan across the node's cores.
T_WALL_START=$(date +%s.%N)

mpiexec -n "${NNODES}" --ppn 1 --cpu-bind none bash -c '
    R=${PMI_RANK:-${PMIX_RANK:-0}}
    NODE_DIR="'"${PATH_ROOT}"'/n_${R}_$(hostname -s)"
    LOG="'"${LOGDIR}"'/rank${R}_$(hostname -s).log"
    mkdir -p "${NODE_DIR}"
    lfs setstripe -c -1 -S 4M "${NODE_DIR}" 2>/dev/null || true

    echo "[rank ${R} $(hostname -s)] starting → ${NODE_DIR}" >&2

    python "'"${SCRIPT_DIR}"'/test_h5_buffer_io.py" \
        --path "${NODE_DIR}" --ups '"${UPS}"' \
        --nbanks '"${NBANKS}"' --ntasks '"${NTASKS}"' \
        --init-vchunks '"${INIT_VCHUNKS}"' \
        --big-vchunks  '"${BIG_VCHUNKS}"' \
        --proj-vchunks '"${PROJ_VCHUNKS}"' \
        --data-vchunks '"${DATA_VCHUNKS}"' \
        > "${LOG}" 2>&1
    RC=$?
    echo "[rank ${R} $(hostname -s)] done rc=${RC}" >&2
    exit ${RC}
'
MPI_RC=$?

T_WALL_END=$(date +%s.%N)
T_WALL=$(awk "BEGIN{printf \"%.2f\", ${T_WALL_END} - ${T_WALL_START}}")

echo ""
echo "=== MPI exit=${MPI_RC}   wall=${T_WALL}s ==="
echo ""

# ---------- Aggregate per-node reports into one summary --------------------
# Each per-node log has lines like:
#   init.h5 seed:       26.49s   (2.71 GB/s)
#   stage 1  read :     ...
#   stage 1  write:     ...
#   radon    read :     ...
#   radon    write:     ...
#   fresnel  read :     ...
#   fresnel  write:     ...
# We parse them, sum bytes across ranks, and divide by max wall for aggregate.

python3 - "${LOGDIR}" "${T_WALL}" <<'PY'
import glob, os, re, sys

logdir, wall = sys.argv[1], float(sys.argv[2])
# stage_label -> [(seconds, gb_per_sec, rank_hint), ...]
STAGES = [
    ("init.h5 seed",   r"init\.h5 seed:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("stage 1 read",   r"stage 1\s+read\s*:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("stage 1 write",  r"stage 1\s+write:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("radon read",     r"radon\s+read\s*:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("radon write",    r"radon\s+write:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("fresnel read",   r"fresnel\s+read\s*:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
    ("fresnel write",  r"fresnel\s+write:\s+([\d.]+)s\s+\(([\d.]+)\s*([KMGT]?B)/s\)"),
]
UNIT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

logs = sorted(glob.glob(os.path.join(logdir, "rank*.log")))
n = len(logs)
if n == 0:
    print(f"[aggregate] no per-node logs in {logdir}")
    sys.exit(0)

print(f"[aggregate] {n} per-node logs")
print("")
print(f"{'stage':16s}  {'per-node avg':>14s}  {'per-node min':>14s}  "
      f"{'per-node max':>14s}  {'aggregate':>14s}")
print("-" * 82)

for label, pat in STAGES:
    per_node_bps = []
    per_node_secs = []
    for lg in logs:
        with open(lg) as f:
            txt = f.read()
        m = re.search(pat, txt)
        if not m:
            continue
        secs = float(m.group(1))
        val  = float(m.group(2))
        unit = m.group(3)
        per_node_bps.append(val * UNIT[unit])
        per_node_secs.append(secs)
    if not per_node_bps:
        continue

    def _h(bps):
        for u in ("B", "KB", "MB", "GB", "TB"):
            if bps < 1024: return f"{bps:.2f} {u}/s"
            bps /= 1024
        return f"{bps:.2f} PB/s"

    avg = sum(per_node_bps) / len(per_node_bps)
    mn  = min(per_node_bps)
    mx  = max(per_node_bps)
    # aggregate = sum(bytes) / max(secs) — bytes = bps * secs per node
    total_bytes = sum(b * s for b, s in zip(per_node_bps, per_node_secs))
    slowest = max(per_node_secs)
    agg = total_bytes / max(slowest, 1e-9)
    print(f"{label:16s}  {_h(avg):>14s}  {_h(mn):>14s}  "
          f"{_h(mx):>14s}  {_h(agg):>14s}")

print("")
print(f"wall (whole mpiexec): {wall:.2f}s   nodes: {n}")
print(f"per-node logs: {logdir}/rank*.log")
PY
