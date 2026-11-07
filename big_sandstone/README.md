# Big sandstone, scans 082–103

22 off-axis 360° scans, 11 heights × 2 columns (18001 projections,
2748 × 8192 each), stitched and reconstructed into one volume. It's the same
pipeline as `../marmoset`.

```bash
cd ~/mosaic_modeling/big_sandstone
bash run_pipeline_sandstone.sh        # on tomo2, uses all 11 GPUs
```

Output in `/data2/19BM/2026-11-Boyer-1026094_rec/big_sandstone_full_082_103_stitched/model_big1x/`:

| file | step | size |
|---|---|---|
| `stitched.h5` | 1: dark/flat, dezinger, rings, stitch, fold | 8999 × 23936 × 20850, 18 TB |
| `paganin.h5` | 2: Paganin | same, 18 TB |
| `rec.h5` | 3: FBP | 23921 × 20850 × 20850, 42 TB |

Check `df -h /data2` first. The run needs about 78 TB.

## Settings

At the top of `run_pipeline_sandstone.sh`:

- **Tile table:** the position of each scan. The outer column is at 3176 px,
  and the rows are about 2118 px apart.
- **Bright ratio:** 0.617 for the inner column, 1 for the outer.
- **Axis:** 944 (tomocupy convention).
- **Angles:** linspace(0, 359.9118), because the files store zeros.
- **Physics:** 70 keV, 1.889 µm pixel, 0.6 m distance.
- **Reconstruction:** Paganin α = 1e-4, ramp filter.

## Status

Not run yet. Start with a 64-slice test:

```bash
ROWS="2000 2064" OUT=/data2/19BM/2026-11-Boyer-1026094_rec/test_sandstone \
  HOSTS=tomo5:4 NGPUS=4 bash run_pipeline_sandstone.sh
```
