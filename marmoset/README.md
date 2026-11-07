# Marmoset brain, scans 180–186

Seven vertical 360° off-axis scans (12801 projections, 2748 × 7168 each),
stitched and reconstructed into one volume.

```bash
cd ~/mosaic_modeling/marmoset
bash run_pipeline_180_186.sh          # on tomo2, uses all 11 GPUs, ~3 h
```

Output in `/data2/19BM/2026-11-Boyer-1026094_rec/Marmoset_test_1612mm_vertical_full_180_186_stitched/model_big1x/`:

| file | step | size |
|---|---|---|
| `stitched.h5` | 1: dark/flat, dezinger, rings, stitch, fold | 6400 × 15072 × 12586, 4.9 TB |
| `paganin.h5` | 2: Paganin | same, 4.9 TB |
| `rec.h5` | 3: FBP | 15054 × 12586 × 12586, 9.5 TB |

## Settings

At the top of `run_pipeline_180_186.sh`:

- axis 876 (tomocupy convention)
- 2051 rows between scans (measured with `../find_overlap.py`)
- 70 keV, 1.9 µm pixel, 1.62 m distance
- Paganin α = 8e-5, ramp filter

## What each step does

1. **Stitch + fold** (`../stitch_fold.py`), per scan:
   - dezinger 5
   - flat field interpolated per projection, from the scan's own flats to
     the next scan's flats
   - fw ring removal

   Then it stitches the scans and folds 360° → 180°. Both use a smooth
   9th-order blend across the overlaps.
2. **Paganin** (`../step7_paganin_large.py`), with tomocupy's edge padding.
3. **FBP** (`../step8_fbp_fold.py`), with tomocupy's Fourier method.

## Examples

```bash
STEPS="2 3" ALPHA=1e-4 bash run_pipeline_180_186.sh      # redo steps 2 and 3
ROWS="6464 6528" OUT=/data2/19BM/2026-11-Boyer-1026094_rec/test \
  HOSTS=tomo5:4 NGPUS=4 bash run_pipeline_180_186.sh       # 64-slice test
```
