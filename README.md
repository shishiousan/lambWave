# Lamb half-space surface reference — revised implementation

Reference implementation for two-dimensional **plane-strain isotropic elastodynamics** in a homogeneous half-space. A vertical line force acts at the free surface, `(x,y)=(0,0)`; the half-space occupies `y>=0`. `x` is horizontal and positive `y` points downward. The main output is **surface displacements** `(ux,uy)` in metres, not velocity, stress, or internal (`y>0`) displacement.

This release supersedes the earlier `lamb_noringing` package. It retains the **non-FFT, time-domain unit-step Green function** for abrupt forces, adds a careful Ricker convolution, and removes VTKHDF as a normal output dependency. See [THEORY_AND_ACCURACY.md](THEORY_AND_ACCURACY.md) for the equations, numerical approximations, and known singularities.

## Installation

Python 3.10+ recommended:

```bash
python -m pip install numpy scipy matplotlib
python verify.py
```

No `vtk` package is required. `matplotlib` is optional if the `--no-plots` switch is used.

## Recommended workflow

**1. Inspect the loading only.** No Green-function tabulation or analytical solution is generated:

```bash
python run.py --preset three_step --inspect-only --out output/preview_step
python run.py --preset ricker --inspect-only --out output/preview_ricker
python run.py --preset rectangle --rectangle-start 0.05 --rectangle-end 0.1 \
              --inspect-only --out output/preview_rectangle
```

Each preview produces:

- `load_time.csv`: `time_s,force_N_per_m` (source waveform sampled for plotting)
- `load_spectrum.csv`: `frequency_hz,angular_frequency_rad_per_s,real_N_s_per_m,imag_N_s_per_m,magnitude_N_s_per_m,phase_rad`
- `load_inspection.png`: two-panel load-in-time and spectral-magnitude plot (unless `--no-plots`)

The frequency output uses the **exact continuous-time Fourier transform of the specified source**, *not* the FFT of `load_time.csv`. Changing the time-domain preview duration **does not** change the continuous spectrum. By convention, `F̃(f)=∫₀^∞ F(t)exp(−i2πft)dt`; line-force spectral units are N s/m. The source is causal, `F(t)=0` for `t<0`. Frequency preview default is `0..120 Hz`, configurable using `--freq-max` and `--nfreq`. Use `--load-dt` and `--load-duration` for the time preview grid.

**2. Generate reference response.** Preview files are always written **first**, followed by the response files:

```bash
python run.py --preset three_step --out output/three_step
python run.py --preset ricker --out output/ricker
python run.py --preset rectangle --amplitude 2e6 --rectangle-start 0.05 \
              --rectangle-end 0.10 --out output/rectangle
```

For arbitrary surface receivers and spatial profiles:

```bash
python run.py --preset ricker --receiver 640 --receiver 1280 --receiver 1000 \
  --xmin 20 --xmax 1600 --dx 5 --dt 0.0005 --tmax 1.0 \
  --snapshot 0.30 --snapshot 0.50 --snapshot 0.85 --out output/my_run
```

To obtain **only receiver traces** and avoid surface-profile calculations, add `--no-profiles`. To suppress PNG plotting while retaining CSV, add `--no-plots`.

### Generic presets

| `--preset` | Definition | Default parameters |
|---|---|---|
| `ricker` | Causal Ricker wavelet `A(1-2π²f₀²(t-t₀)²)exp(-π²f₀²(t-t₀)²)` | `A=2e6 N/m`, `f₀=10 Hz`, `t₀=0.1 s` |
| `rectangle` | One positive plateau `A[H(t-t_start)-H(t-t_end)]` | `A=2e6 N/m`, `[0,0.15) s` |
| `three_step` | `A[H(t)-3H(t-.05)+3H(t-.10)-H(t-.15)]` | `A=2e6 N/m`, jumps at `0,0.05,0.10,0.15 s` |

`three_step` is **three consecutive rectangular force plateaus** `+A,−2A,+A`, not three identical positive rectangles. The jump times may be changed with `--three-step-times T0 T1 T2 T3`; amplitude via `--amplitude`.

### Article presets remain separately accessible

Use `--demo` (not `--preset`) to recover exactly the **load parameter choices** in the two articles, while all generic presets remain unprefixed:

| `--demo` | Article reference | Input |
|---|---|---|
| `kim_ricker` | Kim–Bathe (2021), Eq. 32 | `A=+2e6`, `f₀=10 Hz`, `t₀=0.1 s` |
| `kim_steps` | Kim–Bathe (2021), Eq. 33 | `A=+2e6`, three-step timings |
| `kwon_ricker` | Kwon et al. (2020), Eq. 38 | `A=−1e6`, `f₀=12.5 Hz`, `t₀=0.1 s` |
| `kwon_steps` | Kwon et al. (2020), Eq. 39 | `A=+2e3`, three-step timings |

The examples in `demo/` use these article names. Run `python run_demos.py` to regenerate them all (and one generic `rectangle` case). User-supplied `--amplitude` / `--ricker-frequency` etc. override the demo defaults if explicitly supplied.

**Material defaults for all cases:** `ρ=2200 kg/m³`, `cP=3200 m/s`, `cS=1847.5 m/s`, giving `cR≈1698.597 m/s`. Specify `--rho`, `--cp`, or `--cs` to alter these.

## Response CSV files and ParaView

- `receivers.csv`: columns `time_s,point_id,x_m,y_depth_m,ux_m,uy_m`. It is a *long-form table* containing all receivers and times.
- `surface_t0.300.csv` etc.: columns `x_m,y_depth_m,time_s,ux_m,uy_m`. This file is a profile at the **requested snapshot time** (not a time-step animation).
- `receivers.png`, `surface_profiles.png`: plots, if matplotlib is installed and enabled.
- `metadata.json`: exact material, loading, times, numerical method and caveats.

ParaView opens both CSV formats using the **CSV Reader**. `receivers.csv` can be plotted as tabular `ux_m` or `uy_m` vs `time_s` (e.g. via *Plot Data* after selecting a receiver). For `surface_t...csv`, *Table To Points* assigns `x_m` to X, `y_depth_m` to Y, and zero to Z; then use *Plot Data* / chart views to compare FEM and reference. For multiple receivers, separate `point_id` first using a table-selection/filter. Generic CSV tables do **not** automatically create a ParaView animation time slider. To use a native time slider in the future you would need a time-aware reader/format.

`write_vtkhdf` and the `--vtk-stride` option **have been removed from this release**. Archived demonstration VTKHDF files are not included.

## Python API

```python
import numpy as np
from lamb_surface_exact import ElasticHalfspace, Source, SurfaceStepGreen

material = ElasticHalfspace(rho=2200.0, cp=3200.0, cs=1847.5)
source = Source(kind='ricker', amplitude=2e6, ricker_frequency=10.0, ricker_shift=0.1)
t = np.arange(1001)*0.001
xs = np.array([640.0, 1280.0])
G = SurfaceStepGreen(material, max_tau=5.0)

traces = G.receivers(xs, t, source)        # shape (nt, nreceiver, 2)
profile = G.profile(np.arange(20, 1601, 5), 0.50023, source, dt=0.001)

# A published setup, without embedding article names in the generic presets:
source_k = Source.from_demo('kwon_steps')
```

`Source.force(t)` and `Source.spectrum(f_hz)` can be called independently without constructing a Green solver. `Source.jumps()` exposes exact step times and jump amplitudes for `rectangle` and `three_step`.

For a time-series Ricker convolution, `t` must be equally spaced starting at 0 (the method is discretized only in the **source convolution**, not in the Green function). `G.profile(...)` handles an arbitrary real-valued snapshot time directly without interpolation in displacement. Step responses work at arbitrary time arrays.

## Validation and convergence

```bash
python verify.py
```

The tests verify exact source spectra against independent time-domain quadrature; Ricker Fourier-transform symmetry; Rayleigh velocity; P-wave causality; opposite-side parity; integral identity `dQ/dt=S`; regular integral convergence; and Ricker time-step refinement. In a representative test at `x=640 m`, regular quadrature refinement changed the **three-step** response away from singular arrivals by at most ~`3.31e-10 m`. For **Ricker**, refining `Δt=0.002 → 0.001 → 0.0005 s` reduced the maximum difference from `7.63e-8` to `2.18e-8 m` at matched time samples. These values are *internal numerical checks* and do not constitute an independently tabulated analytical benchmark.

For greater regular-integral accuracy, increase `--n-core` and `--n-tail`, e.g. `--n-core 45001 --n-tail 24001`. For smooth Ricker response, also reduce `--dt`. The `load_time.csv` inspection step size `--load-dt` does *not* control the solution time step `--dt`.

## Important physical limitations

1. Surface-only (`y=0`), 2D plane strain, vertical point-in-plane **line force**. No buried points, inclined interior lines, horizontal forces, stresses, or geometric boundaries are provided in this implementation.
2. The ideal vertical **unit-step** response has a *physical logarithmic Rayleigh arrival singularity*. Hence the numerical maximum sampled at an arrival is not a converged finite physical peak; do not clip it silently. The transient response to a compact three-step force retains sharp exact wavefront features. Use comparisons away from singular arrival instants, or compare explicitly averaged source/measurement quantities.
3. A causal Ricker wavelet has `F(0+)≠0` in general; its initial jump is treated exactly, even if extremely small. For a perfectly smooth source extending into negative time the physical reference problem and its onset would differ.
4. The slowness regular integrals are numerical, and Ricker source convolution is numerical (second-order accurate in its time-cell approximation). The results are **semi-analytical** reference data, not an expression in elementary functions at all observation times.
5. The `x=0` ideal source is singular. For spatial evaluation use a nonzero minimum absolute `x`. The `max_tau` tabulation range must cover the largest sample delay divided by the shortest source–receiver distance.

## References

- Kwon, Bathe, Noh (2020), *An analysis of implicit time integration schemes for wave propagations*, Computers & Structures 230, 106188, §3.3, Eqs. (38)–(39).
- Kim and Bathe (2021), *Accurate solution of wave propagation problems in elasticity*, Computers & Structures 249, 106502, §6, Eqs. (32)–(33).
- Kausel (2006), *Fundamental Solutions in Elastodynamics*, §5.3 (half-space Green-function theory).

These two articles provide the **loading and benchmark setup**; the implementation of the non-FFT time-domain surface Green function is not code supplied with either paper.
