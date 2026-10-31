# Implementation notes — analytic structure, convergence and spectra

## 1. Source and receivers

The homogeneous plane-strain half-space occupies `y>=0`. A vertical line load `F(t)` acts at `(x,y)=(0,0)`, in N/m out-of-plane length. We calculate displacement at `(x,0)`, `x != 0`. Define `μ=ρ cS²`, `a=cS/cP`, `γ=cS/cR` and `τ=t cS/|x|`.

The model is linear, so its response to a finite jump-source `F(t)=∑ ΔFj H(t−tj)` is

`u(x,t)=∑ ΔFj S(x,t−tj)`, where `S` is the unit Heaviside force response and `S(x,t)=0` for `t<0`.

This **finite** superposition does not use Fourier frequencies or a temporal FFT. It is suitable for the discontinuous three-step source of both papers.

## 2. Analytic Green function and regular-integral tabulation

With dimensionless complex slowness `q`, use physical square-root branches:

```text
A(q) = sqrt(q²+a²)       B(q) = sqrt(q²+1)
D(q) = 2q²+1             R(q) = D(q)²-4q²A(q)B(q)
Phi_x(q) = q*(D(q)-2*A(q)*B(q))/R(q)
Phi_y(q) = A(q)/R(q)
```

The Rayleigh pole on `q=iγ` is separated by explicit residues `r_x` and `r_y`, computed from `R'(iγ)`. On the physical branch cut `q=iτ+0⁺`, the unit-step solution for `τ>=a` is:

```text
Sx(x,t) = sgn(x)/μ * [ (1/π) ∫[a,min(τ,1)] Re(Phi_x(i z+0⁺)) dz
                          + r_x H(τ−γ) ]
Sy(x,t) = 1/(πμ) * [ ∫[a,τ] (-Im(Phi_y(i z+0⁺))−r_y/(z−γ)) dz
                        + r_y ln(|τ−γ|/|a−γ|) ]
```

For `τ<a`, `Sx=Sy=0`. The integrals used by the code are **regular**: the singular `r_y/(z−γ)` is subtracted before quadrature; the analytic logarithm restores the removed pole exactly. The non-singular cumulative integrals are tabulated once on a dense nonuniform slowness grid, then interpolated with `PchipInterpolator`. Refinement control is `n_core,n_tail` / `--n-core,--n-tail`.

The logarithmic divergence at `τ=γ` is a real feature of an ideal two-dimensional line-load problem. It is *not* a Gibbs oscillation to remove.

## 3. Ricker waveform using time-domain convolution (no temporal spectral inversion)

For the causal Ricker wavelet

`F(t)=A*[1−2 b²(t−t0)²]*exp(−b²(t−t0)²)*H(t)`, `b=πf0`,

linearity and integration by parts give

`u(t)=F(0+)S(t)+∫₀ᵗ S(t−s) F'(s) ds`, with `F'` here the *ordinary derivative for s>0*.

The first term accounts explicitly for the jump from zero to `F(0+)` at `t=0`. Omitting it gives a different input force. This is especially important if `f0*t0` is not sufficiently large.

The direct convolution would sample the Green-function logarithm near a Rayleigh arrival. Instead define its **time primitive** `Q(t)=∫₀ᵗ S(v) dv`. The `ln|τ−γ|` contribution can be integrated exactly using

`∫ ln|τ−γ| dτ = (τ−γ)(ln|τ−γ|−1)` (the continuous limit at `τ=γ` is 0).

For time cells `[t_j,t_{j+1}]`, approximate `F'(s)` by the exact average

`gj=[F(t_{j+1})−F(t_j)]/[t_{j+1}−t_j]`.

Then

`u(t_n) ≈ F(0+)S(t_n) + ∑_{j=0}^{n−1} gj [Q(t_n−t_j)−Q(t_n−t_{j+1})]`.

This formula has no sampled singularity in its convolution weights, and is second order for smooth source variations away from physical onset singularities. For regularly spaced time histories the finite discrete convolution is accelerated by **zero-padded FFT convolution**. This FFT is *not* a truncation of the continuous frequency-domain elastodynamic solution, and does not introduce Gibbs ringing into the three-step solver (which never uses this convolution).

For a single arbitrary-time spatial profile, `SurfaceStepGreen.profile(x,t,source,dt)` evaluates the above sum directly with the true final time interval, without storing or interpolating an entire time-space solution.

## 4. Exact continuous Fourier spectrum of the input

Both figures and CSV from `--inspect-only` use an *analytic transform*, not an FFT of finite-record samples. Define the angular-frequency convention:

`F̃(ω)=∫₀^∞ F(t)e^(−iωt)dt`, `ω=2πf`.

For a **compact step source** with `∑ ΔFj=0`,

```text
F̃(ω) = [∑ ΔFj exp(−iωtj)]/(iω)         ω≠0
F̃(0) = −∑ ΔFj tj
```

Numerically we use `expm1(−iωtj)` to avoid cancellation near zero. These results have units N·s/m. In particular, the 3-step pulse has zero DC component (positive and negative impulse areas cancel). High-frequency magnitudes decay approximately as `1/|f|`, explaining why the step wavefronts challenge band-limited methods.

For the **causal Ricker**, set `a=(π f0)²`, `t0=ricker_shift`, `g(t)=exp(−a(t−t0)²)` and

```text
I(ω)=∫₀^∞ g(t) exp(−iωt) dt
    = (sqrt(π)/(2sqrt(a))) exp(−a*t0²)
      * erfcx(−sqrt(a)*t0+iω/(2sqrt(a)))
F̃(ω)=A/(2a) * [2a*t0*g(0)+iω*g(0)+ω²*I(ω)]
```

The formula follows from `F(t)=−A*g''(t)/(2a)` on `t>0`; the `g(0)` boundary terms are retained because the paper's Ricker is causal. We use SciPy's complex `erfcx` for stable evaluation.

**Warning:** For a Ricker defined on the entire real line, the spectrum differs slightly because negative-time contributions are included. This solver implements the papers' stated `t>0` causal loading convention.

## 5. How to assess convergence

- Increase `--n-core` and `--n-tail` to assess the regular slowness integrals (both step and Ricker).
- For Ricker, reduce `--dt` to assess the time-cell convolution approximation.
- For discontinuous sources, the horizontal Rayleigh step jumps and vertical logarithmic singularities remain sharply resolved: a finite-grid **peak** is not a meaningful convergence metric exactly at these arrivals.
- For spatial profiles, evaluate at the *specified* times using `profile`, rather than interpolating between two sharp nearby traces.
- For comparison to an FEM nodal force, ensure the FEM loading is equivalent to the half-space line force (force per out-of-plane length), and beware the FEM's implicit spatial source regularization.
- No claim is made that this software produces stress/strain or interior fields, or is a full implementation of the references' analytical solution.

## 6. Performance features

- Green-function regular integrals tabulated once for all receivers and all load types.
- Vectorized response evaluation in receiver blocks to limit peak memory usage.
- Generic step sources assembled from exact time shifts, without convolving sampled singular Green-function values.
- Ricker response uses cell-integrated Green-function convolution, FFT-accelerated **only** for regularly spaced time histories.
- Arbitrary-time Ricker spatial profiles use direct cell-integral sums rather than storing a dense all-times spatial grid.
- Matplotlib and CSV export are independent from core solver; no VTK package is required.
