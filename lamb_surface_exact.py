"""Semi-analytical displacement of an elastic half-space at its *surface*.

Model: plane strain; homogeneous, isotropic, lossless half-space (y >= 0).
Loading: vertical concentrated *line force* F(t) [N/m] at (x,y)=(0,0).
Output: (u_x,u_y) [m] at (x,0), with y positive downwards.

The surface unit-step Green function is evaluated with a Cagniard-type
slowness-contour quadrature; the Rayleigh pole is extracted analytically.
Thus finite step/pulse superpositions do not suffer from temporal FFT ringing.
For Ricker loads, a second-order, cell-integrated Green-function convolution
is used; the FFT there only accelerates a convolution of smooth source changes,
not an inversion of discontinuous frequency data.

The vertical unit-step solution contains a genuine logarithmic singularity at
t=|x|/c_R. A finite point-sampled maximum is not a converged observable.

Dependencies: numpy and scipy. Plotting/CSV export belong to run.py.
No VTK or vtkhdf dependency; this module intentionally has no VTK exporter.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import erfcx
from scipy.fft import rfft, irfft, next_fast_len


@dataclass(frozen=True)
class ElasticHalfspace:
    rho: float = 2200.0
    cp: float = 3200.0
    cs: float = 1847.5

    def __post_init__(self):
        if self.rho <= 0 or not 0 < self.cs < self.cp:
            raise ValueError('Require rho>0 and 0<cs<cp')

    @property
    def mu(self) -> float:
        return self.rho * self.cs**2

    @property
    def alpha(self) -> float:
        return self.cs / self.cp


@dataclass(frozen=True)
class Source:
    """One of `ricker`, `rectangle`, `three_step` (Kim 2021 defaults).

    `amplitude` is a signed line force in N/m. `three_step` is:
    A*[H(t)-3 H(t-.05)+3 H(t-.10)-H(t-.15)],
    i.e. +A, -2A, +A on three successive 0.05-s intervals.

    `ricker` is causal: F(t)=A*(1-2*pi**2*f**2*(t-t0)**2)
      *exp(-pi**2*f**2*(t-t0)**2) for t>=0, zero otherwise.
    The onset F(0+) need not be exactly zero, even with t0=0.1 s.

    `rectangle` equals A on [rectangle_start,rectangle_end), else 0.

    Paper cases are accessed through `Source.from_demo('kim_steps')`.
    """
    kind: str = 'three_step'
    amplitude: float = 2.0e6
    ricker_frequency: float = 10.0
    ricker_shift: float = 0.1
    rectangle_start: float = 0.0
    rectangle_end: float = 0.15
    step_times: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
    step_jumps: tuple[float, ...] = (1.0, -3.0, 3.0, -1.0)

    @classmethod
    def from_demo(cls, name: str) -> 'Source':
        """Paper inputs, unchanged. User may override fields using replace()."""
        data = {
            'kim_ricker': dict(kind='ricker', amplitude=2e6, ricker_frequency=10., ricker_shift=0.1),
            'kwon_ricker': dict(kind='ricker', amplitude=-1e6, ricker_frequency=12.5, ricker_shift=0.1),
            'kim_steps': dict(kind='three_step', amplitude=2e6),
            'kwon_steps': dict(kind='three_step', amplitude=2e3),
        }
        if name not in data:
            raise ValueError(f'Unknown demo {name}. Choose from {tuple(data)}')
        return cls(**data[name])

    def __post_init__(self):
        if self.kind not in ('ricker', 'rectangle', 'three_step', 'custom_steps'):
            raise ValueError(f'Unknown source {self.kind}')
        if not np.isfinite(self.amplitude):
            raise ValueError('Amplitude must be finite')
        if self.kind == 'ricker' and (self.ricker_frequency <= 0 or self.ricker_shift < 0):
            raise ValueError('Ricker frequency must be positive and shift nonnegative')
        if self.kind == 'rectangle' and not 0 <= self.rectangle_start < self.rectangle_end:
            raise ValueError('Require 0<=rectangle_start<rectangle_end')
        if self.kind in ('three_step', 'custom_steps'):
            ts, ds = self.step_times, self.step_jumps
            if not ts or len(ts) != len(ds) or ts[0] < 0 or any(b <= a for a,b in zip(ts, ts[1:])):
                raise ValueError('Step times must be nonnegative, strictly increasing, and match jumps')
            if not np.isclose(sum(ds), 0.0, atol=1e-13):
                warnings.warn('Step source has a nonzero final force plateau')

    def jumps(self) -> tuple[np.ndarray, np.ndarray]:
        """Jumps (seconds, N/m) for piecewise-constant loads."""
        if self.kind == 'rectangle':
            return np.array([self.rectangle_start, self.rectangle_end]), self.amplitude*np.array([1.,-1.])
        if self.kind in ('three_step', 'custom_steps'):
            return np.array(self.step_times), self.amplitude*np.array(self.step_jumps)
        raise ValueError('Ricker is smooth (apart from its causal onset), not a step list')

    def force(self, t: np.ndarray | float) -> np.ndarray:
        """Continuous-time source force [N/m]; causal (zero before time 0)."""
        t = np.asarray(t, dtype=float)
        if self.kind == 'ricker':
            b = np.pi*self.ricker_frequency*(t-self.ricker_shift)
            return np.where(t >= 0, self.amplitude*(1-2*b*b)*np.exp(-b*b), 0.0)
        times, jumps = self.jumps()
        # <= a bin uses its new plateau; piecewise constant sampling of a
        # discontinuity is a convention and does not affect exact spectrum.
        return np.sum(jumps*np.greater_equal(t[..., None], times), axis=-1)

    def spectrum(self, f_hz: np.ndarray | float) -> np.ndarray:
        """Exact continuous Fourier transform F~(f)=integral_0^infty F(t)e^(-i2pi f t) dt.

        Units N*s/m. For steps we integrate the plateaus analytically, avoiding
        FFT artifacts; at f=0 we take the removable limit explicitly.
        For causal Ricker we use erfcx for a stable Gaussian half-line integral.
        """
        f = np.asarray(f_hz, dtype=float)
        w = 2*np.pi*f
        if self.kind != 'ricker':
            times, jumps = self.jumps()
            if not np.isclose(jumps.sum(), 0., atol=1e-12*max(1., abs(self.amplitude))):
                raise ValueError('The infinite-duration final force plateau has no ordinary Fourier transform')
            wt = w[...,None]*times
            # exp(-i*w*t)-1 computed accurately near zero; sum jumps=0
            num = np.sum(jumps*np.expm1(-1j*wt),axis=-1)
            out = np.empty(w.shape,complex)
            mask = np.abs(w) > 1e-12
            out[mask] = num[mask]/(1j*w[mask])
            out[~mask] = -np.dot(times,jumps)
            return out
        a = (np.pi*self.ricker_frequency)**2
        t0 = self.ricker_shift
        z = -np.sqrt(a)*t0 + 1j*w/(2*np.sqrt(a))
        integral_gauss = np.sqrt(np.pi)/(2*np.sqrt(a))*np.exp(-a*t0*t0)*erfcx(z)
        g0 = np.exp(-a*t0*t0)
        return self.amplitude/(2*a)*(2*a*t0*g0+1j*w*g0+w*w*integral_gauss)


class SurfaceStepGreen:
    """Tabulated surface *unit-step* Green function with Rayleigh pole removed.

    Tabulation is independent of load and observation positions, so a single
    instance is reusable for many receiver points, snapshots, and sources.
    `max_tau >= cs*max(t)/min(abs(x))`. Default x=0 is excluded.
    """
    def __init__(self, material: ElasticHalfspace=ElasticHalfspace(),
                 max_tau: float=30.0, n_core: int=30001, n_tail: int=18001):
        self.mat = material
        a = material.alpha
        if n_core < 100 or n_tail < 100:
            raise ValueError('Too few tabulation intervals')
        self.max_tau = max(float(max_tau), 1.5)
        self.gamma = brentq(lambda z: (1-2*z*z)**2 - 4*z*z*np.sqrt(z*z-a*a)*np.sqrt(z*z-1),1.0+1e-9,2.0,xtol=1e-14)
        self.cr = material.cs/self.gamma
        q=1j*self.gamma
        A,B,D,R=self._terms(q)
        dR=8*q*D - 8*q*A*B - 4*q*q*(q*B/A+q*A/B)
        self.ry=float(np.real(A/dR))
        self.rx=float(np.real(q*(D-2*A*B)/dR))
        g=self.gamma
        pieces=[np.linspace(a,1,n_core//2),
                np.linspace(1,g,n_core//4),
                np.linspace(g,1.45,n_core//4),
                np.geomspace(1.45,max(self.max_tau,1.451),n_tail)]
        z=np.unique(np.concatenate(pieces))
        y,x=self._density(z)
        safe=z!=g
        r_y=np.empty_like(z)
        r_y[safe]=y[safe]-self.ry/(z[safe]-g)
        if not np.all(safe):
            index=np.flatnonzero(~safe)[0]
            h=min(2e-6,(g-1)*0.0001)
            ylo=self._density(np.array([g-h]))[0][0]
            yhi=self._density(np.array([g+h]))[0][0]
            r_y[index]=0.5*(ylo+self.ry/h + yhi-self.ry/h)
        x[z>=1.0]=0.0
        self._ix=PchipInterpolator(z,cumulative_trapezoid(x,z,initial=0),extrapolate=False)
        self._iy=PchipInterpolator(z,cumulative_trapezoid(r_y,z,initial=0),extrapolate=False)
        self._ix_primitive=self._ix.antiderivative()
        self._iy_primitive=self._iy.antiderivative()
        self._base_x=float(self._ix_primitive(a))
        self._base_y=float(self._iy_primitive(a))

    def _terms(self,q):
        a=self.mat.alpha
        A=np.sqrt(q*q+a*a+0j); B=np.sqrt(q*q+1.+0j)
        D=2*q*q+1
        R=D*D-4*q*q*A*B
        return A,B,D,R

    def _density(self,z):
        q=1e-14+1j*np.asarray(z,dtype=float)
        A,B,D,R=self._terms(q)
        fy=A/R
        fx=q*(D-2*A*B)/R
        return -np.imag(fy),np.real(fx)

    def _evaluate(self, xs: np.ndarray, t: np.ndarray, *, integrated: bool) -> np.ndarray:
        """Vectorized [time, receiver, (ux,uy)] values.

        `integrated=True` returns Q(x,t)=integral_0^t S(x,s) ds, used for
        high-order-in-time convolution without sampling the Rayleigh pole.
        """
        xs=np.asarray(xs,dtype=float).reshape(-1)
        t=np.asarray(t,dtype=float).reshape(-1)
        if not len(xs) or np.any(xs==0):
            raise ValueError('All receiver x must be nonzero (singular line source)')
        tau=t[:,None]*self.mat.cs/np.abs(xs)[None,:]
        on=tau>self.mat.alpha
        if np.any(tau[on]>self.max_tau*(1+1e-12)):
            raise ValueError(f'tau beyond tabulation {self.max_tau}; enlarge max_tau')
        z=np.where(on,tau,self.mat.alpha)
        sign=np.sign(xs)[None,:]
        if integrated:
            px=(self._ix_primitive(z)-self._base_x)/np.pi+self.rx*np.maximum(z-self.gamma,0.)
            dz=z-self.gamma
            with np.errstate(divide='ignore',invalid='ignore'):
                f=np.where(dz!=0.,dz*(np.log(np.abs(dz))-1.),0.)
            # integral from a to z of ln|tau-gamma|/|a-gamma|
            lz=f - (z-self.mat.alpha)*np.log(self.gamma-self.mat.alpha)
            da=self.mat.alpha-self.gamma
            la=da*(np.log(abs(da))-1.)
            py=(self._iy_primitive(z)-self._base_y+self.ry*(lz-la))/np.pi
            result=np.stack((sign*px,py),axis=-1)
            result *= (np.abs(xs)[None,:,None]/self.mat.cs)/self.mat.mu
        else:
            px=self._ix(z)/np.pi+self.rx*(tau>=self.gamma)
            with np.errstate(divide='ignore'):
                logs=np.log(np.abs(z-self.gamma)/abs(self.mat.alpha-self.gamma))
            py=(self._iy(z)+self.ry*logs)/np.pi
            result=np.stack((sign*px,py),axis=-1)/self.mat.mu
        result[~on]=0.
        return result

    def step(self,x,t):
        """Surface displacement [len(t),2] per unit Heaviside force [N/m]."""
        t=np.asarray(t,float)
        return self._evaluate(np.array([x]),t.reshape(-1),integrated=False)[:,0,:].reshape(t.shape+(2,))

    def step_integral(self,x,t):
        """Exact (up to regular tabulation) integral_0^t S(x,s) ds [m*s/(N/m)]."""
        t=np.asarray(t,float)
        return self._evaluate(np.array([x]),t.reshape(-1),integrated=True)[:,0,:].reshape(t.shape+(2,))

    def response(self,x,t,source: Source=Source()):
        """Single receiver; for Ricker, t must be uniformly spaced from 0."""
        return self.receivers([x],t,source)[:,0,:]

    def receivers(self,x_positions,t,source: Source=Source(),chunk_size=80):
        """Evaluate displacement time histories; return array [nt,nx,2].

        Finite jumps are exact shifted step responses, without any time FFT.
        For Ricker only: a zero-padded FFT *accelerates convolution* of the
        source derivative with **cell-integrated** step response Q(t). This
        prevents sampled logarithmic wavefronts contaminating the convolution.
        The nonzero Ricker onset F(0+) is included exactly as F(0+)*S(t).
        """
        t=np.asarray(t,float)
        if t.ndim!=1 or len(t)==0 or not np.all(np.isfinite(t)):
            raise ValueError('t must be a finite one-dimensional time array')
        xs=np.atleast_1d(x_positions).astype(float)
        if xs.ndim!=1 or len(xs)==0:
            raise ValueError('Receivers must be a one-dimensional nonempty array')
        result=np.empty((len(t),len(xs),2),float)
        if source.kind=='ricker':
            if len(t)<2 or abs(t[0])>1e-12:
                raise ValueError('Ricker convolution requires at least two points and t[0]=0')
            dt=float(t[1]-t[0])
            if dt<=0 or not np.allclose(np.diff(t),dt,rtol=1e-9,atol=1e-12):
                raise ValueError('Ricker convolution requires a uniformly spaced time grid')
            # ∫ S(t-s) F'(s) ds + F(0+) S(t).
            # F' averaged in each bin by exact force differences.
            g_bar=np.diff(source.force(t))/dt
            nfft=next_fast_len(2*len(t)-3)
            gh=rfft(g_bar,n=nfft)
            f0=float(source.force(0.0))
            for j0 in range(0,len(xs),chunk_size):
                block=xs[j0:j0+chunk_size]
                # Integrating S exactly over each *cell* removes all sampled
                # Rayleigh log singularities from the convolution kernel.
                Q=self._evaluate(block,t,integrated=True)
                kern=np.diff(Q,axis=0)
                kh=rfft(kern,n=nfft,axis=0)
                conv=irfft(kh*gh[:,None,None],n=nfft,axis=0)[:len(t)-1]
                result[0,j0:j0+len(block)]=f0*self._evaluate(block,t[:1],integrated=False)[0]
                result[1:,j0:j0+len(block)]=conv
                if f0!=0.0:
                    result[1:,j0:j0+len(block)]+=f0*self._evaluate(block,t[1:],integrated=False)
        else:
            shifts,jumps=source.jumps()
            for j0 in range(0,len(xs),chunk_size):
                block=xs[j0:j0+chunk_size]
                arr=np.zeros((len(t),len(block),2),float)
                for delay,delta in zip(shifts,jumps):
                    arr+=delta*self._evaluate(block,t-delay,integrated=False)
                result[:,j0:j0+len(block)]=arr
        # The continuous Green function is exactly zero prior to the P arrival.
        # Zero out floating-point FFT roundoff in the Ricker convolution here.
        before_p=t[:,None] < (np.abs(xs)[None,:]/self.mat.cp)
        result[before_p] = 0.0
        return result

    def profile(self,x_positions, at_time: float, source: Source=Source(), dt: float=0.001,
                chunk_size: int=80) -> np.ndarray:
        """Surface spatial waveform at an *arbitrary* time [nposition, 2].

        For piecewise-constant inputs: evaluate the shifted Green functions at
        exactly `at_time`. For Ricker: integrate smooth source changes over a
        uniform time partition plus its possibly shorter final interval. No
        interpolation between displacement time levels is required.

        `dt` controls the Ricker convolution quadrature only. Positive x and
        negative x are supported, but x=0 is mathematically singular.
        """
        if not np.isfinite(at_time) or at_time<0 or dt<=0:
            raise ValueError('at_time must be nonnegative and dt positive')
        xs=np.atleast_1d(np.asarray(x_positions,float))
        if source.kind!='ricker' or at_time==0.:
            return self.receivers(xs,np.array([at_time]),source)[0]
        t=np.arange(int(np.floor(at_time/dt))+1,dtype=float)*dt
        t=t[t <= at_time]
        if t.size==0 or abs(t[-1]-at_time)>1e-12:
            t=np.r_[t,at_time]
        else:
            t[-1]=at_time
        differences=np.diff(source.force(t)) / np.diff(t)
        result=np.empty((len(xs),2))
        f0=float(source.force(0.))
        for j in range(0,len(xs),chunk_size):
            block=xs[j:j+chunk_size]
            Q=self._evaluate(block,at_time-t,integrated=True)
            result[j:j+len(block)]=np.tensordot(differences, Q[:-1]-Q[1:],axes=(0,0))
            if f0!=0.:
                result[j:j+len(block)]+=f0*self._evaluate(block,np.array([at_time]),integrated=False)[0]
        result[at_time < np.abs(xs)/self.mat.cp]=0.0
        return result

    def surface_grid(self,x_positions,t,source: Source=Source()):
        return self.receivers(x_positions,t,source)
