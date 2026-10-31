"""CLI for Lamb surface reference and source-load inspection.

Examples
--------
# Inspect source only (does not construct the Green-function solver):
python run.py --preset three_step --inspect-only --out output/load_check

# Run generic cases (Kim 2021 material and source defaults):
python run.py --preset ricker --out output/ricker
python run.py --preset rectangle --rectangle-start .05 --rectangle-end .10 --out output/rectangle
python run.py --preset three_step --out output/three_step

# Preserve the published Kim / Kwon load parameters as demo overrides:
python run.py --demo kwon_steps --out output/kwon_steps

No VTKHDF is generated. Long-form CSV can be loaded directly in ParaView.
"""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
import csv
import json
from pathlib import Path
import warnings
import numpy as np
from lamb_surface_exact import ElasticHalfspace, Source, SurfaceStepGreen


DEMO_NAMES = ('kim_ricker','kwon_ricker','kim_steps','kwon_steps')


def export_load(out: Path, source: Source, duration: float, dt: float,
                fmax: float, nfreq: int, plot: bool=True) -> None:
    """Export exact source time samples and exact *continuous* spectrum.

    The spectrum is not an FFT of the time CSV. This matters particularly for
    discontinuous inputs: finite sampling/FFT would show spectral leakage.
    The spectrum is computed from the mathematical input for all times, not
    only the user-selected load-preview duration.
    """
    if duration <= 0 or dt <= 0 or fmax <= 0 or nfreq < 2:
        raise ValueError('Load inspection requires positive duration, dt, fmax and nfreq>=2')
    t=np.arange(int(np.ceil(duration/dt))+1)*dt
    f=np.linspace(0.,fmax,nfreq)
    values=source.force(t)
    spectrum=source.spectrum(f)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'load_time.csv').open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['time_s','force_N_per_m'])
        w.writerows(zip(t,values))
    with (out/'load_spectrum.csv').open('w',newline='') as fp:
        w=csv.writer(fp)
        w.writerow(['frequency_hz','angular_frequency_rad_per_s',
                    'real_N_s_per_m','imag_N_s_per_m','magnitude_N_s_per_m','phase_rad'])
        w.writerows(zip(f,2*np.pi*f,spectrum.real,spectrum.imag,
                        np.abs(spectrum),np.angle(spectrum)))
    if plot:
        try:
            # Do not override the active Matplotlib backend: this function is also
            # called from Jupyter, where a forced Agg backend hides plt.show().
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                                 'mathtext.fontset':'stix','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
            fig,axs=plt.subplots(1,2,figsize=(11.2,3.6))
            axs[0].plot(t,values,lw=1.4)
            axs[0].set(xlabel='Time (s)',ylabel='Line force (N/m)',title=f'Load: {source.kind}')
            axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-2,3))
            mag=np.abs(spectrum)
            axs[1].plot(f,mag,lw=1.4)
            axs[1].set(xlabel='Frequency (Hz)',ylabel=r'$|\widetilde F(f)|$ (N s/m)',
                       title='Exact continuous Fourier spectrum')
            axs[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,3))
            if source.kind=='ricker':
                axs[1].axvline(source.ricker_frequency,lw=0.9,ls='--',color='0.5',label=f'f = {source.ricker_frequency:g} Hz')
                axs[1].legend(fontsize=8)
            for ax in axs:ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(out/'load_inspection.png',dpi=180)
            plt.close(fig)
        except ImportError:
            warnings.warn('Matplotlib is unavailable: load CSVs were exported without PNG')


def write_receivers(path: Path, times, receiver_x, displacement) -> None:
    """CSV long-form point histories, one row per (time, receiver).

    For ParaView: use CSVReader and optionally `Table To Points`, with
    x_m as X, y_depth_m as Y; time_s is retained for table plots/selection.
    """
    with path.open('w',newline='') as fp:
        w=csv.writer(fp)
        w.writerow(['time_s','point_id','x_m','y_depth_m','ux_m','uy_m'])
        for i,tt in enumerate(times):
            for j,xx in enumerate(receiver_x):
                w.writerow([f'{tt:.12g}',j,f'{xx:.12g}',0.,
                            f'{displacement[i,j,0]:.16g}',f'{displacement[i,j,1]:.16g}'])


def write_surface_snapshot(path: Path, x, at_time: float, values) -> None:
    """Spatial profile at exactly `at_time` (no temporal interpolation)."""
    with path.open('w',newline='') as fp:
        w=csv.writer(fp)
        w.writerow(['x_m','y_depth_m','time_s','ux_m','uy_m'])
        w.writerows((f'{xx:.12g}',0.,f'{at_time:.12g}',f'{u[0]:.16g}',f'{u[1]:.16g}')
                    for xx,u in zip(x,values))


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--preset','--case',dest='preset',choices=['ricker','rectangle','three_step'],default='three_step',
                       help='Generic load type (Kim parameters by default).')
    group.add_argument('--demo',choices=DEMO_NAMES,help='Published loading parameters, separate from generic presets.')
    p.add_argument('--amplitude',type=float,default=None,help='Signed force amplitude in N/m')
    p.add_argument('--ricker-frequency',type=float,default=None,help='Ricker central frequency in Hz')
    p.add_argument('--ricker-shift',type=float,default=None,help='Ricker shift in seconds')
    p.add_argument('--rectangle-start',type=float,default=None,help='Pulse start (s)')
    p.add_argument('--rectangle-end',type=float,default=None,help='Pulse end (s)')
    p.add_argument('--three-step-times',type=float,nargs=4,metavar=('T0','T1','T2','T3'),default=None,
                   help='Times of jumps +1,-3,+3,-1 (s); default 0,.05,.10,.15')
    p.add_argument('--inspect-only',action='store_true',help='Export source plots and CSVs ONLY, without solving wave propagation')
    p.add_argument('--load-dt',type=float,default=0.00025,help='Load plot/CSV resolution (s)')
    p.add_argument('--load-duration',type=float,default=0.30,help='Load plot/CSV end time (s)')
    p.add_argument('--freq-max',type=float,default=120.,help='Max frequency plotted/exported (Hz)')
    p.add_argument('--nfreq',type=int,default=1201,help='Frequency samples in continuous spectrum')
    p.add_argument('--out',type=Path,default=Path('output/surface'))
    p.add_argument('--receiver',type=float,action='append',default=None,help='Surface receiver x (m); repeat to add several')
    p.add_argument('--xmin',type=float,default=20.)
    p.add_argument('--xmax',type=float,default=1600.)
    p.add_argument('--dx',type=float,default=5.)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--tmax',type=float,default=1.)
    p.add_argument('--snapshot',type=float,action='append',default=None,help='Surface snapshot time (s), repeatable')
    p.add_argument('--no-profiles',action='store_true',help='Skip surface profiles (faster for receiver-only runs)')
    p.add_argument('--rho',type=float,default=2200.)
    p.add_argument('--cp',type=float,default=3200.)
    p.add_argument('--cs',type=float,default=1847.5)
    p.add_argument('--n-core',type=int,default=30001,help='Regular slowness quadrature grid; increase to check accuracy')
    p.add_argument('--n-tail',type=int,default=18001)
    p.add_argument('--no-plots',action='store_true',help='Omit PNGs, keep all CSV exports')
    return p


def source_from_args(args):
    src=Source.from_demo(args.demo) if args.demo else Source(kind=args.preset)
    override={}
    for key,argkey in [('amplitude','amplitude'),('ricker_frequency','ricker_frequency'),
                       ('ricker_shift','ricker_shift'),('rectangle_start','rectangle_start'),
                       ('rectangle_end','rectangle_end')]:
        v=getattr(args,argkey)
        if v is not None:override[key]=v
    if args.three_step_times is not None:
        override['step_times']=tuple(args.three_step_times)
    return replace(src,**override)


def plot_displacements(out,t,x,urec,xs,snapshot_data):
    try:
        # Respect the caller's backend (notebook inline/widget or CLI default).
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                             'mathtext.fontset':'stix','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
        fig,axes=plt.subplots(2,len(x),figsize=(5*len(x),6),squeeze=False,sharex=True)
        for j,xx in enumerate(x):
            for d in range(2):
                ax=axes[d,j];ax.plot(t,urec[:,j,d],lw=1.1)
                ax.set_title(f'{"Horizontal" if d==0 else "Vertical"}; x={xx:g} m')
                ax.set_ylabel('Displacement (m)');ax.grid(alpha=.25)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                if d==1:ax.set_xlabel('Time (s)')
        fig.tight_layout();fig.savefig(out/'receivers.png',dpi=160);plt.close(fig)
        if snapshot_data:
            fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
            for ts,prof in snapshot_data:
                for i in range(2):axes[i].plot(xs,prof[:,i],label=f't={ts:g} s')
            for i in range(2):
                axes[i].set_ylabel(('ux','uy')[i]+' (m)');axes[i].grid(alpha=.25);axes[i].legend()
            axes[1].set_xlabel('Surface distance x (m)')
            fig.tight_layout();fig.savefig(out/'surface_profiles.png',dpi=160);plt.close(fig)
    except ImportError:
        warnings.warn('Matplotlib unavailable: no result PNG plots generated')


def main(argv=None):
    p=build_parser();args=p.parse_args(argv)
    try:source=source_from_args(args)
    except ValueError as e:p.error(str(e))
    out=args.out
    # Step 1: ALWAYS produce load data before initializing the Green solver.
    export_load(out,source,args.load_duration,args.load_dt,args.freq_max,args.nfreq,plot=not args.no_plots)
    print(f'Load inspection -> {out / "load_time.csv"}, {out / "load_spectrum.csv"}')
    if args.inspect_only:
        print('Load inspection complete (analytical solution not generated).')
        return
    if not(args.dt>0 and args.dx>0 and args.tmax>0 and args.xmin!=0 and args.xmax>args.xmin):
        p.error('Invalid time/spatial sampling parameters')
    if args.xmin < 0 < args.xmax:
        p.error('Surface x grid cannot contain singular source x=0; use separate negative/positive ranges')
    t=np.arange(int(np.ceil(args.tmax/args.dt))+1)*args.dt
    snapshots=tuple(args.snapshot) if args.snapshot is not None else tuple(s for s in (0.3,0.5,0.85) if s<=t[-1])
    if any(s < 0 or s > t[-1] for s in snapshots):p.error('Snapshot times must lie between 0 and tmax')
    xs=np.arange(args.xmin,args.xmax+.5*args.dx,args.dx)
    receivers=np.asarray(args.receiver if args.receiver else [640.,1280.],float)
    if np.any(xs==0) or np.any(receivers==0):p.error('Cannot evaluate at x=0 (singular line source)')
    try:
        mat=ElasticHalfspace(args.rho,args.cp,args.cs)
        min_x=np.min(np.abs(receivers)) if args.no_profiles else min(np.min(np.abs(xs)),np.min(np.abs(receivers)))
        max_tau=max(3.,t[-1]*mat.cs/min_x*1.01)
        green=SurfaceStepGreen(mat,max_tau=max_tau,n_core=args.n_core,n_tail=args.n_tail)
        urec=green.receivers(receivers,t,source)
        # A snapshot is evaluated *exactly at its requested time*, not
        # interpolated between neighboring potentially singular wavefronts.
        snapshot_data=[]
        if not args.no_profiles:
            # Direct arbitrary-time spatial waveforms. Ricker uses exact
            # cell-integrated kernel quadrature and does NOT require storing
            # the entire time-by-space displacement grid.
            snapshot_data=[(float(s),green.profile(xs,float(s),source,dt=args.dt))
                           for s in snapshots]
        for s,profile in snapshot_data:
            # Retain familiar three-decimal names for paper snapshots, but
            # avoid rounding/collisions for arbitrary off-grid times.
            stamp=(f'{s:.3f}' if abs(s-round(s,3))<1e-12 else
                   f'{s:.9f}'.rstrip('0').rstrip('.'))
            write_surface_snapshot(out/f'surface_t{stamp}.csv',xs,s,profile)
        if not np.all(np.isfinite(urec)):
            warnings.warn('Response includes infinite exact Rayleigh step singularities at sampled arrivals; receiver CSV preserves them')
        write_receivers(out/'receivers.csv',t,receivers,urec)
        metadata=dict(method='Non-FFT slowness-contour step Green function / cell-integrated Ricker convolution',
                      source=asdict(source),paper_demo=args.demo,material=asdict(mat),
                      rayleigh_speed_m_s=green.cr,rayleigh_tau=green.gamma,
                      receivers_m=receivers.tolist(),snapshot_times_s=list(snapshots),
                      zero_depth_surface_only=True,
                      max_tau=green.max_tau,load_preview_csv='load_time.csv',
                      load_spectrum_csv='load_spectrum.csv',
                      spectral_convention='integral F(t)*exp(-i*2*pi*f*t) dt, t from 0 to infinity',
                      singularity_warning='Vertical step response diverges logarithmically at ideal Rayleigh arrival; sampled peak is not converged',
                      vtkhdf_export='Deprecated/removed; use CSV directly in ParaView')
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        if not args.no_plots:plot_displacements(out,t,receivers,urec,xs,snapshot_data)
        print(f'Wrote solution -> {out}; c_R={green.cr:.6f} m/s, {len(t)} time samples')
    except ValueError as e:p.error(str(e))


if __name__=='__main__':main()
