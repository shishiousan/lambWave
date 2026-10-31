"""Regenerate four article-based demos plus a generic rectangle example.

Execute from the package directory: python run_demos.py
All data remain surface-only and CSV-based (no VTKHDF files).
"""
from pathlib import Path
from run import main


def run_all():
    root=Path(__file__).resolve().parent/'demo'
    for demo in ('kim_ricker','kim_steps','kwon_ricker','kwon_steps'):
        print('Generating paper demo:',demo,flush=True)
        main(['--demo',demo,'--out',str(root/demo),
              '--receiver','640','--receiver','1280',
              '--xmin','20','--xmax','1600','--dx','5',
              '--dt','.001','--tmax','1.0'])
    print('Generating generic rectangle',flush=True)
    main(['--preset','rectangle','--rectangle-start','0.05','--rectangle-end','0.10',
          '--out',str(root/'rectangle'), '--dt','.001','--tmax','1.0'])


if __name__=='__main__':run_all()
