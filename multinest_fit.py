#!/usr/bin/env python
"""Stand-alone MultiNest QU fit using polvista's own tools, no GUI.

Reproduces what the Sampling tab's Fit! button does, with its defaults:
  1. spectral index from a least-squares regression against I(nu)
     (fitting.estimate_alpha; epsilon pinned at app.FIT_FIXED_EPSILON for
     the weighted two-component models) -- held fixed while sampling;
  2. fitting.multinest_fit() over every p/X/phi/dphi/scale parameter with
     p in [0, 30 %], X over its full period, phi/sigma_phi log-uniform in
     log10 [0, 6.7] (sign of phi drawn separately), n_live_points from the
     model's registered preset, sampling efficiency 0.2, evidence tol 0.5;
  3. mode-family clustering to pick the winning family (done inside
     multinest_fit).

Only single-component 'powerlaw' spectral shapes are handled (the app's
default); the classic two-component models are of that kind.

Outputs, in <outdir>/<source>/ (source = CSV stem minus '_Pasetto_2018'):
  mn_* (MultiNest files + mn_polvista_meta.npz, loadable in the app via
  Load samples), mn_results.txt, mn_pol_evpa.png, mn_qu.png, mn_corner.png.

Example:
  python multinest_fit.py assets/data/1048+0141_Pasetto_2018.csv [more.csv ...]
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polvista.app import load_vapola_csv, fractional_qu, fixed_epsilons
from polvista.models import MODELS, get_model_by_name
from polvista.fitting import estimate_alpha, multinest_fit, circular_median_pctl
from polvista.sampling import (SAMPLING_P_DEFAULT, SAMPLING_LOG_RANGE, render_corner_figure,
                               CORNER_SAVE_DPI)

DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / 'polvista_test' / 'Pasetto_2018'


def fit_one(csv, args):
    func = get_model_by_name(args.model)
    spec = MODELS[func]
    source = Path(csv).stem.removesuffix('_Pasetto_2018')
    outdir = Path(args.outdir) / source
    outdir.mkdir(parents=True, exist_ok=True)
    basename = str(outdir / 'mn_')

    freq, I, Q, U, I_err, Q_err, U_err = load_vapola_csv(csv)
    wl, q, q_err, u, u_err = fractional_qu(freq, I, Q, U, I_err, Q_err, U_err)
    if args.fmin is not None:
        keep = freq >= args.fmin * 1e9
        wl, q, q_err, u, u_err, freq, I = (a[keep] for a in (wl, q, q_err, u, u_err, freq, I))

    # step 1: spectral pre-fit (powerlaw alpha from I(nu); eps pinned)
    alpha = estimate_alpha(freq, I)
    spectral_pars = fixed_epsilons(spec)
    spectral_pars.update({i: alpha for i in spec.indices('alpha')})

    # step 2: bounds exactly as the Sampling tab builds them
    lo, hi = spec.bounds
    kind_bounds = {}
    for i, p in enumerate(spec.params):
        if p.kind in ('alpha', 'eps', 'X') or p.kind in kind_bounds:
            continue
        if p.kind == 'p':
            kind_bounds['p'] = (SAMPLING_P_DEFAULT[0] / 100, SAMPLING_P_DEFAULT[1] / 100)
        elif p.kind in ('phi', 'dphi'):
            kind_bounds[p.kind] = (*SAMPLING_LOG_RANGE, True)
        else:
            kind_bounds[p.kind] = (lo[i], hi[i])

    np.savez(basename + 'polvista_meta.npz', model=func.__name__,
             spectral_idx=np.array(list(spectral_pars.keys())),
             spectral_val=np.array(list(spectral_pars.values())),
             wl=wl, q=q, q_err=q_err, u=u, u_err=u_err)

    n_live = args.n_live or spec.n_live_points
    print(f'[{source}] {len(wl)} points, alpha={alpha:.4f}, n_live={n_live}', flush=True)
    t0 = time.time()
    best_pars, errs, info = multinest_fit(
        wl, q, q_err, u, u_err, func, spectral_pars, kind_bounds, basename,
        n_live_points=n_live, sampling_efficiency=args.efficiency,
        evidence_tolerance=args.evidence_tol)
    print(f'[{source}] sampling done in {(time.time() - t0) / 60:.1f} min', flush=True)

    report(outdir, source, func, spec, best_pars, errs, info, alpha)
    plots(outdir, source, func, spec, wl, q, q_err, u, u_err, best_pars, errs, info, spectral_pars)


def report(outdir, source, func, spec, best_pars, errs, info, alpha):
    lines = [f'MultiNest fit: {source}   model {func.__name__}   alpha (lsq, fixed) = {alpha:.5g}',
             f"winning family: {info['evidence_share']:.2f}% evidence share; "
             f"{len(info['other_families'])} other families, {info['dropped']['count']} dropped",
             f"chi2 = {info['chi2']:.4g}  dof = {info['dof']}  chi2_k = {info['chi2_red']:.4g}",
             f"ln L = {info['loglike']:.5g}  AIC = {info['aic']:.5g}  BIC = {info['bic']:.5g}",
             f"winner ln Z = {info['winner_evidence_lnZ']:.4g}", '',
             f'{"param":8s} {"value":>14s} {"-err":>11s} {"+err":>11s}  unit']
    for i in info['pol_idx']:
        p = spec.params[i]
        s, unit = {'p': (100, '%'), 'X': (180 / np.pi, 'deg')}.get(
            p.kind, (1, 'rad/m2' if p.kind in ('phi', 'dphi') else ''))
        lines.append(f'{p.name:8s} {best_pars[i] * s:14.5g} {errs[i][0] * s:11.4g} {errs[i][1] * s:11.4g}  {unit}')
    for k, fam in enumerate(info['other_families'], 1):
        lines.append(f"other family {k}: evidence share {fam['evidence_share']:.2f}%  chi2_k {fam['chi2']:.3g}")
    txt = '\n'.join(lines)
    print(txt, flush=True)
    (outdir / 'mn_results.txt').write_text(txt + '\n')


def plots(outdir, source, func, spec, wl, q, q_err, u, u_err, best_pars, errs, info, spectral_pars):
    pol_idx = info['pol_idx']
    kinds_pol = [spec.params[i].kind for i in pol_idx]
    labels = [spec.params[i].latex for i in pol_idx]

    # corner (polvista's own renderer, winner + other families)
    winner = dict(pars=[best_pars[i] for i in pol_idx],
                  errs=np.array([[errs[i][0] for i in pol_idx], [errs[i][1] for i in pol_idx]]),
                  samples=info['winner_samples'], evidence_share=info['evidence_share'],
                  chi2=info['winner_chi2'], lnZ=info['winner_lnZ'],
                  evidence_lnZ=info['winner_evidence_lnZ'])
    fig = render_corner_figure(kinds_pol, labels, [winner] + info['other_families'],
                               info['dropped'], 0, len(pol_idx))
    fig.savefig(outdir / 'mn_corner.png', dpi=min(CORNER_SAVE_DPI, 300))

    # data + best fit + posterior draws
    rng = np.random.default_rng(0)
    S = info['winner_samples']
    draws = S[rng.choice(len(S), min(60, len(S)), replace=False)]

    def full(pol):
        m = [0.0] * len(spec.params)
        for i, v in zip(pol_idx, pol):
            m[i] = v
        for i, v in spectral_pars.items():
            m[i] = v
        return m

    wlg = np.linspace(wl.min() * 0.9, wl.max() * 1.05, 600)
    w2, w2g = (wl * 1e3) ** 2, (wlg * 1e3) ** 2
    P_best = func(wlg, best_pars)

    def curves(P, ref_deg=None):
        ang = 0.5 * np.unwrap(np.angle(P))
        if ref_deg is not None:
            ang += np.pi * np.round((np.radians(ref_deg[0]) - ang[0]) / np.pi)
        return np.abs(P), np.degrees(ang)
    p_best, ev_best = curves(P_best)
    evd = np.interp(wl, wlg, ev_best)
    ev_raw = 0.5 * np.angle(q + 1j * u)
    ev_obs = np.degrees(ev_raw + np.pi * np.round((np.radians(evd) - ev_raw) / np.pi))
    r = np.hypot(q, u)
    p_err = np.sqrt(q ** 2 * q_err ** 2 + u ** 2 * u_err ** 2) / r
    ev_err = np.degrees(0.5 * np.sqrt(q ** 2 * u_err ** 2 + u ** 2 * q_err ** 2) / r ** 2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for d in draws:
        pc, ec = curves(func(wlg, full(d)), ev_best)
        a1.plot(w2g, pc * 100, color='C0', alpha=0.08, lw=1)
        a2.plot(w2g, ec, color='C0', alpha=0.08, lw=1)
    a1.plot(w2g, p_best * 100, 'C3', lw=2, label='winning family')
    a2.plot(w2g, ev_best, 'C3', lw=2)
    a1.errorbar(w2, np.hypot(q, u) * 100, p_err * 100, fmt='k.', capsize=2, label='data')
    a2.errorbar(w2, ev_obs, ev_err, fmt='k.', capsize=2)
    a1.set(xlabel=r'$\lambda^2$ [mm$^2$]', ylabel='p [%]')
    a2.set(xlabel=r'$\lambda^2$ [mm$^2$]', ylabel='EVPA [deg]')
    a1.legend()
    fig.suptitle(f'{source} - {func.__name__} (MultiNest)')
    fig.tight_layout()
    fig.savefig(outdir / 'mn_pol_evpa.png', dpi=200)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for d in draws:
        P = func(wlg, full(d))
        a1.plot(w2g, P.real, color='C0', alpha=0.08, lw=1)
        a2.plot(w2g, P.imag, color='C0', alpha=0.08, lw=1)
    a1.plot(w2g, P_best.real, 'C3', lw=2)
    a2.plot(w2g, P_best.imag, 'C3', lw=2)
    a1.errorbar(w2, q, q_err, fmt='k.', capsize=2)
    a2.errorbar(w2, u, u_err, fmt='k.', capsize=2)
    a1.set(xlabel=r'$\lambda^2$ [mm$^2$]', ylabel='q = Q/I')
    a2.set(xlabel=r'$\lambda^2$ [mm$^2$]', ylabel='u = U/I')
    fig.tight_layout()
    fig.savefig(outdir / 'mn_qu.png', dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--model', default='class_Ext_Ext')
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--n-live', type=int, help="live points (default: the model's preset)")
    ap.add_argument('--efficiency', type=float, default=0.2)
    ap.add_argument('--evidence-tol', type=float, default=0.5)
    ap.add_argument('--fmin', type=float, help='drop data below this frequency [GHz]')
    args = ap.parse_args()
    for csv in args.csv:
        fit_one(csv, args)


if __name__ == '__main__':
    main()
