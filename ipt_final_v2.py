#!/usr/bin/env python3
"""
IPT Final Experiments v2 — CORRECTED MODEL
==========================================
Uses original OrthogonalRNN (orthogonal W, stochastic input) + Ridge R² as Psi.
Imports dynamics from the validated ipt_experiment codebase.

Experiments:
  C2: Finite-size scaling to N=8192
  B2: Hopf angle theta vs GRU training quality
  A2: betaG collapse ordering (two versions)

Usage:
    python ipt_final_v2.py --exp C2
    python ipt_final_v2.py --exp B2
    python ipt_final_v2.py --exp A2
    python ipt_final_v2.py --exp all
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import from original codebase — DO NOT MODIFY those files
_IPT_PATH = os.path.expanduser(
    '~/projects1/IPT/Calude_IPT/3_IPT_Memory/ipt_experiment')
sys.path.insert(0, _IPT_PATH)

# List available modules for debugging
print(f"IPT codebase: {_IPT_PATH}")
_py_files = [f for f in os.listdir(_IPT_PATH) if f.endswith('.py')]
print(f"Available modules: {_py_files}")

from dynamics import OrthogonalRNN

# Try importing Lyapunov function — name may vary
compute_lyapunov_exponent = None
import importlib
_dynamics = importlib.import_module('dynamics')
for _name in ('compute_lyapunov_exponent', 'compute_lyapunov',
              'lyapunov_exponent', 'max_lyapunov'):
    if hasattr(_dynamics, _name):
        compute_lyapunov_exponent = getattr(_dynamics, _name)
        print(f"Imported Lyapunov function: dynamics.{_name}")
        break
if compute_lyapunov_exponent is None:
    print("[WARN] No Lyapunov function found in dynamics.py — "
          "will use custom implementation only")
    def compute_lyapunov_exponent(system, steps=10000, warmup=200, **kwargs):
        """Fallback: generate trajectory and compute Lyapunov on it."""
        traj = system.generate_trajectory(steps, warmup=warmup)
        return compute_lyapunov_on_trajectory(system, traj, steps=steps)

# Try importing GRUSelfModel — may have different name
try:
    from self_model import GRUSelfModel
    print("Imported: self_model.GRUSelfModel")
except ImportError:
    try:
        from self_model import SelfModel as GRUSelfModel
        print("Imported: self_model.SelfModel (as GRUSelfModel)")
    except ImportError:
        # Last resort: inspect the module
        import self_model as _sm
        _classes = [c for c in dir(_sm) if isinstance(getattr(_sm, c), type)
                    and issubclass(getattr(_sm, c), nn.Module)
                    and c != 'Module']
        if _classes:
            GRUSelfModel = getattr(_sm, _classes[0])
            print(f"Imported: self_model.{_classes[0]} (auto-detected)")
        else:
            raise ImportError(f"No nn.Module subclass found in self_model.py. "
                             f"Contents: {dir(_sm)}")

# ─── Patch OrthogonalRNN if .jacobian() is missing ────────────
if not hasattr(OrthogonalRNN, 'jacobian'):
    def _jacobian(self, x):
        """
        Jacobian of x_{t+1} = tanh(λ W x_t + ...) w.r.t. x_t.
        J = diag(1 - x_{t+1}^2) · λ W
        NOTE: this ignores the noise term u_t (correct for Jacobian w.r.t. x).
        x should be a 1D tensor of shape (N,) or 2D of shape (1, N).
        """
        if x.dim() == 2:
            x = x.squeeze(0)
        with torch.no_grad():
            pre = self.lam * (x @ self.W.T)
            x_next = torch.tanh(pre)
            sech2 = 1.0 - x_next ** 2  # shape (N,)
            J = sech2.unsqueeze(1) * (self.lam * self.W)  # diag(sech2) @ (λW)
        return J
    OrthogonalRNN.jacobian = _jacobian
    print("[PATCH] Added .jacobian() method to OrthogonalRNN")

# Note: GRUSelfModel sets self.hidden_dim in __init__, no patch needed

# ─── Global config ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE_SM = 'cuda:1' if torch.cuda.device_count() > 1 else DEVICE

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def measure_ridge_r2(traj_np, alpha=1.0, train_frac=0.6):
    """Ridge regression R-squared predicting x_{t+1} from x_t."""
    T = traj_np.shape[0]
    X, Y = traj_np[:-1], traj_np[1:]
    split = int(train_frac * len(X))
    ridge = Ridge(alpha=alpha)
    ridge.fit(X[:split], Y[:split])
    return max(ridge.score(X[split:], Y[split:]), 0.0)


def compute_chi(lam_arr, psi_arr):
    """Susceptibility chi = dPsi/dlambda via centered finite difference."""
    return np.gradient(psi_arr, lam_arr)


def find_zero_crossing(lam_arr, vals):
    """Linear interpolation to find zero crossing."""
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            f = -vals[i] / (vals[i + 1] - vals[i])
            return lam_arr[i] + f * (lam_arr[i + 1] - lam_arr[i])
    return np.nan


def measure_R_eff(traj_np, n_bins=50):
    """Participation ratio from top-2 PCA."""
    if traj_np.shape[0] < 10:
        return 1.0
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(traj_np)
        ranges = proj.max(0) - proj.min(0)
        if np.any(ranges < 1e-10):
            return 1.0
        xb = np.linspace(proj[:, 0].min(), proj[:, 0].max(), n_bins + 1)
        yb = np.linspace(proj[:, 1].min(), proj[:, 1].max(), n_bins + 1)
        hist, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=[xb, yb])
        p = hist.flatten()
        p = p[p > 0]
        p = p / p.sum()
        return 1.0 / np.sum(p ** 2)
    except Exception:
        return 1.0


def compute_lyapunov_on_trajectory(system, trajectory, steps=10000):
    """Compute maximal Lyapunov exponent along a pre-generated trajectory."""
    T = trajectory.shape[0]
    steps = min(steps, T - 1)
    N = system.N

    v = torch.randn(N, device=system.device)
    v = v / v.norm()

    log_stretches = []
    for t in range(steps):
        x_t = trajectory[t]
        J = system.jacobian(x_t)
        v_new = J @ v
        stretch = v_new.norm()
        if stretch > 0:
            log_stretches.append(torch.log(stretch).item())
            v = v_new / stretch

    return np.mean(log_stretches) if log_stretches else 0.0


def generate_trajectory_biased(system, steps, warmup, x_star, beta_G):
    """Generate trajectory with input noise biased toward x*."""
    N = system.N
    device = system.device
    total = warmup + steps
    x = torch.randn(1, N, device=device) * 0.1
    noise = torch.randn(total, N, device=device)
    x_star_exp = x_star.unsqueeze(0)

    trajectory = []
    for t in range(total):
        diff = x_star_exp - x
        norm_diff = diff.norm(dim=1, keepdim=True).clamp(min=1e-8)
        u = beta_G * diff / norm_diff + noise[t:t + 1]
        x = system.step(x, u)
        if t >= warmup:
            trajectory.append(x.squeeze(0))

    return torch.stack(trajectory)


def save_json(data, path):
    """Save data as JSON with numpy type conversion."""
    def conv(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): conv(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [conv(v) for v in obj]
        return obj
    with open(path, 'w') as f:
        json.dump(conv(data), f, indent=2)


def load_json(path):
    """Load JSON checkpoint."""
    path = str(path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT C2 — Finite-Size Scaling
# ═══════════════════════════════════════════════════════════════

def run_exp_C2():
    """
    Finite-size scaling with original OrthogonalRNN + stochastic input.
    Psi = Ridge R-squared, sweep lambda for each N.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT C2 — FINITE-SIZE SCALING (original model)")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_C2'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'

    N_values = [256, 512, 1024, 2048, 4096, 8192]
    lam_range = (0.8, 1.3)
    warmup = 2000
    measure = 20000
    lyap_steps = 10000

    ckpt = load_json(ckpt_path) or {}
    t0 = time.time()

    for N in N_values:
        n_lam = 80 if N == 8192 else 100
        seeds = [42, 43] if N == 8192 else [42, 43, 44]
        lam_arr = np.linspace(*lam_range, n_lam)

        for seed in seeds:
            key = f"N{N}_s{seed}"
            if key in ckpt:
                print(f"  {key}: already done, skipping")
                continue

            psi_arr = np.zeros(n_lam)
            lyap_arr = np.zeros(n_lam)

            system = OrthogonalRNN(N, 1.0, input_scale=0.01,
                                   seed=seed, device=DEVICE)

            for i, lam in enumerate(tqdm(lam_arr, desc=f"N={N} s={seed}")):
                system.lam = lam
                traj = system.generate_trajectory(measure, warmup=warmup)
                traj_np = traj.cpu().numpy().astype(np.float64)

                psi_arr[i] = measure_ridge_r2(traj_np)

                lyap_arr[i] = compute_lyapunov_on_trajectory(
                    system, traj, steps=lyap_steps)

                del traj, traj_np
                torch.cuda.empty_cache()

            ckpt[key] = {
                'N': N, 'seed': seed,
                'lambdas': lam_arr.tolist(),
                'psi': psi_arr.tolist(),
                'lyapunov': lyap_arr.tolist(),
            }
            save_json(ckpt, ckpt_path)

            chi = compute_chi(lam_arr, psi_arr)
            idx_max = np.argmax(chi)
            lam_c_lyap = find_zero_crossing(lam_arr, lyap_arr)
            print(f"  {key}: lam_c(chi)={lam_arr[idx_max]:.4f}, "
                  f"chi_max={chi[idx_max]:.4f}, lam_c(Lyap)={lam_c_lyap:.4f}")

            del system
            torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 3600
    print(f"\nC2 data collection: {elapsed:.1f}h")

    # ── Analysis ──────────────────────────────────────────────
    print("\n[C2] Analyzing...")

    table_rows = []
    all_sweep = {}

    for N in N_values:
        seeds = [42, 43] if N == 8192 else [42, 43, 44]
        chi_maxs, lam_cs, psi_peaks = [], [], []
        sweep_agg = {'lams': [], 'psis': [], 'chis': []}

        for seed in seeds:
            key = f"N{N}_s{seed}"
            if key not in ckpt:
                print(f"  WARNING: {key} missing!")
                continue
            d = ckpt[key]
            lams = np.array(d['lambdas'])
            psis = np.array(d['psi'])
            chi = compute_chi(lams, psis)

            idx = np.argmax(chi)
            chi_maxs.append(chi[idx])
            lam_cs.append(lams[idx])
            psi_peaks.append(psis[idx])

            sweep_agg['lams'].append(lams)
            sweep_agg['psis'].append(psis)
            sweep_agg['chis'].append(chi)

        if not chi_maxs:
            continue

        all_sweep[N] = sweep_agg
        table_rows.append({
            'N': N,
            'chi_max_mean': float(np.mean(chi_maxs)),
            'chi_max_std': float(np.std(chi_maxs)),
            'lambda_c_mean': float(np.mean(lam_cs)),
            'lambda_c_std': float(np.std(lam_cs)),
            'psi_peak_mean': float(np.mean(psi_peaks)),
            'psi_peak_std': float(np.std(psi_peaks)),
        })

    # Print table
    print("\n" + "-" * 80)
    print(f"{'N':>6} | {'chi_max (mean+/-std)':>22} | "
          f"{'lam_c (mean+/-std)':>20} | {'Psi_peak (mean+/-std)':>22}")
    print("-" * 80)
    for row in table_rows:
        print(f"{row['N']:>6} | "
              f"{row['chi_max_mean']:>9.5f}+/-{row['chi_max_std']:<9.5f} | "
              f"{row['lambda_c_mean']:>8.4f}+/-{row['lambda_c_std']:<8.4f} | "
              f"{row['psi_peak_mean']:>9.5f}+/-{row['psi_peak_std']:<9.5f}")
    print("-" * 80)

    # Save CSV
    csv_path = out_dir / 'scaling_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Table: {csv_path}")

    # Sanity check
    lam_c_avg = np.mean([r['lambda_c_mean'] for r in table_rows])
    if lam_c_avg < 0.95:
        print(f"\nWARNING: lam_c ~ {lam_c_avg:.3f} is too low! "
              "Expected ~1.043. Possible Gaussian W instead of orthogonal.")
    elif 1.0 < lam_c_avg < 1.1:
        print(f"\nSANITY CHECK PASSED: lam_c ~ {lam_c_avg:.3f} "
              "consistent with expected ~1.043")

    # ── Figure 1: chi_max scaling ─────────────────────────────
    Ns = np.array([r['N'] for r in table_rows])
    chi_arr = np.array([r['chi_max_mean'] for r in table_rows])
    chi_err = np.array([r['chi_max_std'] for r in table_rows])

    def power_law(x, a, kappa):
        return a * x ** kappa

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, chi_arr, yerr=chi_err, fmt='o', color='C0',
                capsize=4, markersize=7, label='Data')
    kappa, kappa_err = np.nan, np.nan
    try:
        popt, pcov = curve_fit(power_law, Ns, chi_arr, p0=[0.01, 1.0],
                               sigma=chi_err + 1e-10, absolute_sigma=True)
        kappa, kappa_err = popt[1], np.sqrt(pcov[1, 1])
        N_fit = np.logspace(np.log10(Ns[0] * 0.8), np.log10(Ns[-1] * 1.2), 100)
        ax.plot(N_fit, power_law(N_fit, *popt), '--', color='C0',
                label=f'kappa = {kappa:.3f} +/- {kappa_err:.3f}')
        print(f"\nchi_max ~ N^kappa:  kappa = {kappa:.4f} +/- {kappa_err:.4f}")
    except Exception as e:
        print(f"Fit failed: {e}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (system size)')
    ax.set_ylabel('chi_max (peak susceptibility)')
    ax.set_title('C2: Susceptibility Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_dir / 'chi_scaling.png')
    plt.close(fig)
    print(f"Figure: {out_dir / 'chi_scaling.png'}")

    # ── Figure 2: Psi_peak scaling ────────────────────────────
    psi_arr = np.array([r['psi_peak_mean'] for r in table_rows])
    psi_err = np.array([r['psi_peak_std'] for r in table_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, psi_arr, yerr=psi_err, fmt='s', color='C1',
                capsize=4, markersize=7, label='Data')
    try:
        popt_p, pcov_p = curve_fit(power_law, Ns, psi_arr, p0=[1.0, 0.1],
                                   sigma=psi_err + 1e-10, absolute_sigma=True)
        N_fit = np.logspace(np.log10(Ns[0] * 0.8), np.log10(Ns[-1] * 1.2), 100)
        ax.plot(N_fit, power_law(N_fit, *popt_p), '--', color='C1',
                label=f'exp = {popt_p[1]:.3f} +/- {np.sqrt(pcov_p[1, 1]):.3f}')
        print(f"Psi_peak ~ N^alpha:  alpha = {popt_p[1]:.4f} "
              f"+/- {np.sqrt(pcov_p[1, 1]):.4f}")
    except Exception:
        pass
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (system size)')
    ax.set_ylabel('Psi_peak (Ridge R-squared)')
    ax.set_title('C2: Psi Peak Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_dir / 'psi_scaling.png')
    plt.close(fig)
    print(f"Figure: {out_dir / 'psi_scaling.png'}")

    # ── Figure 3: Data collapse ───────────────────────────────
    nu = 2.615
    lam_c_est = 1.0426

    def collapse_quality(beta_nu, nu_val=nu, lc_val=lam_c_est):
        all_x, all_y = [], []
        for N in all_sweep:
            for i in range(len(all_sweep[N]['lams'])):
                lams = all_sweep[N]['lams'][i]
                psis = all_sweep[N]['psis'][i]
                x_s = (lams - lc_val) * N ** (1.0 / nu_val)
                y_s = psis * N ** beta_nu
                all_x.extend(x_s.tolist())
                all_y.extend(y_s.tolist())
        all_x, all_y = np.array(all_x), np.array(all_y)
        x_lo, x_hi = np.percentile(all_x, [5, 95])
        bins = np.linspace(x_lo, x_hi, 41)
        total_var, n_valid = 0, 0
        for j in range(40):
            mask = (all_x >= bins[j]) & (all_x < bins[j + 1])
            if np.sum(mask) > 2:
                total_var += np.var(all_y[mask])
                n_valid += 1
        return total_var / max(n_valid, 1)

    bn_range = np.linspace(-0.5, 0.5, 200)
    quals = [collapse_quality(bn) for bn in bn_range]
    best_bn = bn_range[np.argmin(quals)]
    var_raw = collapse_quality(0.0)
    var_best = collapse_quality(best_bn)
    Q = 1.0 - var_best / var_raw if var_raw > 0 else 0.0
    print(f"Best beta/nu = {best_bn:.3f}, Q = {Q:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = plt.cm.viridis(np.linspace(0.15, 0.9, len(all_sweep)))
    for ni, N in enumerate(sorted(all_sweep.keys())):
        for i in range(len(all_sweep[N]['lams'])):
            lams = all_sweep[N]['lams'][i]
            psis = all_sweep[N]['psis'][i]
            x_s = (lams - lam_c_est) * N ** (1.0 / nu)
            y_s = psis * N ** best_bn
            label = f'N={N}' if i == 0 else None
            ax.plot(x_s, y_s, '.', color=colors_map[ni], alpha=0.5,
                    markersize=3, label=label)
    ax.set_xlabel(r'$(\lambda - \lambda_c) \cdot N^{1/\nu}$')
    ax.set_ylabel(r'$\Psi \cdot N^{\beta/\nu}$')
    ax.set_title(f'Data Collapse (nu={nu}, beta/nu={best_bn:.3f}, Q={Q:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'data_collapse.png')
    plt.close(fig)
    print(f"Figure: {out_dir / 'data_collapse.png'}")

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[C2] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT B2 — Hopf Angle vs GRU Training
# ═══════════════════════════════════════════════════════════════

def run_exp_B2():
    """
    Hopf angle theta vs GRU training quality.
    Original model has no gamma coupling — measure theta_system and theta_coupled
    as a function of GRU training steps at lambda_c.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT B2 — HOPF ANGLE vs GRU TRAINING")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_B2'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'

    N = 1024
    lam_c = 1.043
    train_steps_list = [0, 100, 500, 1000, 5000, 10000, 50000]
    n_seeds = 10
    seeds = [42 + i for i in range(n_seeds)]
    warmup = 2000
    measure = 20000
    extra_for_train = 50000

    ckpt = load_json(ckpt_path) or {'results': []}
    existing = {(r['seed'], r['train_steps']) for r in ckpt['results']}
    t0 = time.time()

    for seed in seeds:
        remaining = [ns for ns in train_steps_list
                     if (seed, ns) not in existing]
        if not remaining:
            print(f"  seed={seed}: all done")
            continue

        print(f"\n  seed={seed}: {len(remaining)} points remaining")

        # Create system at lambda_c
        system = OrthogonalRNN(N, lam_c, input_scale=0.01,
                               seed=seed, device=DEVICE)

        # Long trajectory: measurement + GRU training data
        total = measure + extra_for_train
        traj = system.generate_trajectory(total, warmup=warmup)
        measure_traj = traj[:measure]
        train_traj = traj[measure:]

        traj_np = measure_traj.cpu().numpy().astype(np.float64)

        # System-level metrics (independent of GRU)
        psi_ridge = measure_ridge_r2(traj_np)
        lyap = compute_lyapunov_exponent(
            system, steps=10000, warmup=warmup)

        # theta_system from Jacobian at last state
        x_final = measure_traj[-1]
        with torch.no_grad():
            J_sys = system.jacobian(x_final)
            eigs_sys = torch.linalg.eigvals(J_sys)
        max_idx_sys = torch.argmax(torch.abs(eigs_sys))
        mu_sys = eigs_sys[max_idx_sys]
        theta_system = abs(np.degrees(np.arctan2(
            mu_sys.imag.item(), mu_sys.real.item())))
        spec_radius_sys = torch.max(torch.abs(eigs_sys)).item()

        print(f"    theta_sys={theta_system:.2f} deg, "
              f"rho_sys={spec_radius_sys:.4f}, "
              f"Psi_ridge={psi_ridge:.4f}, Lyap={lyap:.4f}")

        for n_train in tqdm(train_steps_list, desc=f"  seed={seed}"):
            if (seed, n_train) in existing:
                continue

            if n_train == 0:
                # Untrained GRU — random weights
                theta_coupled = theta_system
                spec_radius_gru = 0.0
                gru_loss = float('nan')
                gru_r2 = 0.0
            else:
                # Create and train GRU
                model = GRUSelfModel(N, hidden_factor=1.0, n_layers=1)
                model = model.to(DEVICE_SM)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=1e-3, weight_decay=1e-5)

                # Training data
                train_data = train_traj.to(DEVICE_SM)
                seq_len = 32
                n_win = min(train_data.shape[0] - seq_len - 1, 5000)
                if n_win <= 0:
                    continue
                inputs = torch.stack(
                    [train_data[i:i + seq_len] for i in range(n_win)])
                targets = torch.stack(
                    [train_data[i + 1:i + seq_len + 1] for i in range(n_win)])
                dataset = torch.utils.data.TensorDataset(inputs, targets)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=256, shuffle=True, drop_last=True)

                # Train for exactly n_train gradient steps
                model.train()
                step = 0
                losses = []
                done = False
                while not done:
                    for x_b, y_b in loader:
                        if step >= n_train:
                            done = True
                            break
                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda'):
                            preds, _ = model(x_b)
                            loss = nn.functional.mse_loss(preds, y_b)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        losses.append(loss.item())
                        step += 1

                gru_loss = float(np.mean(losses[-min(50, len(losses)):]))

                # GRU prediction R-squared on measurement trajectory
                model.eval()
                m_traj = measure_traj.to(DEVICE_SM)
                with torch.no_grad():
                    preds_list = []
                    h = None
                    for s in range(0, measure - 1, 1000):
                        e = min(s + 1000, measure - 1)
                        chunk = m_traj[s:e].unsqueeze(0)
                        p, h = model(chunk, h)
                        preds_list.append(p.squeeze(0))
                        h = h.detach()
                    preds_all = torch.cat(preds_list, dim=0)
                x_true = m_traj[1:].cpu().numpy()
                x_pred = preds_all.cpu().numpy()
                ml = min(len(x_true), len(x_pred))
                ss_res = np.sum((x_true[:ml] - x_pred[:ml]) ** 2)
                ss_tot = np.sum(
                    (x_true[:ml] - x_true[:ml].mean(0)) ** 2)
                gru_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

                # theta_coupled: GRU hidden-state Jacobian eigenvalues
                try:
                    # Run GRU through trajectory to get realistic h_t
                    h_run = None
                    with torch.no_grad():
                        for s in range(0, min(2000, measure), 100):
                            chunk = m_traj[s:s + 100].unsqueeze(0)
                            _, h_run = model(chunk, h_run)
                            h_run = h_run.detach()

                    hidden_dim = model.hidden_dim
                    x_fixed = m_traj[-2].unsqueeze(0).unsqueeze(0)

                    # Find the GRU/RNN layer
                    _gru_layer = None
                    for attr_name in ('gru', 'rnn', 'recurrent'):
                        if hasattr(model, attr_name):
                            _gru_layer = getattr(model, attr_name)
                            break
                    if _gru_layer is None:
                        raise AttributeError(
                            "Cannot find GRU layer on model "
                            f"(tried: gru, rnn, recurrent). "
                            f"Attributes: {[a for a in dir(model) if not a.startswith('_')]}")

                    # cuDNN GRU requires training mode for backward pass
                    _gru_layer.train()

                    def gru_h_step(h_flat):
                        h_in = h_flat.reshape(1, 1, hidden_dim)
                        gru_out, h_new = _gru_layer(x_fixed, h_in)
                        return h_new.reshape(-1)

                    h_current = h_run.reshape(-1).detach().clone()
                    J_hh = torch.autograd.functional.jacobian(
                        gru_h_step, h_current)
                    _gru_layer.eval()

                    eigs_gru = torch.linalg.eigvals(J_hh)
                    spec_radius_gru = torch.max(
                        torch.abs(eigs_gru)).item()

                    # Combined: eigenvalues from both subsystems
                    # Block-triangular => eigenvalues are union
                    eigs_sys_sm = eigs_sys.to(DEVICE_SM)
                    all_eigs = torch.cat([eigs_sys_sm, eigs_gru])
                    all_mags = torch.abs(all_eigs)
                    idx_lead = torch.argmax(all_mags)
                    mu_cpl = all_eigs[idx_lead]
                    theta_coupled = abs(np.degrees(np.arctan2(
                        mu_cpl.imag.item(), mu_cpl.real.item())))

                except Exception as e:
                    print(f"    Jacobian failed (seed={seed}, "
                          f"steps={n_train}): {e}")
                    theta_coupled = float('nan')
                    spec_radius_gru = float('nan')

                del model, m_traj
                torch.cuda.empty_cache()

            result = {
                'seed': seed,
                'train_steps': n_train,
                'theta_system': float(theta_system),
                'theta_coupled': float(theta_coupled),
                'spec_radius_sys': float(spec_radius_sys),
                'spec_radius_gru': float(spec_radius_gru),
                'psi_ridge': float(psi_ridge),
                'lyapunov': float(lyap),
                'gru_loss': float(gru_loss),
                'gru_r2': float(gru_r2),
            }
            ckpt['results'].append(result)
            existing.add((seed, n_train))
            save_json(ckpt, ckpt_path)

        del system, traj, measure_traj, train_traj
        torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 3600
    print(f"\nB2 data collection: {elapsed:.1f}h")

    # ── Analysis ──────────────────────────────────────────────
    print("\n[B2] Analyzing...")

    by_steps = defaultdict(list)
    for r in ckpt['results']:
        by_steps[r['train_steps']].append(r)

    table_rows = []
    for ns in train_steps_list:
        pts = by_steps.get(ns, [])
        if not pts:
            continue
        table_rows.append({
            'train_steps': ns,
            'theta_sys_mean': float(np.mean(
                [p['theta_system'] for p in pts])),
            'theta_sys_std': float(np.std(
                [p['theta_system'] for p in pts])),
            'theta_cpl_mean': float(np.nanmean(
                [p['theta_coupled'] for p in pts])),
            'theta_cpl_std': float(np.nanstd(
                [p['theta_coupled'] for p in pts])),
            'rho_gru_mean': float(np.nanmean(
                [p['spec_radius_gru'] for p in pts])),
            'rho_gru_std': float(np.nanstd(
                [p['spec_radius_gru'] for p in pts])),
            'gru_r2_mean': float(np.mean(
                [p['gru_r2'] for p in pts])),
            'gru_r2_std': float(np.std(
                [p['gru_r2'] for p in pts])),
            'gru_loss_mean': float(np.nanmean(
                [p['gru_loss'] for p in pts])),
        })

    print("\n" + "-" * 100)
    print(f"{'Steps':>8} | {'theta_sys':>14} | {'theta_coupled':>14} | "
          f"{'rho_GRU':>14} | {'GRU R2':>14}")
    print("-" * 100)
    for row in table_rows:
        print(f"{row['train_steps']:>8} | "
              f"{row['theta_sys_mean']:>6.2f}+/-{row['theta_sys_std']:<5.2f} | "
              f"{row['theta_cpl_mean']:>6.2f}+/-{row['theta_cpl_std']:<5.2f} | "
              f"{row['rho_gru_mean']:>6.3f}+/-{row['rho_gru_std']:<5.3f} | "
              f"{row['gru_r2_mean']:>6.4f}+/-{row['gru_r2_std']:<5.4f}")
    print("-" * 100)

    # Save CSV
    csv_path = out_dir / 'theta_vs_training.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Table: {csv_path}")

    # ── Figures ───────────────────────────────────────────────
    steps_arr = np.array([r['train_steps'] for r in table_rows])
    # Replace 0 with small value for log scale
    steps_plot = np.where(steps_arr == 0, 10, steps_arr).astype(float)

    theta_sys = np.array([r['theta_sys_mean'] for r in table_rows])
    theta_sys_e = np.array([r['theta_sys_std'] for r in table_rows])
    theta_cpl = np.array([r['theta_cpl_mean'] for r in table_rows])
    theta_cpl_e = np.array([r['theta_cpl_std'] for r in table_rows])
    rho_gru = np.array([r['rho_gru_mean'] for r in table_rows])
    rho_gru_e = np.array([r['rho_gru_std'] for r in table_rows])
    gru_r2 = np.array([r['gru_r2_mean'] for r in table_rows])
    gru_r2_e = np.array([r['gru_r2_std'] for r in table_rows])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.errorbar(steps_plot, theta_sys, yerr=theta_sys_e, fmt='o-',
                color='C0', capsize=4, label='theta_system (dynamics only)')
    ax.errorbar(steps_plot, theta_cpl, yerr=theta_cpl_e, fmt='s-',
                color='C3', capsize=4, label='theta_coupled (sys + GRU)')
    ax.set_xscale('log')
    ax.set_xlabel('GRU training steps')
    ax.set_ylabel('Hopf angle theta (degrees)')
    ax.set_title('B2: Hopf Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(steps_plot, rho_gru, yerr=rho_gru_e, fmt='D-',
                color='C2', capsize=4, label='rho(J_GRU)')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('GRU training steps')
    ax.set_ylabel('Spectral radius')
    ax.set_title('B2: GRU Jacobian Spectral Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(steps_plot, gru_r2, yerr=gru_r2_e, fmt='^-',
                color='C4', capsize=4, label='GRU prediction R2')
    ax.set_xscale('log')
    ax.set_xlabel('GRU training steps')
    ax.set_ylabel('R-squared')
    ax.set_title('B2: GRU Prediction Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    gru_loss = np.array([r['gru_loss_mean'] for r in table_rows])
    valid = np.isfinite(gru_loss)
    if np.any(valid):
        ax.plot(steps_plot[valid], gru_loss[valid], 'v-', color='C5',
                markersize=6, label='GRU training loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('GRU training steps')
    ax.set_ylabel('MSE loss')
    ax.set_title('B2: GRU Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'hopf_angle_curve.png')
    plt.close(fig)
    print(f"Figure: {out_dir / 'hopf_angle_curve.png'}")

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[B2] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT A2 — betaG Collapse Ordering
# ═══════════════════════════════════════════════════════════════

def train_gru_biased(model, train_traj, x_star, beta_G,
                     n_epochs=20, seq_len=32, batch_size=256,
                     patience=5, device=None):
    """
    Train GRU with biased loss:
      L = MSE(pred, true) + betaG * MSE(pred, x*)
    Returns final prediction-only loss.
    """
    if device is None:
        device = DEVICE_SM
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    traj = train_traj.to(device)
    T, Nd = traj.shape
    x_star_dev = x_star.to(device)

    n_win = T - seq_len - 1
    if n_win <= 0:
        return float('inf')
    n_win = min(n_win, 15000)
    inputs = torch.stack([traj[i:i + seq_len] for i in range(n_win)])
    targets = torch.stack([traj[i + 1:i + seq_len + 1] for i in range(n_win)])
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # x* target expanded for sequence
    x_star_seq = x_star_dev.unsqueeze(0).unsqueeze(0).expand(
        1, seq_len, Nd)

    best_pred_loss = float('inf')
    wait = 0

    for epoch in range(n_epochs):
        ep_pred_losses = []
        for x_b, y_b in loader:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds, _ = model(x_b)
                loss_pred = nn.functional.mse_loss(preds, y_b)
                if beta_G > 0:
                    loss_goal = nn.functional.mse_loss(
                        preds,
                        x_star_seq.expand(preds.shape[0], -1, -1))
                    loss = loss_pred + beta_G * loss_goal
                else:
                    loss = loss_pred
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_pred_losses.append(loss_pred.item())

        avg_pred = np.mean(ep_pred_losses)
        if avg_pred < best_pred_loss - 1e-6:
            best_pred_loss = avg_pred
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    return best_pred_loss


def evaluate_gru_r2(model, test_traj, device=None):
    """Compute R-squared of GRU predictions on test trajectory."""
    if device is None:
        device = DEVICE_SM
    model.eval()
    traj = test_traj.to(device)
    T = traj.shape[0]

    with torch.no_grad():
        preds_list = []
        h = None
        for s in range(0, T - 1, 500):
            e = min(s + 500, T - 1)
            chunk = traj[s:e].unsqueeze(0)
            p, h = model(chunk, h)
            preds_list.append(p.squeeze(0))
            h = h.detach()
        preds = torch.cat(preds_list, dim=0)

    x_true = traj[1:].cpu().numpy()
    x_pred = preds.cpu().numpy()
    ml = min(len(x_true), len(x_pred))
    ss_res = np.sum((x_true[:ml] - x_pred[:ml]) ** 2)
    ss_tot = np.sum((x_true[:ml] - x_true[:ml].mean(0)) ** 2)
    return max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))


def run_exp_A2():
    """
    betaG collapse ordering — two versions:
    V1: Trajectory-biased GRU training (dynamics unchanged)
    V2: Soft dynamical bias (biased input noise)
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT A2 — betaG COLLAPSE ORDERING")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_A2'
    out_dir.mkdir(exist_ok=True)

    N = 1024
    beta_G_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    seeds = [42, 43, 44]
    n_lam = 60
    lam_arr = np.linspace(0.8, 1.3, n_lam)
    warmup = 2000
    measure = 20000

    # Fixed target vector x*
    rng = np.random.RandomState(12345)
    x_star_np = rng.randn(N).astype(np.float32)
    x_star_np /= np.linalg.norm(x_star_np)
    x_star = torch.from_numpy(x_star_np).to(DEVICE)

    # ────────────────────────────────────────────────────────
    # VERSION 2: Soft dynamical bias (simpler, run first)
    # ────────────────────────────────────────────────────────
    print("\n--- A2 Version 2: Soft dynamical bias ---")

    ckpt_v2_path = out_dir / 'checkpoint_v2.json'
    ckpt_v2 = load_json(ckpt_v2_path) or {}
    t0 = time.time()

    for beta_G in beta_G_values:
        for seed in seeds:
            key = f"bG{beta_G:.4f}_s{seed}"
            if key in ckpt_v2:
                continue

            system = OrthogonalRNN(N, 1.0, input_scale=0.01,
                                   seed=seed, device=DEVICE)
            psi_arr = np.zeros(n_lam)
            lyap_arr = np.zeros(n_lam)
            reff_arr = np.zeros(n_lam)

            desc = f"V2 bG={beta_G:.3f} s={seed}"
            for i, lam in enumerate(tqdm(lam_arr, desc=desc)):
                system.lam = lam

                if beta_G == 0.0:
                    traj = system.generate_trajectory(
                        measure, warmup=warmup)
                else:
                    traj = generate_trajectory_biased(
                        system, measure, warmup, x_star, beta_G)

                traj_np = traj.cpu().numpy().astype(np.float64)
                psi_arr[i] = measure_ridge_r2(traj_np)
                lyap_arr[i] = compute_lyapunov_on_trajectory(
                    system, traj, steps=5000)
                reff_arr[i] = measure_R_eff(traj_np)

                del traj, traj_np

            ckpt_v2[key] = {
                'beta_G': beta_G, 'seed': seed,
                'lambdas': lam_arr.tolist(),
                'psi': psi_arr.tolist(),
                'lyapunov': lyap_arr.tolist(),
                'R_eff': reff_arr.tolist(),
            }
            save_json(ckpt_v2, ckpt_v2_path)

            chi = compute_chi(lam_arr, psi_arr)
            idx = np.argmax(chi)
            print(f"  {key}: lam_c(chi)={lam_arr[idx]:.4f}, "
                  f"chi_max={chi[idx]:.5f}")

            del system
            torch.cuda.empty_cache()

    print(f"V2 time: {(time.time() - t0)/3600:.1f}h")

    # ────────────────────────────────────────────────────────
    # VERSION 1: Trajectory-biased GRU training
    # ────────────────────────────────────────────────────────
    print("\n--- A2 Version 1: Trajectory-biased GRU training ---")

    ckpt_v1_path = out_dir / 'checkpoint_v1.json'
    ckpt_v1 = load_json(ckpt_v1_path) or {}
    t1 = time.time()

    # Step 1: Baselines (independent of betaG)
    if 'baselines' not in ckpt_v1:
        print("  Computing baselines (Ridge R2, Lyapunov)...")
        ckpt_v1['baselines'] = {}
        for seed in seeds:
            system = OrthogonalRNN(N, 1.0, input_scale=0.01,
                                   seed=seed, device=DEVICE)
            psi_b = np.zeros(n_lam)
            lyap_b = np.zeros(n_lam)
            reff_b = np.zeros(n_lam)

            for i, lam in enumerate(tqdm(lam_arr,
                                         desc=f"V1 baseline s={seed}")):
                system.lam = lam
                traj = system.generate_trajectory(measure, warmup=warmup)
                traj_np = traj.cpu().numpy().astype(np.float64)
                psi_b[i] = measure_ridge_r2(traj_np)
                lyap_b[i] = compute_lyapunov_on_trajectory(
                    system, traj, steps=5000)
                reff_b[i] = measure_R_eff(traj_np)
                del traj, traj_np

            ckpt_v1['baselines'][f"s{seed}"] = {
                'psi_ridge': psi_b.tolist(),
                'lyapunov': lyap_b.tolist(),
                'R_eff': reff_b.tolist(),
            }
            save_json(ckpt_v1, ckpt_v1_path)
            del system
            torch.cuda.empty_cache()

    # Step 2: GRU training with biased loss at each (betaG, seed, lambda)
    for beta_G in beta_G_values:
        for seed in seeds:
            key = f"bG{beta_G:.4f}_s{seed}"
            if key in ckpt_v1 and key != 'baselines':
                continue

            system = OrthogonalRNN(N, 1.0, input_scale=0.01,
                                   seed=seed, device=DEVICE)
            gru_r2_arr = np.zeros(n_lam)

            desc = f"V1 bG={beta_G:.3f} s={seed}"
            for i, lam in enumerate(tqdm(lam_arr, desc=desc)):
                system.lam = lam
                traj = system.generate_trajectory(measure, warmup=warmup)

                # Split trajectory
                split = int(0.6 * measure)
                train_traj = traj[:split]
                test_traj = traj[split:]

                # Train GRU with biased loss
                model = GRUSelfModel(N, hidden_factor=1.0, n_layers=1)
                train_gru_biased(model, train_traj, x_star, beta_G,
                                 n_epochs=20, seq_len=32, patience=5)

                # Evaluate GRU R2 on test set
                gru_r2_arr[i] = evaluate_gru_r2(model, test_traj)

                del model, traj, train_traj, test_traj
                torch.cuda.empty_cache()

            ckpt_v1[key] = {
                'beta_G': beta_G, 'seed': seed,
                'gru_r2': gru_r2_arr.tolist(),
            }
            save_json(ckpt_v1, ckpt_v1_path)

            chi_gru = compute_chi(lam_arr, gru_r2_arr)
            idx = np.argmax(chi_gru)
            print(f"  {key}: lam_c(chi_GRU)={lam_arr[idx]:.4f}, "
                  f"chi_max_GRU={chi_gru[idx]:.5f}")

            del system
            torch.cuda.empty_cache()

    print(f"V1 time: {(time.time() - t1)/3600:.1f}h")

    # ── Analysis ──────────────────────────────────────────────
    print("\n[A2] Analyzing...")

    # --- V2 analysis ---
    v2_rows = []
    for beta_G in beta_G_values:
        delta_lams, chi_maxs, reff_peaks = [], [], []
        for seed in seeds:
            key = f"bG{beta_G:.4f}_s{seed}"
            d = ckpt_v2[key]
            lams = np.array(d['lambdas'])
            psis = np.array(d['psi'])
            lyaps = np.array(d['lyapunov'])
            reffs = np.array(d['R_eff'])

            chi = compute_chi(lams, psis)
            idx_chi = np.argmax(chi)
            lam_c_chi = lams[idx_chi]
            lam_c_lyap = find_zero_crossing(lams, lyaps)

            delta_lams.append(lam_c_chi - lam_c_lyap
                              if np.isfinite(lam_c_lyap) else np.nan)
            chi_maxs.append(chi[idx_chi])
            reff_peaks.append(reffs[idx_chi])

        v2_rows.append({
            'beta_G': beta_G,
            'delta_lam_mean': float(np.nanmean(delta_lams)),
            'delta_lam_std': float(np.nanstd(delta_lams)),
            'chi_max_mean': float(np.nanmean(chi_maxs)),
            'chi_max_std': float(np.nanstd(chi_maxs)),
            'R_eff_mean': float(np.nanmean(reff_peaks)),
            'R_eff_std': float(np.nanstd(reff_peaks)),
        })

    # --- V1 analysis ---
    v1_rows = []
    for beta_G in beta_G_values:
        delta_lams, chi_maxs, gru_r2_peaks = [], [], []
        for seed in seeds:
            key = f"bG{beta_G:.4f}_s{seed}"
            baseline = ckpt_v1['baselines'][f"s{seed}"]
            lyaps = np.array(baseline['lyapunov'])
            lam_c_lyap = find_zero_crossing(lam_arr, lyaps)

            gru_r2 = np.array(ckpt_v1[key]['gru_r2'])
            chi_gru = compute_chi(lam_arr, gru_r2)
            idx_chi = np.argmax(chi_gru)
            lam_c_chi_gru = lam_arr[idx_chi]

            delta_lams.append(lam_c_chi_gru - lam_c_lyap
                              if np.isfinite(lam_c_lyap) else np.nan)
            chi_maxs.append(chi_gru[idx_chi])
            gru_r2_peaks.append(gru_r2[idx_chi])

        v1_rows.append({
            'beta_G': beta_G,
            'delta_lam_mean': float(np.nanmean(delta_lams)),
            'delta_lam_std': float(np.nanstd(delta_lams)),
            'chi_max_mean': float(np.nanmean(chi_maxs)),
            'chi_max_std': float(np.nanstd(chi_maxs)),
            'gru_r2_peak_mean': float(np.nanmean(gru_r2_peaks)),
            'gru_r2_peak_std': float(np.nanstd(gru_r2_peaks)),
        })

    # Print tables
    print("\n  === V2: Soft Dynamical Bias ===")
    print("-" * 80)
    print(f"{'bG':>8} | {'Delta_lam':>16} | {'chi_max':>16} | {'R_eff':>16}")
    print("-" * 80)
    for r in v2_rows:
        print(f"{r['beta_G']:>8.3f} | "
              f"{r['delta_lam_mean']:>7.4f}+/-{r['delta_lam_std']:<5.4f} | "
              f"{r['chi_max_mean']:>7.5f}+/-{r['chi_max_std']:<5.5f} | "
              f"{r['R_eff_mean']:>7.1f}+/-{r['R_eff_std']:<5.1f}")

    print("\n  === V1: Trajectory-Biased GRU Training ===")
    print("-" * 80)
    print(f"{'bG':>8} | {'Delta_lam':>16} | {'chi_max_GRU':>16} | "
          f"{'GRU_R2_peak':>16}")
    print("-" * 80)
    for r in v1_rows:
        print(f"{r['beta_G']:>8.3f} | "
              f"{r['delta_lam_mean']:>7.4f}+/-{r['delta_lam_std']:<5.4f} | "
              f"{r['chi_max_mean']:>7.5f}+/-{r['chi_max_std']:<5.5f} | "
              f"{r['gru_r2_peak_mean']:>7.4f}+/-{r['gru_r2_peak_std']:<5.4f}")

    # Save CSVs
    csv_v1 = out_dir / 'collapse_table_v1.csv'
    csv_v2 = out_dir / 'collapse_table_v2.csv'
    for csv_p, rows in [(csv_v1, v1_rows), (csv_v2, v2_rows)]:
        with open(csv_p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Tables: {csv_v1}, {csv_v2}")

    # ── Figures ───────────────────────────────────────────────
    bG = np.array([r['beta_G'] for r in v2_rows])
    bG_plot = bG.copy()
    bG_plot[0] = bG[1] / 5  # offset for log scale

    # --- V2 figure ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    v2_dl = np.array([r['delta_lam_mean'] for r in v2_rows])
    v2_dl_e = np.array([r['delta_lam_std'] for r in v2_rows])
    v2_chi = np.array([r['chi_max_mean'] for r in v2_rows])
    v2_chi_e = np.array([r['chi_max_std'] for r in v2_rows])
    v2_re = np.array([r['R_eff_mean'] for r in v2_rows])
    v2_re_e = np.array([r['R_eff_std'] for r in v2_rows])

    axes[0].errorbar(bG_plot, v2_dl, yerr=v2_dl_e, fmt='o-', color='C0',
                     capsize=3, label='V2')
    axes[0].set_ylabel('Delta_lambda')
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(bG_plot, v2_re, yerr=v2_re_e, fmt='s-', color='C1',
                     capsize=3, label='V2')
    axes[1].set_ylabel('R_eff at peak')
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(bG_plot, v2_chi, yerr=v2_chi_e, fmt='D-', color='C2',
                     capsize=3, label='V2')
    axes[2].set_ylabel('chi_max')
    axes[2].set_xlabel('betaG (teleological pressure)')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)

    axes[0].set_title('A2 V2: Collapse Ordering (Soft Dynamical Bias)')
    fig.tight_layout()
    fig.savefig(out_dir / 'collapse_ordering_v2.png')
    plt.close(fig)

    # --- V1 figure ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    v1_dl = np.array([r['delta_lam_mean'] for r in v1_rows])
    v1_dl_e = np.array([r['delta_lam_std'] for r in v1_rows])
    v1_chi = np.array([r['chi_max_mean'] for r in v1_rows])
    v1_chi_e = np.array([r['chi_max_std'] for r in v1_rows])
    v1_gr = np.array([r['gru_r2_peak_mean'] for r in v1_rows])
    v1_gr_e = np.array([r['gru_r2_peak_std'] for r in v1_rows])

    axes[0].errorbar(bG_plot, v1_dl, yerr=v1_dl_e, fmt='o-', color='C0',
                     capsize=3, label='V1')
    axes[0].set_ylabel('Delta_lambda (GRU)')
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(bG_plot, v1_gr, yerr=v1_gr_e, fmt='s-', color='C4',
                     capsize=3, label='V1')
    axes[1].set_ylabel('GRU R2 at peak')
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(bG_plot, v1_chi, yerr=v1_chi_e, fmt='D-', color='C2',
                     capsize=3, label='V1')
    axes[2].set_ylabel('chi_max (GRU)')
    axes[2].set_xlabel('betaG (teleological pressure)')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)

    axes[0].set_title(
        'A2 V1: Collapse Ordering (Trajectory-Biased GRU Training)')
    fig.tight_layout()
    fig.savefig(out_dir / 'collapse_ordering_v1.png')
    plt.close(fig)

    # --- Comparison figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(bG_plot, v1_dl, yerr=v1_dl_e, fmt='o-', color='C0',
                capsize=3, label='V1: biased GRU training')
    ax.errorbar(bG_plot, v2_dl, yerr=v2_dl_e, fmt='s--', color='C3',
                capsize=3, label='V2: biased dynamics')
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('Delta_lambda')
    ax.set_title('A2: Delta_lambda Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(bG_plot, v1_chi, yerr=v1_chi_e, fmt='o-', color='C0',
                capsize=3, label='V1: chi_max(GRU)')
    ax.errorbar(bG_plot, v2_chi, yerr=v2_chi_e, fmt='s--', color='C3',
                capsize=3, label='V2: chi_max(Ridge)')
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('chi_max')
    ax.set_title('A2: Susceptibility Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'comparison.png')
    plt.close(fig)

    print(f"Figures: {out_dir / 'collapse_ordering_v1.png'}, "
          f"{out_dir / 'collapse_ordering_v2.png'}, "
          f"{out_dir / 'comparison.png'}")

    save_json(ckpt_v1, out_dir / 'raw_data_v1.json')
    save_json(ckpt_v2, out_dir / 'raw_data_v2.json')
    print("[A2] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='IPT Final Experiments v2 (corrected model)')
    parser.add_argument('--exp', type=str, required=True,
                        choices=['A2', 'B2', 'C2', 'all'],
                        help='Which experiment to run')
    args = parser.parse_args()

    print(f"IPT Final Experiments v2 — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"Dynamics device: {DEVICE}")
    print(f"Self-model device: {DEVICE_SM}")
    print()

    # Pre-initialize CUDA contexts
    for i in range(torch.cuda.device_count()):
        torch.zeros(1, device=f'cuda:{i}')
    torch.cuda.synchronize()

    # ── Startup Diagnostic ────────────────────────────────────
    print("=" * 50)
    print("  STARTUP DIAGNOSTIC")
    print("=" * 50)
    _ok = True
    try:
        _sys = OrthogonalRNN(32, 1.0, input_scale=0.01, seed=0, device=DEVICE)
        print(f"  [OK] OrthogonalRNN created (N=32)")

        # Test trajectory generation
        _traj = _sys.generate_trajectory(100, warmup=50)
        print(f"  [OK] generate_trajectory -> shape {_traj.shape}")

        # Test jacobian
        _J = _sys.jacobian(_traj[0])
        print(f"  [OK] jacobian -> shape {_J.shape}")

        # Test eigenvalues
        _eigs = torch.linalg.eigvals(_J)
        _mu = _eigs[torch.argmax(torch.abs(_eigs))]
        _theta = abs(np.degrees(np.arctan2(_mu.imag.item(), _mu.real.item())))
        print(f"  [OK] eigenvalues: spectral radius={torch.max(torch.abs(_eigs)).item():.4f}, theta={_theta:.1f} deg")

        # Test Lyapunov (imported)
        _lyap = compute_lyapunov_exponent(_sys, steps=200, warmup=50)
        print(f"  [OK] compute_lyapunov_exponent -> {_lyap:.4f}")

        # Test Lyapunov (custom, on trajectory)
        _lyap2 = compute_lyapunov_on_trajectory(_sys, _traj, steps=50)
        print(f"  [OK] compute_lyapunov_on_trajectory -> {_lyap2:.4f}")

        # Sanity: check lambda_c is in correct range
        _sys.lam = 1.04
        _traj2 = _sys.generate_trajectory(200, warmup=100)
        _psi = measure_ridge_r2(_traj2.cpu().numpy().astype(np.float64))
        _sys.lam = 0.90
        _traj3 = _sys.generate_trajectory(200, warmup=100)
        _psi0 = measure_ridge_r2(_traj3.cpu().numpy().astype(np.float64))
        print(f"  [OK] Ridge R2: lam=0.90 -> {_psi0:.4f}, lam=1.04 -> {_psi:.4f}")
        if _psi < _psi0:
            print(f"  [WARN] Psi at lam=1.04 < Psi at lam=0.90 — unexpected, "
                  f"but small N=32 can be noisy")

        del _sys, _traj, _traj2, _traj3, _J, _eigs
    except Exception as e:
        print(f"  [FAIL] OrthogonalRNN: {e}")
        _ok = False

    if args.exp in ('B2', 'A2', 'all'):
        try:
            # Test GRUSelfModel construction — try different signatures
            _gru = None
            _constructor_sig = None
            try:
                _gru = GRUSelfModel(32, hidden_factor=1.0, n_layers=1)
                _constructor_sig = "GRUSelfModel(N, hidden_factor=, n_layers=)"
            except TypeError:
                try:
                    _gru = GRUSelfModel(input_dim=32, hidden_dim=32, n_layers=1)
                    _constructor_sig = "GRUSelfModel(input_dim=, hidden_dim=, n_layers=)"
                except TypeError:
                    try:
                        from config import SelfModelConfig
                        _cfg = SelfModelConfig()
                        _gru = GRUSelfModel(32, _cfg)
                        _constructor_sig = "GRUSelfModel(N, config)"
                    except Exception:
                        _gru = GRUSelfModel(32)
                        _constructor_sig = "GRUSelfModel(N)"

            print(f"  [OK] GRUSelfModel created via: {_constructor_sig}")

            # Test forward pass
            _gru = _gru.to(DEVICE)
            _x_test = torch.randn(2, 10, 32, device=DEVICE)  # batch=2, seq=10, N=32
            _out = _gru(_x_test)
            if isinstance(_out, tuple) and len(_out) == 2:
                _preds, _h = _out
                print(f"  [OK] forward -> (preds: {_preds.shape}, hidden: {_h.shape})")
            else:
                print(f"  [WARN] forward returns {type(_out)}, expected (preds, hidden) tuple")
                print(f"         You may need to adjust model() calls in B2/A2")
                _ok = False

            # Test attributes
            _has_gru = hasattr(_gru, 'gru') or hasattr(_gru, 'rnn')
            _gru_attr = 'gru' if hasattr(_gru, 'gru') else ('rnn' if hasattr(_gru, 'rnn') else 'MISSING')
            print(f"  [{'OK' if _has_gru else 'WARN'}] GRU layer attribute: .{_gru_attr}")
            if not _has_gru:
                print("         B2 theta_coupled computation will fail — skip B2 or fix")

            try:
                _hdim = _gru.hidden_dim
                print(f"  [OK] hidden_dim = {_hdim}")
            except Exception as e:
                print(f"  [WARN] hidden_dim: {e}")

            del _gru
        except Exception as e:
            print(f"  [FAIL] GRUSelfModel: {e}")
            import traceback
            traceback.print_exc()
            _ok = False

    torch.cuda.empty_cache()

    if not _ok:
        print("\n  DIAGNOSTIC FAILED — fix issues above before running.")
        print("  Aborting.")
        sys.exit(1)
    print("\n  ALL CHECKS PASSED — starting experiments\n")
    print("=" * 50)

    t0 = time.time()

    if args.exp in ('C2', 'all'):
        run_exp_C2()
    if args.exp in ('B2', 'all'):
        run_exp_B2()
    if args.exp in ('A2', 'all'):
        run_exp_A2()

    total_h = (time.time() - t0) / 3600
    print(f"\nTotal runtime: {total_h:.2f} hours")
    print("ALL DONE.")


if __name__ == '__main__':
    main()
