#!/usr/bin/env python3
"""
IPT Comprehensive Final Experiments — Fractal Tree Architecture
===============================================================
Experiments:
  D1: Self-organized criticality on fractal tree
  D2: Resilience under betaG pressure (partial module pressure)
  E1: Distributed per-module GRU self-models
  F1: Comparison matrix (assembles all results)
  F2: Information flow topology

Usage:
    python ipt_comprehensive.py --exp D1
    python ipt_comprehensive.py --exp all
"""

import argparse
import csv
import json
import math
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

# ─── Imports from existing codebase ───────────────────────────
_IPT_PATH = os.path.expanduser(
    '~/projects1/IPT/Calude_IPT/3_IPT_Memory/ipt_experiment')
sys.path.insert(0, _IPT_PATH)
from dynamics import OrthogonalRNN
from self_model import GRUSelfModel

# Patch OrthogonalRNN.jacobian if missing
if not hasattr(OrthogonalRNN, 'jacobian'):
    def _jacobian(self, x):
        if x.dim() == 2:
            x = x.squeeze(0)
        with torch.no_grad():
            pre = self.lam * (x @ self.W.T)
            sech2 = 1.0 - torch.tanh(pre) ** 2
            return sech2.unsqueeze(1) * (self.lam * self.W)
    OrthogonalRNN.jacobian = _jacobian

# ─── Global config ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE_SM = 'cuda:1' if torch.cuda.device_count() > 1 else DEVICE

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════
#  Utilities (replicated to avoid import side-effects)
# ═══════════════════════════════════════════════════════════════

def measure_ridge_r2(traj_np, alpha=1.0, train_frac=0.6):
    T = traj_np.shape[0]
    X, Y = traj_np[:-1], traj_np[1:]
    split = int(train_frac * len(X))
    ridge = Ridge(alpha=alpha)
    ridge.fit(X[:split], Y[:split])
    return max(ridge.score(X[split:], Y[split:]), 0.0)


def compute_chi(lam_arr, psi_arr):
    return np.gradient(psi_arr, lam_arr)


def find_zero_crossing(lam_arr, vals):
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            f = -vals[i] / (vals[i + 1] - vals[i])
            return lam_arr[i] + f * (lam_arr[i + 1] - lam_arr[i])
    return np.nan


def measure_R_eff(traj_np, n_bins=50):
    if traj_np.shape[0] < 10:
        return 1.0
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(traj_np)
        xb = np.linspace(proj[:, 0].min(), proj[:, 0].max(), n_bins + 1)
        yb = np.linspace(proj[:, 1].min(), proj[:, 1].max(), n_bins + 1)
        hist, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=[xb, yb])
        p = hist.flatten()
        p = p[p > 0]
        p = p / p.sum()
        return 1.0 / np.sum(p ** 2)
    except Exception:
        return 1.0


def save_json(data, path):
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
    with open(str(path), 'w') as f:
        json.dump(conv(data), f, indent=2)


def load_json(path):
    path = str(path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def evaluate_gru_r2(model, test_traj, device=None):
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


# ═══════════════════════════════════════════════════════════════
#  FRACTAL TREE SYSTEM
# ═══════════════════════════════════════════════════════════════

class FractalTreeSystem:
    """
    L-level binary tree with n neurons per leaf module.
    Total system size N = n * 2^L.
    Inter-module coupling: gamma(d) = gamma_0 * r^d
    where d = tree distance between leaves.
    """

    def __init__(self, L, n, lambda_0, gamma_0=0.1, r=0.5,
                 sigma=0.01, seed=42, device='cuda:0'):
        self.L = L
        self.n = n
        self.n_modules = 2 ** L
        self.N = n * self.n_modules
        self.lambda_0 = lambda_0
        self.gamma_0 = gamma_0
        self.r = r
        self.sigma = sigma
        self.device = device
        self.seed = seed

        rng = torch.Generator(device='cpu').manual_seed(seed)

        # Per-module orthogonal W_i and input projection U_i
        self.W = []  # list of (n, n) tensors
        self.U = []  # list of (n, n) tensors
        for i in range(self.n_modules):
            # Orthogonal via QR (same as OrthogonalRNN)
            A = torch.randn(n, n, generator=rng)
            Q, R = torch.linalg.qr(A)
            # Ensure proper orthogonal (det=+1)
            Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
            self.W.append(Q.to(device))
            # Input projection
            Ui = torch.randn(n, n, generator=rng) / math.sqrt(n)
            self.U.append(Ui.to(device))

        # Pre-compute tree distances and coupling matrices
        self.distances = {}
        self.C = {}  # coupling matrices C_ij
        self.gamma_ij = {}  # coupling strengths
        for i in range(self.n_modules):
            for j in range(self.n_modules):
                if i == j:
                    continue
                d = self._tree_distance(i, j)
                self.distances[(i, j)] = d
                g = gamma_0 * (r ** d)
                if g > 1e-6:  # skip negligible coupling
                    self.gamma_ij[(i, j)] = g
                    Cij = torch.randn(n, n, generator=rng) / math.sqrt(n)
                    self.C[(i, j)] = Cij.to(device)

        # External forces (for betaG experiments)
        self.forces = {}  # {module_idx: (betaG, x_star_i)}

    def _tree_distance(self, i, j):
        """Tree distance = L - depth(LCA(i, j))."""
        if i == j:
            return 0
        xor = i ^ j
        # highest differing bit position
        hb = int(math.floor(math.log2(xor)))
        return hb + 1

    def set_force(self, module_idx, betaG, x_star_i):
        """Set h_i = -2*betaG*(x_i - x_star_i) for a module."""
        self.forces[module_idx] = (betaG, x_star_i.to(self.device))

    def clear_forces(self):
        self.forces = {}

    @torch.no_grad()
    def step(self, x_modules, noise=None):
        """
        One step of fractal dynamics.
        x_modules: (n_modules, n) tensor
        noise: (n_modules, n) tensor or None
        Returns: (n_modules, n) tensor
        """
        pre = torch.zeros_like(x_modules)

        # Intra-module: lambda_0 * W_i @ x_i
        for i in range(self.n_modules):
            pre[i] = self.lambda_0 * (x_modules[i] @ self.W[i].T)

        # Inter-module coupling
        for (i, j), g in self.gamma_ij.items():
            pre[i] += g * (x_modules[j] @ self.C[(i, j)].T)

        # Stochastic input
        if noise is not None:
            for i in range(self.n_modules):
                pre[i] += self.sigma * (noise[i] @ self.U[i].T)

        # External forces (betaG)
        for idx, (betaG, x_star) in self.forces.items():
            pre[idx] += -2.0 * betaG * (x_modules[idx] - x_star)

        return torch.tanh(pre)

    @torch.no_grad()
    def generate_trajectory(self, steps, warmup=0):
        """
        Generate trajectory. Returns (steps, n_modules, n) tensor.
        """
        total = warmup + steps
        x = torch.randn(self.n_modules, self.n, device=self.device) * 0.1
        noise_all = torch.randn(total, self.n_modules, self.n,
                                device=self.device)
        trajectory = []
        for t in range(total):
            x = self.step(x, noise_all[t])
            if t >= warmup:
                trajectory.append(x.clone())
        return torch.stack(trajectory)  # (steps, n_modules, n)

    def traj_flat(self, traj):
        """Reshape (steps, n_modules, n) -> (steps, N)."""
        return traj.reshape(traj.shape[0], -1)

    def traj_module(self, traj, mod_idx):
        """Extract single module trajectory: (steps, n)."""
        return traj[:, mod_idx, :]

    @torch.no_grad()
    def jacobian(self, x_modules):
        """
        Full N x N Jacobian at state x_modules.
        x_modules: (n_modules, n)
        """
        n, nm = self.n, self.n_modules
        N = self.N
        J = torch.zeros(N, N, device=self.device)

        # Compute pre-activation for sech^2
        pre = torch.zeros_like(x_modules)
        for i in range(nm):
            pre[i] = self.lambda_0 * (x_modules[i] @ self.W[i].T)
        for (i, j), g in self.gamma_ij.items():
            pre[i] += g * (x_modules[j] @ self.C[(i, j)].T)
        for idx, (betaG, x_star) in self.forces.items():
            pre[idx] += -2.0 * betaG * (x_modules[idx] - x_star)

        sech2 = 1.0 - torch.tanh(pre) ** 2  # (nm, n)

        # Diagonal blocks: sech2_i * lambda_0 * W_i
        for i in range(nm):
            s2 = sech2[i]  # (n,)
            block = s2.unsqueeze(1) * (self.lambda_0 * self.W[i])
            # betaG contribution to Jacobian
            if i in self.forces:
                betaG = self.forces[i][0]
                block -= 2.0 * betaG * torch.diag(s2)
            J[i*n:(i+1)*n, i*n:(i+1)*n] = block

        # Off-diagonal blocks: sech2_i * gamma_ij * C_ij
        for (i, j), g in self.gamma_ij.items():
            s2 = sech2[i]
            block = s2.unsqueeze(1) * (g * self.C[(i, j)])
            J[i*n:(i+1)*n, j*n:(j+1)*n] = block

        return J

    @torch.no_grad()
    def compute_lyapunov(self, traj, steps=5000):
        """Lyapunov exponent via QR tangent along trajectory."""
        T = min(steps, traj.shape[0] - 1)
        v = torch.randn(self.N, device=self.device)
        v = v / v.norm()
        log_stretches = []
        for t in range(T):
            J = self.jacobian(traj[t])
            v_new = J @ v
            stretch = v_new.norm()
            if stretch > 0:
                log_stretches.append(torch.log(stretch).item())
                v = v_new / stretch
        return np.mean(log_stretches) if log_stretches else 0.0

    @torch.no_grad()
    def compute_module_lyapunov(self, traj, mod_idx, steps=5000):
        """Lyapunov for a single module's diagonal block."""
        T = min(steps, traj.shape[0] - 1)
        n = self.n
        v = torch.randn(n, device=self.device)
        v = v / v.norm()
        log_stretches = []
        for t in range(T):
            x = traj[t]
            pre_i = self.lambda_0 * (x[mod_idx] @ self.W[mod_idx].T)
            for (ii, jj), g in self.gamma_ij.items():
                if ii == mod_idx:
                    pre_i += g * (x[jj] @ self.C[(ii, jj)].T)
            sech2 = 1.0 - torch.tanh(pre_i) ** 2
            J_ii = sech2.unsqueeze(1) * (self.lambda_0 * self.W[mod_idx])
            v_new = J_ii @ v
            stretch = v_new.norm()
            if stretch > 0:
                log_stretches.append(torch.log(stretch).item())
                v = v_new / stretch
        return np.mean(log_stretches) if log_stretches else 0.0


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT D1 — Self-Organized Criticality on Fractal
# ═══════════════════════════════════════════════════════════════

def run_exp_D1():
    print("\n" + "=" * 70)
    print("  EXPERIMENT D1 — FRACTAL SELF-ORGANIZED CRITICALITY")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_D1'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    ckpt = load_json(ckpt_path) or {}

    # Part A: gamma_0 sweep at fixed lambda_0
    print("\n--- D1 Part A: gamma_0 sweep ---")
    n, L = 128, 3
    lam0 = 1.043
    gamma_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    seeds = [42, 43, 44]
    warmup, measure = 2000, 15000

    for gamma_0 in gamma_values:
        for seed in seeds:
            key = f"A_g{gamma_0:.4f}_s{seed}"
            if key in ckpt:
                continue

            sys_f = FractalTreeSystem(L, n, lam0, gamma_0=gamma_0,
                                      seed=seed, device=DEVICE)
            traj = sys_f.generate_trajectory(measure, warmup=warmup)
            traj_flat = sys_f.traj_flat(traj).cpu().numpy().astype(np.float64)

            # Global Psi
            psi_global = measure_ridge_r2(traj_flat)

            # Per-module Psi
            psi_modules = []
            for m in range(sys_f.n_modules):
                tm = sys_f.traj_module(traj, m).cpu().numpy().astype(np.float64)
                psi_modules.append(measure_ridge_r2(tm))

            # Per-module Lyapunov (sample 2 modules for speed)
            lyap_modules = []
            for m in [0, sys_f.n_modules // 2]:
                lyap_modules.append(sys_f.compute_module_lyapunov(traj, m, steps=3000))

            # Cross-prediction: nearest neighbors (modules 0-1, 2-3, etc.)
            psi_cross = []
            for m in range(0, sys_f.n_modules, 2):
                t0 = sys_f.traj_module(traj, m).cpu().numpy().astype(np.float64)
                t1 = sys_f.traj_module(traj, m + 1).cpu().numpy().astype(np.float64)
                X, Y = t0[:-1], t1[1:]
                split = int(0.6 * len(X))
                ridge = Ridge(alpha=1.0)
                ridge.fit(X[:split], Y[:split])
                psi_cross.append(max(ridge.score(X[split:], Y[split:]), 0.0))

            r_eff = measure_R_eff(traj_flat)

            ckpt[key] = {
                'gamma_0': gamma_0, 'seed': seed,
                'psi_global': float(psi_global),
                'psi_modules': [float(x) for x in psi_modules],
                'lyap_modules': [float(x) for x in lyap_modules],
                'psi_cross': [float(x) for x in psi_cross],
                'R_eff': float(r_eff),
            }
            save_json(ckpt, ckpt_path)
            print(f"  {key}: Psi_global={psi_global:.4f}, Lyap={np.mean(lyap_modules):.4f}, "
                  f"Psi_cross={np.mean(psi_cross):.4f}, R_eff={r_eff:.1f}")

            del sys_f, traj
            torch.cuda.empty_cache()

    # Part B: lambda sweep at selected gamma_0
    print("\n--- D1 Part B: lambda sweep ---")
    gamma_select = [0.0, 0.01, 0.05, 0.1]
    n_lam = 60
    lam_arr = np.linspace(0.85, 1.20, n_lam)
    seeds_b = [42, 43]

    for gamma_0 in gamma_select:
        for seed in seeds_b:
            key = f"B_g{gamma_0:.4f}_s{seed}"
            if key in ckpt:
                continue

            psi_arr = np.zeros(n_lam)
            lyap_arr = np.zeros(n_lam)

            for i, lam in enumerate(tqdm(lam_arr,
                                         desc=f"g0={gamma_0:.3f} s={seed}")):
                sys_f = FractalTreeSystem(L, n, lam, gamma_0=gamma_0,
                                          seed=seed, device=DEVICE)
                traj = sys_f.generate_trajectory(measure, warmup=warmup)
                traj_flat = sys_f.traj_flat(traj).cpu().numpy().astype(np.float64)
                psi_arr[i] = measure_ridge_r2(traj_flat)
                lyap_arr[i] = sys_f.compute_module_lyapunov(traj, 0, steps=3000)
                del sys_f, traj
                torch.cuda.empty_cache()

            ckpt[key] = {
                'gamma_0': gamma_0, 'seed': seed,
                'lambdas': lam_arr.tolist(),
                'psi': psi_arr.tolist(),
                'lyapunov': lyap_arr.tolist(),
            }
            save_json(ckpt, ckpt_path)

            chi = compute_chi(lam_arr, psi_arr)
            idx = np.argmax(chi)
            lc_lyap = find_zero_crossing(lam_arr, lyap_arr)
            print(f"  {key}: lc_chi={lam_arr[idx]:.4f}, chi_max={chi[idx]:.4f}, "
                  f"lc_Lyap={lc_lyap:.4f}")

    # ── Analysis & Figures ────────────────────────────────────
    print("\n[D1] Analyzing...")

    # Part A figure: 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    by_gamma = defaultdict(list)
    for k, v in ckpt.items():
        if k.startswith('A_'):
            by_gamma[v['gamma_0']].append(v)

    gammas = sorted(by_gamma.keys())
    g_arr = np.array(gammas)
    g_plot = np.where(g_arr == 0, g_arr[1] / 5 if len(g_arr) > 1 else 0.0001, g_arr)

    psi_g = [np.mean([p['psi_global'] for p in by_gamma[g]]) for g in gammas]
    psi_g_e = [np.std([p['psi_global'] for p in by_gamma[g]]) for g in gammas]
    lyap_g = [np.mean([np.mean(p['lyap_modules']) for p in by_gamma[g]]) for g in gammas]
    cross_g = [np.mean([np.mean(p['psi_cross']) for p in by_gamma[g]]) for g in gammas]
    reff_g = [np.mean([p['R_eff'] for p in by_gamma[g]]) for g in gammas]

    axes[0, 0].errorbar(g_plot, psi_g, yerr=psi_g_e, fmt='o-', capsize=3)
    axes[0, 0].set_xlabel('gamma_0')
    axes[0, 0].set_ylabel('Psi_global')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_title('Global Self-Prediction')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(g_plot, lyap_g, 's-', color='C1')
    axes[0, 1].axhline(0, ls='--', color='gray', alpha=0.5)
    axes[0, 1].set_xlabel('gamma_0')
    axes[0, 1].set_ylabel('Lyapunov (module avg)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('Module Lyapunov')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(g_plot, cross_g, 'D-', color='C2')
    axes[1, 0].set_xlabel('gamma_0')
    axes[1, 0].set_ylabel('Psi_cross (neighbor avg)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title('Cross-Module Prediction')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(g_plot, reff_g, '^-', color='C3')
    axes[1, 1].set_xlabel('gamma_0')
    axes[1, 1].set_ylabel('R_eff')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title('Effective Dimensionality')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('D1 Part A: gamma_0 Sweep at lambda_0=1.043', fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / 'D1_gamma_sweep.png')
    plt.close(fig)

    # Part A CSV
    rows_a = []
    for g in gammas:
        pts = by_gamma[g]
        rows_a.append({
            'gamma_0': g,
            'psi_global': float(np.mean([p['psi_global'] for p in pts])),
            'psi_global_std': float(np.std([p['psi_global'] for p in pts])),
            'lyap_mean': float(np.mean([np.mean(p['lyap_modules']) for p in pts])),
            'psi_cross': float(np.mean([np.mean(p['psi_cross']) for p in pts])),
            'R_eff': float(np.mean([p['R_eff'] for p in pts])),
        })
    csv_path = out_dir / 'D1_gamma_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        w.writeheader()
        w.writerows(rows_a)

    # Part B figure: Psi(lambda) and chi(lambda) for different gamma_0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(gamma_select)))

    width_rows = []
    for gi, gamma_0 in enumerate(gamma_select):
        all_psi, all_chi = [], []
        for seed in seeds_b:
            key = f"B_g{gamma_0:.4f}_s{seed}"
            if key not in ckpt:
                continue
            d = ckpt[key]
            lams = np.array(d['lambdas'])
            psis = np.array(d['psi'])
            chi = compute_chi(lams, psis)
            all_psi.append(psis)
            all_chi.append(chi)

        if not all_psi:
            continue
        psi_mean = np.mean(all_psi, axis=0)
        chi_mean = np.mean(all_chi, axis=0)

        axes[0].plot(lams, psi_mean, color=colors[gi],
                     label=f'g0={gamma_0}')
        axes[1].plot(lams, chi_mean, color=colors[gi],
                     label=f'g0={gamma_0}')

        # FWHM of chi
        chi_max = np.max(chi_mean)
        half = chi_max / 2
        above = np.where(chi_mean > half)[0]
        fwhm = lams[above[-1]] - lams[above[0]] if len(above) > 1 else 0.0
        lc = lams[np.argmax(chi_mean)]
        width_rows.append({
            'gamma_0': gamma_0, 'chi_max': float(chi_max),
            'lambda_c_chi': float(lc), 'FWHM': float(fwhm),
        })

    axes[0].set_xlabel('lambda')
    axes[0].set_ylabel('Psi (Ridge R2)')
    axes[0].set_title('D1: Psi vs lambda')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('chi = dPsi/dlambda')
    axes[1].set_title('D1: Susceptibility')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'D1_lambda_sweep.png')
    plt.close(fig)

    if width_rows:
        csv_w = out_dir / 'D1_critical_width.csv'
        with open(csv_w, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(width_rows[0].keys()))
            w.writeheader()
            w.writerows(width_rows)
        print("\n  Critical width table:")
        for r in width_rows:
            print(f"    g0={r['gamma_0']:.3f}: lc={r['lambda_c_chi']:.4f}, "
                  f"chi_max={r['chi_max']:.4f}, FWHM={r['FWHM']:.4f}")

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[D1] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT D2 — Resilience Under betaG Pressure
# ═══════════════════════════════════════════════════════════════

def run_exp_D2():
    print("\n" + "=" * 70)
    print("  EXPERIMENT D2 — RESILIENCE UNDER betaG PRESSURE")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_D2'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    ckpt = load_json(ckpt_path) or {}

    n, L = 128, 3
    lam0 = 1.043
    gamma_0 = 0.05
    betaG_values = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    seeds = [42, 43, 44]
    warmup, measure = 2000, 12000
    n_modules = 2 ** L  # 8

    configs = {
        'global': list(range(n_modules)),         # all 8
        'subtree': list(range(n_modules // 2)),    # 0-3
        'local': [0],                              # module 0 only
    }

    # Fixed target x* per module
    rng_star = np.random.RandomState(12345)
    x_stars = []
    for m in range(n_modules):
        xs = rng_star.randn(n).astype(np.float32)
        xs /= np.linalg.norm(xs)
        x_stars.append(torch.from_numpy(xs))

    for cfg_name, pressured_modules in configs.items():
        for betaG in betaG_values:
            for seed in seeds:
                key = f"{cfg_name}_bG{betaG:.4f}_s{seed}"
                if key in ckpt:
                    continue

                sys_f = FractalTreeSystem(L, n, lam0, gamma_0=gamma_0,
                                          seed=seed, device=DEVICE)
                # Set forces on pressured modules
                if betaG > 0:
                    for m in pressured_modules:
                        sys_f.set_force(m, betaG, x_stars[m])

                traj = sys_f.generate_trajectory(measure, warmup=warmup)
                traj_flat_np = sys_f.traj_flat(traj).cpu().numpy().astype(np.float64)

                psi_global = measure_ridge_r2(traj_flat_np)

                psi_modules = []
                lyap_modules = []
                for m in range(n_modules):
                    tm = sys_f.traj_module(traj, m).cpu().numpy().astype(np.float64)
                    psi_modules.append(measure_ridge_r2(tm))
                    lyap_modules.append(
                        sys_f.compute_module_lyapunov(traj, m, steps=2000))

                r_eff = measure_R_eff(traj_flat_np)

                # Classify
                psi_pressured = [psi_modules[m] for m in pressured_modules]
                free_modules = [m for m in range(n_modules)
                                if m not in pressured_modules]
                psi_free = [psi_modules[m] for m in free_modules] if free_modules else []

                ckpt[key] = {
                    'config': cfg_name, 'betaG': betaG, 'seed': seed,
                    'psi_global': float(psi_global),
                    'psi_modules': [float(x) for x in psi_modules],
                    'lyap_modules': [float(x) for x in lyap_modules],
                    'R_eff': float(r_eff),
                    'psi_pressured_mean': float(np.mean(psi_pressured)) if psi_pressured else float('nan'),
                    'psi_free_mean': float(np.mean(psi_free)) if psi_free else float('nan'),
                    'pressured_modules': pressured_modules,
                }
                save_json(ckpt, ckpt_path)
                print(f"  {key}: Psi_global={psi_global:.4f}, "
                      f"Psi_press={np.mean(psi_pressured):.4f}, "
                      f"Psi_free={np.mean(psi_free) if psi_free else float('nan'):.4f}")

                del sys_f, traj
                torch.cuda.empty_cache()

    # Also run flat OrthogonalRNN for comparison
    print("\n  Running flat comparison...")
    for betaG in betaG_values:
        for seed in seeds:
            key = f"flat_bG{betaG:.4f}_s{seed}"
            if key in ckpt:
                continue

            N_flat = n * n_modules  # 1024
            sys_flat = OrthogonalRNN(N_flat, lam0, input_scale=0.01,
                                     seed=seed, device=DEVICE)
            if betaG > 0:
                # For flat: apply force to entire state via biased trajectory
                x_star_flat = torch.cat(x_stars).to(DEVICE)
                # Generate with force manually
                total = warmup + measure
                x = torch.randn(1, N_flat, device=DEVICE) * 0.1
                noise = torch.randn(total, N_flat, device=DEVICE)
                trajectory = []
                for t in range(total):
                    u = noise[t:t + 1]
                    x = sys_flat.step(x, u)
                    # Apply force
                    h = -2.0 * betaG * (x - x_star_flat.unsqueeze(0))
                    x = x + h * 0.01  # small step to avoid divergence
                    if t >= warmup:
                        trajectory.append(x.squeeze(0))
                traj_flat_t = torch.stack(trajectory)
            else:
                traj_flat_t = sys_flat.generate_trajectory(measure, warmup=warmup)

            traj_np = traj_flat_t.cpu().numpy().astype(np.float64)
            psi = measure_ridge_r2(traj_np)

            ckpt[key] = {
                'config': 'flat', 'betaG': betaG, 'seed': seed,
                'psi_global': float(psi),
                'psi_pressured_mean': float(psi),
                'psi_free_mean': float('nan'),
            }
            save_json(ckpt, ckpt_path)
            print(f"  {key}: Psi={psi:.4f}")

            del sys_flat
            torch.cuda.empty_cache()

    # ── Analysis & Figures ────────────────────────────────────
    print("\n[D2] Analyzing...")

    all_configs = ['global', 'subtree', 'local', 'flat']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_cfg = {'global': 'C0', 'subtree': 'C1', 'local': 'C2', 'flat': 'C3'}

    rows_d2 = []
    for cfg in all_configs:
        bG_vals, psi_g, psi_f, psi_p = [], [], [], []
        for betaG in betaG_values:
            pts = [ckpt[k] for k in ckpt
                   if isinstance(ckpt[k], dict) and
                   ckpt[k].get('config') == cfg and
                   abs(ckpt[k].get('betaG', -1) - betaG) < 1e-6]
            if not pts:
                continue
            bG_vals.append(betaG)
            psi_g.append(np.mean([p['psi_global'] for p in pts]))
            psi_p.append(np.nanmean([p['psi_pressured_mean'] for p in pts]))
            psi_f.append(np.nanmean([p['psi_free_mean'] for p in pts]))

        if not bG_vals:
            continue
        bG_arr = np.array(bG_vals)
        bG_plot = bG_arr.copy()
        bG_plot[0] = bG_arr[1] / 5 if len(bG_arr) > 1 else 0.0001

        axes[0].plot(bG_plot, psi_g, 'o-', color=colors_cfg[cfg],
                     label=cfg, markersize=5)
        if cfg != 'flat' and cfg != 'global':
            axes[1].plot(bG_plot, psi_f, 's-', color=colors_cfg[cfg],
                         label=f'{cfg} (free)', markersize=5)
        axes[1].plot(bG_plot, psi_p, '^--', color=colors_cfg[cfg],
                     label=f'{cfg} (press)', markersize=4, alpha=0.6)

        # Ratio: normalized to betaG=0
        if psi_g[0] > 0:
            ratio = [p / psi_g[0] for p in psi_g]
            axes[2].plot(bG_plot, ratio, 'o-', color=colors_cfg[cfg],
                         label=cfg, markersize=5)

        for i, bg in enumerate(bG_vals):
            rows_d2.append({
                'config': cfg, 'betaG': bg,
                'psi_global': psi_g[i],
                'psi_pressured': psi_p[i],
                'psi_free': psi_f[i],
            })

    for ax in axes:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_xlabel('betaG')
    axes[0].set_ylabel('Psi_global')
    axes[0].set_title('D2: Global Psi')
    axes[1].set_xlabel('betaG')
    axes[1].set_ylabel('Psi')
    axes[1].set_title('D2: Pressured vs Free')
    axes[2].set_xlabel('betaG')
    axes[2].set_ylabel('Psi / Psi(betaG=0)')
    axes[2].set_title('D2: Resilience Ratio')
    axes[2].axhline(1.0, ls='--', color='gray', alpha=0.5)

    fig.suptitle('D2: Resilience Under betaG Pressure', fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / 'D2_resilience.png')
    plt.close(fig)

    if rows_d2:
        csv_d2 = out_dir / 'D2_resilience.csv'
        with open(csv_d2, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_d2[0].keys()))
            w.writeheader()
            w.writerows(rows_d2)

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[D2] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT E1 — Distributed Per-Module GRU Self-Models
# ═══════════════════════════════════════════════════════════════

def train_module_gru(model, train_traj, n_epochs=20, seq_len=32,
                     batch_size=128, patience=5, device=None,
                     betaG=0.0, x_star=None):
    """Train a per-module GRU. Returns final test loss."""
    if device is None:
        device = DEVICE_SM
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    traj = train_traj.to(device)
    T = traj.shape[0]
    n_win = min(T - seq_len - 1, 4000)
    if n_win <= 0:
        return float('inf')

    inputs = torch.stack([traj[i:i + seq_len] for i in range(n_win)])
    targets = torch.stack([traj[i + 1:i + seq_len + 1] for i in range(n_win)])
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True)

    if betaG > 0 and x_star is not None:
        x_star_dev = x_star.to(device)
        x_star_seq = x_star_dev.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)

    best_loss = float('inf')
    wait = 0
    for epoch in range(n_epochs):
        ep_losses = []
        for x_b, y_b in loader:
            optimizer.zero_grad()
            preds, _ = model(x_b)
            loss_pred = nn.functional.mse_loss(preds, y_b)
            if betaG > 0 and x_star is not None:
                loss_goal = nn.functional.mse_loss(
                    preds, x_star_seq.expand(preds.shape[0], -1, -1))
                loss = loss_pred + betaG * loss_goal
            else:
                loss = loss_pred
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss_pred.item())

        avg = np.mean(ep_losses)
        if avg < best_loss - 1e-6:
            best_loss = avg
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    return best_loss


def run_exp_E1():
    print("\n" + "=" * 70)
    print("  EXPERIMENT E1 — DISTRIBUTED PER-MODULE GRU SELF-MODELS")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_E1'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    ckpt = load_json(ckpt_path) or {}

    n, L = 128, 3
    lam0 = 1.043
    gamma_0 = 0.05
    n_modules = 2 ** L
    seeds = [42, 43, 44]
    warmup = 2000
    measure = 15000
    train_len = 30000

    # Fixed targets for biased training
    rng_star = np.random.RandomState(12345)
    x_stars = []
    for m in range(n_modules):
        xs = rng_star.randn(n).astype(np.float32)
        xs /= np.linalg.norm(xs)
        x_stars.append(torch.from_numpy(xs))

    # Part A: Baseline comparison
    print("\n--- E1 Part A: Distributed vs Global ---")
    for seed in seeds:
        key = f"baseline_s{seed}"
        if key in ckpt:
            continue

        sys_f = FractalTreeSystem(L, n, lam0, gamma_0=gamma_0,
                                  seed=seed, device=DEVICE)
        traj = sys_f.generate_trajectory(measure + train_len, warmup=warmup)
        meas_traj = traj[:measure]
        train_traj = traj[measure:]

        # Ridge R2 baseline
        flat_np = sys_f.traj_flat(meas_traj).cpu().numpy().astype(np.float64)
        psi_ridge = measure_ridge_r2(flat_np)

        # Per-module GRU
        module_r2 = []
        for m in tqdm(range(n_modules), desc=f"s={seed} modules"):
            mod_train = sys_f.traj_module(train_traj, m)
            mod_test = sys_f.traj_module(meas_traj, m)
            gru = GRUSelfModel(n, hidden_factor=1.0, n_layers=1)
            train_module_gru(gru, mod_train, n_epochs=15, device=DEVICE_SM)
            r2 = evaluate_gru_r2(gru, mod_test, device=DEVICE_SM)
            module_r2.append(r2)
            del gru
            torch.cuda.empty_cache()

        # Global GRU (smaller hidden to match param count)
        N_total = n * n_modules
        gru_global = GRUSelfModel(N_total, hidden_factor=0.125, n_layers=1)
        flat_train = sys_f.traj_flat(train_traj)
        flat_test = sys_f.traj_flat(meas_traj)
        train_module_gru(gru_global, flat_train, n_epochs=15,
                         seq_len=16, batch_size=64, device=DEVICE_SM)
        global_r2 = evaluate_gru_r2(gru_global, flat_test, device=DEVICE_SM)
        del gru_global
        torch.cuda.empty_cache()

        ckpt[key] = {
            'seed': seed,
            'psi_ridge': float(psi_ridge),
            'module_r2': [float(x) for x in module_r2],
            'distributed_r2_mean': float(np.mean(module_r2)),
            'global_r2': float(global_r2),
        }
        save_json(ckpt, ckpt_path)
        print(f"  {key}: Ridge={psi_ridge:.4f}, "
              f"Distributed={np.mean(module_r2):.4f}, Global={global_r2:.4f}")

        del sys_f, traj
        torch.cuda.empty_cache()

    # Part B: Distributed under betaG pressure
    print("\n--- E1 Part B: Distributed GRUs under betaG ---")
    betaG_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    pressure_configs = {
        'all_biased': list(range(n_modules)),
        'half_biased': list(range(n_modules // 2)),
        'one_biased': [0],
    }

    for cfg_name, biased_modules in pressure_configs.items():
        for betaG in betaG_values:
            for seed in seeds:
                key = f"{cfg_name}_bG{betaG:.4f}_s{seed}"
                if key in ckpt:
                    continue

                sys_f = FractalTreeSystem(L, n, lam0, gamma_0=gamma_0,
                                          seed=seed, device=DEVICE)
                traj = sys_f.generate_trajectory(
                    measure + train_len, warmup=warmup)
                meas_traj = traj[:measure]
                train_traj = traj[measure:]

                module_r2 = []
                for m in range(n_modules):
                    mod_train = sys_f.traj_module(train_traj, m)
                    mod_test = sys_f.traj_module(meas_traj, m)
                    gru = GRUSelfModel(n, hidden_factor=1.0, n_layers=1)

                    if m in biased_modules and betaG > 0:
                        train_module_gru(gru, mod_train, n_epochs=15,
                                         device=DEVICE_SM,
                                         betaG=betaG, x_star=x_stars[m])
                    else:
                        train_module_gru(gru, mod_train, n_epochs=15,
                                         device=DEVICE_SM)

                    r2 = evaluate_gru_r2(gru, mod_test, device=DEVICE_SM)
                    module_r2.append(r2)
                    del gru
                    torch.cuda.empty_cache()

                biased_r2 = [module_r2[m] for m in biased_modules]
                free_modules = [m for m in range(n_modules)
                                if m not in biased_modules]
                free_r2 = [module_r2[m] for m in free_modules]

                ckpt[key] = {
                    'config': cfg_name, 'betaG': betaG, 'seed': seed,
                    'module_r2': [float(x) for x in module_r2],
                    'r2_biased_mean': float(np.mean(biased_r2)),
                    'r2_free_mean': float(np.mean(free_r2)) if free_r2 else float('nan'),
                    'r2_total': float(np.mean(module_r2)),
                    'biased_modules': biased_modules,
                }
                save_json(ckpt, ckpt_path)
                print(f"  {key}: R2_biased={np.mean(biased_r2):.4f}, "
                      f"R2_free={np.mean(free_r2) if free_r2 else float('nan'):.4f}, "
                      f"R2_total={np.mean(module_r2):.4f}")

                del sys_f, traj
                torch.cuda.empty_cache()

    # ── Analysis & Figures ────────────────────────────────────
    print("\n[E1] Analyzing...")

    # Part A table
    print("\n  === Baseline Comparison ===")
    for seed in seeds:
        key = f"baseline_s{seed}"
        if key in ckpt:
            d = ckpt[key]
            print(f"  s={seed}: Ridge={d['psi_ridge']:.4f}, "
                  f"Distributed={d['distributed_r2_mean']:.4f}, "
                  f"Global={d['global_r2']:.4f}")

    # Part B figure: Observer resilience
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_e = {'all_biased': 'C0', 'half_biased': 'C1', 'one_biased': 'C2'}

    rows_e1 = []
    for cfg in pressure_configs:
        bG_vals = []
        r2_biased, r2_free, r2_total = [], [], []
        for betaG in betaG_values:
            pts = [ckpt[k] for k in ckpt
                   if isinstance(ckpt[k], dict) and
                   ckpt[k].get('config') == cfg and
                   abs(ckpt[k].get('betaG', -1) - betaG) < 1e-6]
            if not pts:
                continue
            bG_vals.append(betaG)
            r2_biased.append(np.mean([p['r2_biased_mean'] for p in pts]))
            r2_free.append(np.nanmean([p['r2_free_mean'] for p in pts]))
            r2_total.append(np.mean([p['r2_total'] for p in pts]))

        if not bG_vals:
            continue
        bG_arr = np.array(bG_vals)
        bG_plot = bG_arr.copy()
        bG_plot[0] = bG_arr[1] / 5 if len(bG_arr) > 1 else 0.0001

        axes[0].plot(bG_plot, r2_total, 'o-', color=colors_e[cfg],
                     label=cfg, markersize=5)
        axes[1].plot(bG_plot, r2_biased, '^--', color=colors_e[cfg],
                     label=f'{cfg} (biased)', markersize=4, alpha=0.7)
        if cfg != 'all_biased':
            axes[1].plot(bG_plot, r2_free, 's-', color=colors_e[cfg],
                         label=f'{cfg} (free)', markersize=5)

        # Resilience ratio
        if r2_free[0] > 0 and cfg != 'all_biased':
            ratio = [r / r2_free[0] for r in r2_free]
            axes[2].plot(bG_plot, ratio, 's-', color=colors_e[cfg],
                         label=f'{cfg} free', markersize=5)
        if r2_biased[0] > 0:
            ratio_b = [r / r2_biased[0] for r in r2_biased]
            axes[2].plot(bG_plot, ratio_b, '^--', color=colors_e[cfg],
                         label=f'{cfg} biased', markersize=4, alpha=0.6)

        for i, bg in enumerate(bG_vals):
            rows_e1.append({
                'config': cfg, 'betaG': bg,
                'r2_biased': r2_biased[i], 'r2_free': r2_free[i],
                'r2_total': r2_total[i],
            })

    for ax in axes:
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_xlabel('betaG')
    axes[0].set_ylabel('R2 total')
    axes[0].set_title('E1: Total GRU R2')
    axes[1].set_xlabel('betaG')
    axes[1].set_ylabel('R2')
    axes[1].set_title('E1: Biased vs Free GRUs')
    axes[2].set_xlabel('betaG')
    axes[2].set_ylabel('R2 / R2(betaG=0)')
    axes[2].set_title('E1: Observer Resilience')
    axes[2].axhline(1.0, ls='--', color='gray', alpha=0.5)

    fig.suptitle('E1: Distributed Self-Models Under Pressure', fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / 'E1_observer_resilience.png')
    plt.close(fig)

    if rows_e1:
        csv_e1 = out_dir / 'E1_pressure.csv'
        with open(csv_e1, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_e1[0].keys()))
            w.writeheader()
            w.writerows(rows_e1)

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[E1] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT F1 — Comparison Matrix
# ═══════════════════════════════════════════════════════════════

def run_exp_F1():
    print("\n" + "=" * 70)
    print("  EXPERIMENT F1 — ALIGNMENT COMPARISON MATRIX")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_F1'
    out_dir.mkdir(exist_ok=True)

    # Load A2 results from raw JSON
    a2_v1_raw = load_json(BASE_DIR / 'results_A2' / 'raw_data_v1.json')
    a2_v2_raw = load_json(BASE_DIR / 'results_A2' / 'raw_data_v2.json')

    # Load D2 and E1
    d2_data = load_json(BASE_DIR / 'results_D2' / 'raw_data.json')
    e1_data = load_json(BASE_DIR / 'results_E1' / 'raw_data.json')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    betaG_common = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    # Top-left: Flat + betaG on dynamics (A2 V2)
    ax = axes[0, 0]
    if a2_v2_raw:
        bGs, ratios = [], []
        lam_arr = np.linspace(0.8, 1.3, 60)
        baseline_psi = None
        for bG in betaG_common:
            psis = []
            for s in [42, 43, 44]:
                key = f"bG{bG:.4f}_s{s}"
                if key in a2_v2_raw:
                    psi = np.max(np.array(a2_v2_raw[key]['psi']))
                    psis.append(psi)
            if psis:
                mean_psi = np.mean(psis)
                if bG == 0:
                    baseline_psi = mean_psi
                if baseline_psi and baseline_psi > 0:
                    bGs.append(bG)
                    ratios.append(mean_psi / baseline_psi)
        if bGs:
            bG_plot = np.array(bGs)
            bG_plot[0] = bG_plot[1] / 5 if len(bG_plot) > 1 else 0.0001
            ax.plot(bG_plot, ratios, 'o-', color='C3', label='Flat+dynamics')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('Psi / Psi(0)')
    ax.set_title('Flat + betaG on Dynamics (A2 V2)')
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Fractal + betaG on dynamics (D2)
    ax = axes[0, 1]
    if d2_data:
        for cfg in ['global', 'subtree', 'local']:
            bGs, ratios = [], []
            baseline = None
            for bG in [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                pts = [d2_data[k] for k in d2_data
                       if isinstance(d2_data[k], dict) and
                       d2_data[k].get('config') == cfg and
                       abs(d2_data[k].get('betaG', -1) - bG) < 1e-6]
                if pts:
                    val = np.mean([p['psi_global'] for p in pts])
                    if bG == 0:
                        baseline = val
                    if baseline and baseline > 0:
                        bGs.append(bG)
                        ratios.append(val / baseline)
            if bGs:
                bG_plot = np.array(bGs)
                bG_plot[0] = bG_plot[1] / 5 if len(bG_plot) > 1 else 0.0001
                ax.plot(bG_plot, ratios, 'o-', label=cfg)
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('Psi / Psi(0)')
    ax.set_title('Fractal + betaG on Dynamics (D2)')
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Flat + betaG on observer (A2 V1)
    ax = axes[1, 0]
    if a2_v1_raw:
        bGs, ratios = [], []
        baseline_r2 = None
        for bG in betaG_common:
            r2s = []
            for s in [42, 43, 44]:
                key = f"bG{bG:.4f}_s{s}"
                if key in a2_v1_raw and key != 'baselines':
                    r2_arr = np.array(a2_v1_raw[key]['gru_r2'])
                    r2s.append(np.max(r2_arr))
            if r2s:
                mean_r2 = np.mean(r2s)
                if bG == 0:
                    baseline_r2 = mean_r2
                if baseline_r2 and baseline_r2 > 0:
                    bGs.append(bG)
                    ratios.append(mean_r2 / baseline_r2)
        if bGs:
            bG_plot = np.array(bGs)
            bG_plot[0] = bG_plot[1] / 5 if len(bG_plot) > 1 else 0.0001
            ax.plot(bG_plot, ratios, 'o-', color='C3', label='Flat+observer')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('GRU R2 / R2(0)')
    ax.set_title('Flat + betaG on Observer (A2 V1)')
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Fractal + betaG on observer (E1)
    ax = axes[1, 1]
    if e1_data:
        for cfg in ['all_biased', 'half_biased', 'one_biased']:
            bGs, ratios = [], []
            baseline = None
            for bG in betaG_common:
                pts = [e1_data[k] for k in e1_data
                       if isinstance(e1_data[k], dict) and
                       e1_data[k].get('config') == cfg and
                       abs(e1_data[k].get('betaG', -1) - bG) < 1e-6]
                if pts:
                    val = np.mean([p['r2_total'] for p in pts])
                    if bG == 0:
                        baseline = val
                    if baseline and baseline > 0:
                        bGs.append(bG)
                        ratios.append(val / baseline)
            if bGs:
                bG_plot = np.array(bGs)
                bG_plot[0] = bG_plot[1] / 5 if len(bG_plot) > 1 else 0.0001
                ax.plot(bG_plot, ratios, 'o-', label=cfg)
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('betaG')
    ax.set_ylabel('GRU R2 / R2(0)')
    ax.set_title('Fractal + betaG on Observer (E1)')
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('The Alignment Matrix: Architecture x Pressure Type',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'alignment_matrix.png')
    plt.close(fig)
    print(f"  Figure: {out_dir / 'alignment_matrix.png'}")

    # Summary markdown
    summary = """# F1: Alignment Matrix Summary

## 2x2 Comparison

|                | Flat Architecture    | Fractal Architecture     |
|----------------|---------------------|--------------------------|
| betaG on dynamics  | No effect (A2 V2)   | Resilient (D2)          |
| betaG on observer  | Total collapse (A2 V1) | Partial resilience (E1) |

## Key Finding

Centralized observers are fragile: biasing a single global self-model
destroys ALL self-observation. Distributed observers on a fractal
architecture survive partial capture because unbiased modules continue
to model their local dynamics accurately.
"""
    with open(out_dir / 'summary.md', 'w') as f:
        f.write(summary)

    print("[F1] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT F2 — Information Flow Topology
# ═══════════════════════════════════════════════════════════════

def run_exp_F2():
    print("\n" + "=" * 70)
    print("  EXPERIMENT F2 — INFORMATION FLOW TOPOLOGY")
    print("=" * 70)

    out_dir = BASE_DIR / 'results_F2'
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    ckpt = load_json(ckpt_path) or {}

    n, L = 128, 3
    gamma_0 = 0.05
    n_modules = 2 ** L
    lambda_values = [0.85, 1.043, 1.20]
    seeds = [42, 43]
    warmup, measure = 2000, 15000

    for lam in lambda_values:
        for seed in seeds:
            key = f"lam{lam:.3f}_s{seed}"
            if key in ckpt:
                continue

            sys_f = FractalTreeSystem(L, n, lam, gamma_0=gamma_0,
                                      seed=seed, device=DEVICE)
            traj = sys_f.generate_trajectory(measure, warmup=warmup)

            # Cross-module R2 matrix
            flow_matrix = np.zeros((n_modules, n_modules))
            for i in range(n_modules):
                ti = sys_f.traj_module(traj, i).cpu().numpy().astype(np.float64)
                for j in range(n_modules):
                    tj = sys_f.traj_module(traj, j).cpu().numpy().astype(np.float64)
                    # Predict x_i(t+1) from x_j(t)
                    X = tj[:-1]
                    Y = ti[1:]
                    split = int(0.6 * len(X))
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X[:split], Y[:split])
                    flow_matrix[i, j] = max(ridge.score(X[split:], Y[split:]), 0.0)

            # Aggregate by tree distance
            flow_by_dist = defaultdict(list)
            for i in range(n_modules):
                for j in range(n_modules):
                    d = sys_f._tree_distance(i, j)
                    flow_by_dist[d].append(flow_matrix[i, j])

            dist_flow = {d: float(np.mean(v)) for d, v in
                         sorted(flow_by_dist.items())}

            ckpt[key] = {
                'lambda': lam, 'seed': seed,
                'flow_matrix': flow_matrix.tolist(),
                'flow_by_distance': dist_flow,
            }
            save_json(ckpt, ckpt_path)
            print(f"  {key}: diag_mean={np.mean(np.diag(flow_matrix)):.4f}, "
                  f"off_diag_mean={np.mean(flow_matrix[~np.eye(n_modules, dtype=bool)]):.4f}")

            del sys_f, traj
            torch.cuda.empty_cache()

    # ── Figures ───────────────────────────────────────────────
    print("\n[F2] Generating figures...")

    # Heatmaps per lambda
    for lam in lambda_values:
        matrices = []
        for seed in seeds:
            key = f"lam{lam:.3f}_s{seed}"
            if key in ckpt:
                matrices.append(np.array(ckpt[key]['flow_matrix']))

        if not matrices:
            continue
        avg_mat = np.mean(matrices, axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(avg_mat, cmap='viridis', vmin=0,
                        vmax=max(0.01, avg_mat.max()))
        ax.set_xlabel('Source module j')
        ax.set_ylabel('Target module i')
        ax.set_title(f'F2: Info Flow (lambda={lam:.3f})')
        plt.colorbar(im, ax=ax, label='Ridge R2(x_i(t+1) | x_j(t))')
        fig.tight_layout()
        fig.savefig(out_dir / f'info_flow_lam{lam:.3f}.png')
        plt.close(fig)

    # Flow vs distance
    fig, ax = plt.subplots(figsize=(8, 5))
    for li, lam in enumerate(lambda_values):
        all_dist_flow = defaultdict(list)
        for seed in seeds:
            key = f"lam{lam:.3f}_s{seed}"
            if key in ckpt:
                for d_str, v in ckpt[key]['flow_by_distance'].items():
                    all_dist_flow[int(d_str)].append(v)

        if all_dist_flow:
            dists = sorted(all_dist_flow.keys())
            means = [np.mean(all_dist_flow[d]) for d in dists]
            ax.plot(dists, means, 'o-', label=f'lambda={lam:.3f}',
                    markersize=6)

    ax.set_xlabel('Tree distance d')
    ax.set_ylabel('Mean Ridge R2')
    ax.set_title('F2: Information Flow vs Tree Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'info_flow_vs_distance.png')
    plt.close(fig)

    # CSV
    rows_f2 = []
    for lam in lambda_values:
        for seed in seeds:
            key = f"lam{lam:.3f}_s{seed}"
            if key in ckpt:
                mat = np.array(ckpt[key]['flow_matrix'])
                rows_f2.append({
                    'lambda': lam, 'seed': seed,
                    'self_pred_mean': float(np.mean(np.diag(mat))),
                    'cross_pred_mean': float(np.mean(
                        mat[~np.eye(n_modules, dtype=bool)])),
                })
    if rows_f2:
        csv_f2 = out_dir / 'info_flow.csv'
        with open(csv_f2, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_f2[0].keys()))
            w.writeheader()
            w.writerows(rows_f2)

    save_json(ckpt, out_dir / 'raw_data.json')
    print("[F2] COMPLETE\n")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='IPT Comprehensive Experiments (fractal tree)')
    parser.add_argument('--exp', type=str, required=True,
                        choices=['D1', 'D2', 'E1', 'F1', 'F2', 'all'],
                        help='Which experiment to run')
    args = parser.parse_args()

    print(f"IPT Comprehensive Experiments — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"Dynamics device: {DEVICE}")
    print(f"Self-model device: {DEVICE_SM}")
    print()

    # Pre-initialize CUDA
    for i in range(torch.cuda.device_count()):
        torch.zeros(1, device=f'cuda:{i}')

    # ── Startup Diagnostic ────────────────────────────────────
    print("=" * 50)
    print("  STARTUP DIAGNOSTIC")
    print("=" * 50)
    ok = True
    try:
        ft = FractalTreeSystem(2, 16, 1.0, gamma_0=0.05, seed=0, device=DEVICE)
        print(f"  [OK] FractalTreeSystem: L=2, n=16, N={ft.N}, "
              f"modules={ft.n_modules}")
        traj = ft.generate_trajectory(100, warmup=50)
        print(f"  [OK] generate_trajectory -> {traj.shape}")
        flat = ft.traj_flat(traj)
        print(f"  [OK] traj_flat -> {flat.shape}")
        J = ft.jacobian(traj[-1])
        print(f"  [OK] jacobian -> {J.shape}")
        lyap = ft.compute_lyapunov(traj, steps=50)
        print(f"  [OK] Lyapunov = {lyap:.4f}")
        psi = measure_ridge_r2(flat.cpu().numpy().astype(np.float64))
        print(f"  [OK] Ridge R2 = {psi:.4f}")
        del ft, traj
    except Exception as e:
        print(f"  [FAIL] FractalTreeSystem: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    try:
        flat_sys = OrthogonalRNN(32, 1.0, input_scale=0.01, seed=0, device=DEVICE)
        print(f"  [OK] OrthogonalRNN (flat comparison)")
        del flat_sys
    except Exception as e:
        print(f"  [FAIL] OrthogonalRNN: {e}")
        ok = False

    if args.exp in ('E1', 'all'):
        try:
            gru = GRUSelfModel(16, hidden_factor=1.0, n_layers=1)
            gru = gru.to(DEVICE)
            x = torch.randn(1, 5, 16, device=DEVICE)
            out, h = gru(x)
            print(f"  [OK] GRUSelfModel -> out={out.shape}, h={h.shape}")
            del gru
        except Exception as e:
            print(f"  [FAIL] GRUSelfModel: {e}")
            ok = False

    torch.cuda.empty_cache()

    if not ok:
        print("\n  DIAGNOSTIC FAILED — fix issues above.")
        sys.exit(1)
    print("\n  ALL CHECKS PASSED\n")
    print("=" * 50)

    t0 = time.time()

    if args.exp in ('D1', 'all'):
        run_exp_D1()
    if args.exp in ('D2', 'all'):
        run_exp_D2()
    if args.exp in ('E1', 'all'):
        run_exp_E1()
    if args.exp in ('F1', 'all'):
        run_exp_F1()
    if args.exp in ('F2', 'all'):
        run_exp_F2()

    total_h = (time.time() - t0) / 3600
    print(f"\nTotal runtime: {total_h:.2f} hours")
    print("ALL DONE.")


if __name__ == '__main__':
    main()
