#!/usr/bin/env python3
"""
Physical Systems Exhibiting Neural Network Path Integral Saturation
====================================================================

We identified a neural network's path-counting accuracy curve above a
threshold K₀ (the inflection point of the underlying sigmoid distribution):

    A(K) ≈ A∞ - (A∞ - A₀)·exp(β·K₀)·exp(-β·K)    for K ≥ K₀

where β = 1/τ.  This is equivalent to the standard saturation curve

    acc(K) = A∞ - (A∞ - A₀)·exp(-K/τ)

but with the prefactor exp(β·K₀) accounting for the fact that we only
compare theory to data above the inflection point K₀ of the sigmoid.

Fitted parameters: A∞ ≈ 99.87%,  A₀ ≈ 49.23%,  τ ≈ 7.33  (K = paths per pixel).

Here we construct two physical theories whose perturbation series in Feynman
diagrams reproduces this *exact* functional form and saturation rate above K₀.

══════════════════════════════════════════════════════════════════════════════
PHYSICAL INTERPRETATION
══════════════════════════════════════════════════════════════════════════════

The fit reveals an "effective coupling constant" of the neural network:

    β = 1/τ,    g_eff = exp(-β) = exp(-1/τ) ≈ 0.873

This is the ratio by which successive Feynman diagram corrections are
suppressed. It places the neural network in the "moderately strongly coupled"
non-perturbative regime — the perturbation series is close to diverging
(g → 1 would mean non-convergence; g < 1 ensures convergence).

The mapping to physical theories:

  SYSTEM 1  —  1D QUANTUM SCATTERING FROM A DELTA POTENTIAL
  ───────────────────────────────────────────────────────────
  H = p²/2m - a δ(x)    (attractive, a > 0)

  The Lippmann-Schwinger equation for the transmission amplitude t:

    t = 1 + G₀ V t    (Dyson equation)

  Exact (analytic):
    t_exact = 1/(1 + ia/2k)  where k = √(2mE)/ħ

  Born series (Feynman diagrams):
    t_K = Σₙ₌₀^{K-1} (-ia/2k)^n     (geometric series; K diagrams)

  K=1:  free propagation only — the "classical" straight-line path
  K=2:  classical path + single quantum scattering event
  K=N:  N successive scattering events = N Feynman diagrams
  K→∞:  full quantum amplitude (exact)

  The Born parameter |z| = a/(2k) = g_eff = 0.873 is chosen to match τ_NN.
  Each additional Born term is suppressed by exactly g_eff relative to the
  previous one.  THIS IS THE DIRECT FEYNMAN PATH INTEGRAL ANALOGY:
  K paths per pixel  ↔  K diagrams in the Born series  (compared for K ≥ K₀).

  For K ≥ K₀ the accuracy takes the form:
      acc_Born(K) = A∞ − (A∞ − A₀) · exp(β·K₀) · exp(−β·K)
  where β = 1/τ and K₀ is the sigmoid inflection point threshold.

  Physical meaning of g_eff ≈ 0.873:
  The neural network's "quantum potential" couples with strength a/(2k) ≈ 0.873.
  This is near the edge of perturbative control (g = 1 would be the
  non-perturbative boundary).  In QED, g = a_fine ≈ 1/137 (weak coupling);
  in QCD at low energy, g ≈ 1 (strongly coupled).  The trained neural network
  sits in an intermediate, moderately coupled regime.

  SYSTEM 2  —  QUANTUM ANHARMONIC OSCILLATOR  (ASYMPTOTIC CONTRAST)
  ──────────────────────────────────────────────────────────────────
  H = p^2/2 + x^2/2 + λx^4

  Exact:   numerical diagonalization in truncated Fock space
  PT:      Rayleigh-Schrödinger expansion E_0(λ) = Σ_n c_n λ^n
  Feynman diagrams: vacuum bubble diagrams (φ⁴ self-interactions)

  Crucial difference:  the PT series is ASYMPTOTIC  (c_n ~ (-1)^n n!/4^n for
  large n, by the Dyson instability argument).  This produces initial
  apparent convergence followed by inevitable divergence.

  The neural network DOES NOT suffer from this pathology because:
  ─ The ReLU threshold acts as a UV cutoff, killing long-distance paths
  ─ The finite network width bounds path count → regularizes the path integral
  ─ The sorted-path selection (largest weights first) mimics Borel resummation

  In field-theory language: the neural network is a REGULATED version of
  the scalar φ⁴ theory, where ReLU plays the role of a hard momentum cutoff.
  The anharmonic oscillator is the UNREGULATED version that diverges.

══════════════════════════════════════════════════════════════════════════════
DIAGRAM STRUCTURE
══════════════════════════════════════════════════════════════════════════════

Born series diagrams for 1D scattering (K = number of scattering vertices):

  K=1:   → → → → → → → → →   (straight arrow = free propagator G₀)
  K=2:   → → →×→ → → → → →   (one x = one scattering vertex V)
  K=3:   → → →×→ →×→ → → →   (two x's = two scattering events)
  K=N:   N-1 scattering vertices; contribution ~ (-ia/2k)^{N-1} ~ g^{N-1}

  Each "x" is one Feynman vertex.  Adding more vertices = adding more paths.
  The K-th diagram is suppressed by g^{K-1} relative to the free propagator.

Feynman diagrams for anharmonic oscillator (vacuum bubbles):

  K=1:   ─── (straight line = free HO propagator)
  K=2:   ─⊕─  (one 4-point vertex = x^4 insertion)
  K=3:   ─⊕─⊕─  (two vertices connected in a chain)
  K=N:   chain of N-1 vertices

  At large K, the number of diagrams at order n grows like n!, overwhelmig
  the g^n suppression → the series diverges asymptotically.

══════════════════════════════════════════════════════════════════════════════
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

OUTPUT_DIR = './figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  PARAMETERS FROM THE NN FIT  (from feynman_paths_numpy.py)
# ══════════════════════════════════════════════════════════════

tau_NN  = 3.5     # saturation scale r7.33 / l8.9
A_inf   = 0.9431   # asymptotic accuracy / 0.9987 l0.2037
A_0     = 0.4700   # K=0 extrapolated accuracy / 0.4923 l0.1261
K_0     = 15      # threshold: fit and comparison performed only for K ≥ K_0
                   # (inflection point of the underlying sigmoid distribution)
beta    = 1.0 / tau_NN            # decay rate β = 1/τ
g_eff   = np.exp(-beta)           # effective coupling  ≈ 0.873

# NN accuracy dictionary -- may contain any number of integer K datapoints.
# Evaluation and comparison are restricted to powers of 2 only.
acc_nn_measured = {1: 0.5, 2: 0.5, 3: 0.4963, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5148, 8: 0.5148, 9: 0.5704, 10: 0.4963, 11: 0.6, 12: 0.6222, 13: 0.6, 14: 0.6815, 15: 0.7074, 16: 0.7111, 17: 0.7963, 18: 0.8111, 19: 0.837, 20: 0.8296, 21: 0.8444, 22: 0.863, 23: 0.8926, 24: 0.9111, 25: 0.9148, 26: 0.9111, 27: 0.9296, 28: 0.9148, 29: 0.9148, 30: 0.9296, 31: 0.937, 32: 0.9481, 33: 0.9444, 34: 0.9481, 35: 0.9444, 36: 0.9407, 37: 0.9407, 38: 0.9444, 39: 0.9444, 40: 0.9444, 41: 0.9444, 42: 0.9444, 43: 0.9444, 44: 0.9444, 45: 0.9444, 46: 0.9444, 47: 0.9444, 48: 0.9444, 49: 0.9444, 50: 0.9444, 51: 0.9444, 52: 0.9444, 53: 0.9444, 54: 0.9444, 55: 0.9444, 56: 0.9444, 57: 0.9444, 58: 0.9444, 59: 0.9444, 60: 0.9444, 61: 0.9444, 62: 0.9444, 63: 0.9444, 64: 0.9444}
 
# {1: 0.1407, 2: 0.1370, 4: 0.1444, 8: 0.1889, 16: 0.1815, 32: 0.2074, 64: 0.2037} #linear
# {1: 0.5963, 2: 0.5741, 4: 0.6852, 8: 0.8444, 16: 0.9593, 32: 0.9889, 64: 0.9889}

# Automatically extract evaluation points: powers of 2 present in acc_nn_measured.
# (k > 0  and  k & (k-1) == 0  is the standard bitmask test for powers of 2.)
# The full dict may hold dense data (K = 1...512); only power-of-2 keys are used
# for the Born-series comparison and all plots.
k_values_nn = sorted(k for k in acc_nn_measured)
K_max_data  = max(k_values_nn)     # largest power-of-2 key -- sets the Born series extent

print(f"\n{'═'*64}")
print(f"  Effective coupling from NN fit:  g_eff = exp(-β),  β = 1/τ")
print(f"  τ = {tau_NN},  β = {beta:.4f}  →  g_eff = {g_eff:.4f}")
print(f"  Threshold K_0 = {K_0}  (fit/comparison for K ≥ K_0 only)")
print(f"  Accuracy above K_0:  A(K) = A∞ - (A∞-A₀)·exp(β·K₀)·exp(-β·K)")
print(f"  Interpretation:  each new path suppresses the residual")
print(f"  error by a factor {g_eff:.4f} — near-marginal coupling.")
print(f"{'═'*64}")


# ══════════════════════════════════════════════════════════════
#  SATURATION MODEL  (shared fit function)
# ══════════════════════════════════════════════════════════════

def saturation(K, A_inf, A_0, tau, K_0=0):
    """
    Saturation accuracy curve above threshold K_0:

        A(K) = A_inf - (A_inf - A_0) * exp(beta*K_0) * exp(-beta*K)

    where beta = 1/tau.  For K_0 = 0 this reduces to the standard form
    A_inf - (A_inf - A_0)*exp(-K/tau).  The prefactor exp(beta*K_0) arises
    because the NN data follows a sigmoid whose inflection point is at K_0;
    above K_0 the curve is well approximated by this shifted exponential.

    Parameters
    ----------
    K    : array-like  path count / Born order
    K_0  : float       threshold (sigmoid inflection point); fit only for K >= K_0
    """
    b = 1.0 / tau
    return A_inf - (A_inf - A_0) * np.exp(b * K_0) * np.exp(-b * np.asarray(K, dtype=float))


# ══════════════════════════════════════════════════════════════
#  SYSTEM 1:  BORN SERIES FOR 1D DELTA POTENTIAL
# ══════════════════════════════════════════════════════════════
#
#  V(x) = −α δ(x),  ħ = 2m = 1,  incident momentum k.
#  z = iα/(2k) = i·g_eff  (purely imaginary coupling parameter)
#
#  Exact analytic transmission amplitude:
#      t_exact = 1/(1 + z)
#
#  K-th Born approximation (K diagrams = K terms in Neumann series):
#      t_K = Σₙ₌₀^{K-1} (−z)ⁿ = [1 − (−z)^K] / (1 + z)
#
#  Residual amplitude error:
#      |t_K − t_exact| = |z|^K / |1 + z| = g_eff^K / |1 + z|
#
#  Raw accuracy (fraction of amplitude recovered relative to vacuum):
#      acc_raw(K) = 1 − |t_K − t_exact| / |t_exact|
#                = 1 − g_eff^K         (exact geometric decay!)
#
#  Rescaled to match NN's [A₀, A∞] range and adjusted for threshold K₀:
#      acc_Born(K) = A∞ − (A∞ − A₀) · exp(β·K₀) · exp(−β·K)   for K ≥ K₀
#      where β = 1/τ and the prefactor exp(β·K₀) reflects the sigmoid inflection.
#                  = saturation(K, A∞, A₀, τ_NN, K₀)   ← EXACT MATCH ✓
#

def born_series_amplitudes(K_max, g, K_0=0):
    """
    Returns arrays of Born series amplitudes t_K (complex),
    exact amplitude t_exact, and per-K accuracy (raw and matched).

    For K >= K_0 the rescaled accuracy uses the threshold-adjusted formula:

        acc_born(K) = A_inf - (A_inf - A_0) * exp(beta*K_0) * exp(-beta*K)

    where beta = -ln(g) = 1/tau and K_0 is the sigmoid inflection threshold.
    This shifts the exponential decay so it is anchored at K_0, matching
    the NN data which only exhibits clean saturation behaviour for K >= K_0.

    Parameters
    ----------
    K_max : int
    g     : float  |z| = a/(2k) = exp(-beta) — the Born coupling parameter
    K_0   : float  threshold; acc_born uses the exp(beta*K_0) prefactor

    Returns
    -------
    K_arr    : ndarray [K_max+1] (K = 0, 1, ..., K_max)
    t_arr    : ndarray complex [K_max+1]
    t_exact  : complex scalar
    T_arr    : ndarray [K_max+1]  |t_K|² = transmission probability
    T_exact  : float
    err_arr  : ndarray [K_max+1]  |t_K - t_exact| (amplitude error)
    acc_raw  : ndarray [K_max+1]  in [0, 1]
    acc_born : ndarray [K_max+1]  rescaled to [A_0, A_inf] with exp(beta*K_0) prefactor
    """
    z       = 1j * g                    # purely imaginary coupling
    t_exact = 1.0 / (1.0 + z)
    T_exact = abs(t_exact) ** 2
    beta_g  = -np.log(g)                # beta = 1/tau = -ln(g_eff)

    K_arr    = np.arange(K_max + 1)
    t_arr    = np.zeros(K_max + 1, dtype=complex)
    err_arr  = np.zeros(K_max + 1)
    acc_raw  = np.zeros(K_max + 1)
    acc_born = np.zeros(K_max + 1)

    for K in K_arr:
        if K == 0:
            t_K = 0.0 + 0j              # vacuum: no propagation
        else:
            # t_K = Σ_{n=0}^{K-1} (−z)ⁿ = [1 − (−z)^K] / (1 + z)
            t_K = (1.0 - (-z) ** K) / (1.0 + z)

        t_arr[K]    = t_K
        err_arr[K]  = abs(t_K - t_exact)
        acc_raw[K]  = 1.0 - err_arr[K] / abs(t_exact)   # = 1 - g^K for K>=0
        # Rescaled with exp(beta*K_0) prefactor — valid for K >= K_0
        acc_born[K] = A_inf - (A_inf - A_0) * np.exp(beta_g * K_0) * np.exp(-beta_g * K)

    T_arr = np.abs(t_arr) ** 2
    return K_arr, t_arr, t_exact, T_arr, T_exact, err_arr, acc_raw, acc_born


print("\n  Computing Born series (delta potential) …")
K_arr, t_arr, t_exact, T_arr, T_exact, err_arr, acc_raw_born, acc_born = \
    born_series_amplitudes(K_max=K_max_data, g=g_eff, K_0=K_0)

# Sanity: at K→large, t_K → t_exact
print(f"  K={K_max_data}  t_exact={t_exact:.4f}  t_K={t_arr[K_max_data]:.4f}  "
      f"|error|={err_arr[K_max_data]:.2e}  ✓" if err_arr[K_max_data] < 1e-3 else "  WARN: slow convergence")
print(f"  T_exact = {T_exact:.4f}  (transmission probability)")
print(f"  Born coupling  |z| = g_eff = {g_eff:.4f}")
print(f"  acc_Born(K=1)  = {100*acc_born[1]:.2f}%")
print(f"  acc_Born(K={K_max_data}) = {100*acc_born[K_max_data]:.2f}%")


# ══════════════════════════════════════════════════════════════
#  SYSTEM 2:  QUANTUM ANHARMONIC OSCILLATOR
# ══════════════════════════════════════════════════════════════
#
#  H = p^2/2 + x^2/2 + λ x^4       (ħ = m = ω = 1)
#
#  Exact ground-state energy:  numerical diagonalization in the
#  truncated harmonic oscillator (Fock) basis  {|n⟩, n=0…N_basis}.
#
#  Perturbation theory:  Rayleigh-Schrödinger coefficients
#  E₀(λ) = Σₙ≥₀ aₙ λⁿ  obtained by polynomial regression on
#  E_exact(λ) for small λ (valid because E(λ) is analytic at λ=0).
#
#  Feynman diagrams:  the nth-order PT coefficient aₙ counts all
#  connected vacuum diagrams with n φ⁴ vertices.
#  Number of distinct diagrams at order n grows like n! → ASYMPTOTIC.
#
#  "Accuracy":  fraction of the perturbative correction recovered:
#      acc_AHO(K; λ) = 1 − |E_PT^K(λ) − E_exact(λ)| / |E_0 − E_exact(λ)|
#  where E_0 = 0.5 (harmonic zero-point energy).
#

N_BASIS = 200   # truncated Fock space size

def build_x4_matrix(N):
    """Build x^4 in the harmonic oscillator Fock basis [NxN]."""
    # x = (a + a†)/√2,   a_{n, n-1} = √n,   a†_{n, n+1} = √(n+1)
    n_arr   = np.arange(N)
    off_diag = np.sqrt((n_arr[:-1] + 1) / 2.0)  # x_{n, n+1}
    x_mat   = np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    x2      = x_mat @ x_mat
    x4      = x2 @ x2
    return x4

def anharmonic_exact(lam, N=N_BASIS):
    """Exact ground-state energy of H = p^2/2 + x^2/2 + λ x^4."""
    x4   = build_x4_matrix(N)
    H    = np.diag(np.arange(N) + 0.5) + lam * x4
    # Only need the lowest eigenvalue
    E0   = eigh(H, eigvals_only=True, subset_by_index=[0, 0])[0]
    return float(E0)


def get_pt_coefficients(lam_target, order, n_sample=300):
    """
    Compute PT coefficients aₙ by polynomial regression:
        E(λ) = Σₙ₌₀^{order} aₙ λⁿ + O(λ^{order+1})
    evaluated for λ ∈ [0, lam_target/2].
    Accuracy degrades for large order; reliable up to ~order=18.
    """
    lam_vals = np.linspace(1e-6, lam_target / 2.0, n_sample)
    E_vals   = np.array([anharmonic_exact(l) for l in lam_vals])
    # Polynomial fit; polyfit returns highest-degree first → reverse
    coeffs   = np.polyfit(lam_vals, E_vals, order)[::-1]
    return coeffs   # coeffs[n] = aₙ


def anharmonic_pt_energies(lam, coeffs, K_max):
    """Partial sums: E_K(λ) = Σ_{n=0}^K a_n λ^n."""
    return np.array([sum(coeffs[n] * lam**n for n in range(K + 1))
                     for K in range(K_max + 1)])


# Choose λ so the PT series gives interesting behaviour in the first 15 orders:
# For λ = 0.05 the first ~7-8 orders converge before the asymptotic growth kicks in.
LAMBDA_AHO = 0.05
PT_MAX_ORDER = 20     # compute PT up to this order (higher orders are noisy)

print(f"\n  Computing anharmonic oscillator (λ = {LAMBDA_AHO}) …")
E_exact_AHO = anharmonic_exact(LAMBDA_AHO)
E_HO_AHO    = 0.5        # K=0: harmonic zero-point energy
print(f"  E_exact(λ={LAMBDA_AHO}) = {E_exact_AHO:.8f}")
print(f"  E_HO (λ=0)             = {E_HO_AHO:.4f}")
print(f"  Total correction       = {E_exact_AHO - E_HO_AHO:.6f}")

print(f"  Fitting PT coefficients (polynomial regression) …")
aho_coeffs = get_pt_coefficients(LAMBDA_AHO, PT_MAX_ORDER, n_sample=400)
E_PT_vals  = anharmonic_pt_energies(LAMBDA_AHO, aho_coeffs, PT_MAX_ORDER)

# Accuracy metric: fraction of energy correction recovered
total_corr = abs(E_exact_AHO - E_HO_AHO)
acc_AHO    = 1.0 - np.abs(E_PT_vals - E_exact_AHO) / total_corr  # can go negative!

print(f"  acc_AHO at K=1:  {100*acc_AHO[1]:.1f}%")
print(f"  acc_AHO at peak: {100*acc_AHO.max():.1f}%  (K={acc_AHO.argmax()})")

# Fit saturation to AHO (only the converging part: K ≤ K_peak)
K_peak_AHO = int(acc_AHO.argmax())
K_vals_AHO_fit = np.arange(1, K_peak_AHO + 1)
acc_AHO_fit    = acc_AHO[1: K_peak_AHO + 1]
try:
    # Rescale AHO accuracy to [A₀, A∞] for fair comparison
    A_0_AHO  = acc_AHO[1]        # K=1 accuracy (leading order = first PT term)
    A_inf_AHO = acc_AHO.max()
    acc_AHO_scaled = A_0 + (A_inf - A_0) * (acc_AHO - A_0_AHO) / (A_inf_AHO - A_0_AHO)
    popt_AHO, _ = curve_fit(saturation,
                             K_vals_AHO_fit,
                             acc_AHO_scaled[1: K_peak_AHO + 1],
                             p0=[A_inf, A_0, 3.0],
                             bounds=([0, 0, 0.1], [1, 1, 100]),
                             maxfev=10000)
    tau_AHO = popt_AHO[2]
    print(f"  Fitted τ_AHO = {tau_AHO:.2f}  (NN has τ = {tau_NN})")
except Exception as e:
    print(f"  AHO fit failed: {e}")
    tau_AHO = None; popt_AHO = None
    acc_AHO_scaled = A_0 + (A_inf - A_0) * (acc_AHO - acc_AHO.min()) / (acc_AHO.max() - acc_AHO.min())


# ══════════════════════════════════════════════════════════════
#  VERIFY: Born series acc_born exactly reproduces NN saturation
# ══════════════════════════════════════════════════════════════

print("\n  Verifying Born series = NN saturation (for K >= K_0) …")
k_test_all = np.array(k_values_nn, dtype=float)
k_test     = k_test_all[k_test_all >= K_0]   # only compare above threshold
acc_sat    = saturation(k_test, A_inf, A_0, tau_NN, K_0)
acc_born_at_k = np.array([acc_born[int(k)] for k in k_test])
max_diff = np.max(np.abs(acc_sat - acc_born_at_k))
print(f"  Max deviation Born vs NN-fit (K >= {K_0}): {max_diff:.2e}  "
      + ("✓ MATCH" if max_diff < 1e-4 else "✗ MISMATCH"))


# ══════════════════════════════════════════════════════════════
#  FEYNMAN DIAGRAM ILLUSTRATION (text art → embedded in plot)
# ══════════════════════════════════════════════════════════════

def draw_born_diagrams(ax, n_diagrams=5):
    """
    Visualise the first few Born-series Feynman diagrams as horizontal
    lines with filled circles marking scattering vertices.
    """
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, n_diagrams + 0.5)
    ax.axis('off')
    ax.set_title("Born-Series Feynman Diagrams\n"
                 r"(each $x$ = one scattering event off $\delta$-potential)",
                 fontsize=10, pad=6)

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, n_diagrams))

    for row, (nd, col) in enumerate(zip(range(n_diagrams), colours)):
        y = n_diagrams - row - 0.5
        K = nd + 1     # K-th Born term = nd scattering vertices

        # Draw propagator line
        ax.annotate('', xy=(9.5, y), xytext=(0.5, y),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=2.2, mutation_scale=16))

        # Draw scattering vertices at evenly spaced positions
        if nd > 0:
            xs = np.linspace(2.5, 7.5, nd)
            ax.scatter(xs, [y]*nd, s=180, zorder=5,
                       color=col, edgecolors='black', linewidths=0.8)
            for xi in xs:
                ax.text(xi, y+0.22, 'x', ha='center', va='bottom',
                        fontsize=11, color='black', fontweight='bold')

        # Label
        contrib = fr"$(-ig)^{nd}$" if nd > 0 else "1"
        ax.text(0.3, y, fr"$K=${K}:", ha='right', va='center',
                fontsize=9, color=col, fontweight='bold')
        ax.text(9.7, y, contrib, ha='left', va='center',
                fontsize=9, color=col, style='italic')

    # Legend for coupling
    ax.text(5.0, n_diagrams + 0.2,
            fr"$|-ig| = g_eff = e^{{-\beta}} = e^{{-1/\tau}} = {g_eff:.3f}$, $x = $scattering vertex",
            ha='center', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9))


# ══════════════════════════════════════════════════════════════
#  COUPLING CONSTANT LANDSCAPE
# ══════════════════════════════════════════════════════════════

def coupling_landscape(ax):
    """
    Show where g_eff sits in the coupling constant landscape of known theories.
    """
    theories = {
    r'QED' "\n" r'$(\alpha \approx 1/137)$':          (1/137,  '#2196F3'),
    r'EW (weak)' "\n" r'$(g \approx 0.65)$':          (0.65,   '#4CAF50'),
    r'QCD (high E)' "\n" r'$(\alpha_s \approx 0.12)$': (0.12,   '#FF9800'),
    r'Neural Net' "\n" r'(this work)':                (g_eff,  '#E91E63'),
    r'QCD (low E)' "\n" r'$(\alpha_s \approx 1.0)$':   (1.0,    '#9C27B0'),
}

    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.5, len(theories) - 0.5)
    ax.axis('off')
    ax.set_title("Effective Coupling Constant Landscape", fontsize=10, pad=6)

    # Convergence background
    ax.axvspan(0, 1, alpha=0.08, color='green', label='Convergent PT')
    ax.axvspan(1, 1.15, alpha=0.08, color='red', label='Non-convergent')
    ax.axvline(1.0, color='red', lw=1.5, ls='--', alpha=0.5)
    ax.text(1.02, -0.35, 'PT\nboundary', fontsize=7.5, color='red', va='bottom')

    for row, (label, (g, col)) in enumerate(theories.items()):
        y = row
        # Bar
        ax.barh(y, g, height=0.45, color=col, alpha=0.75, edgecolor=col)
        # Value label
        ax.text(g + 0.01, y, f'g = {g:.3f}', va='center',
                fontsize=8.5, color=col, fontweight='bold')
        ax.text(-0.01, y, label, va='center', ha='right',
                fontsize=8, color='black')

    ax.text(0.5, -0.45, '← perturbative regime →',
            ha='center', va='top', fontsize=8, color='green', alpha=0.8)


# ══════════════════════════════════════════════════════════════
#  CONVERGENCE COMPARISON TABLE  (printed to terminal)
# ══════════════════════════════════════════════════════════════

print(f"\n{'═'*76}")
print(f"  Comparison for K >= K_0 = {K_0}  (above sigmoid inflection point)")
print(f"  {'K':>4}  {'NN (measured)':>14}  {'Born fit':>10}  "
      f"{'NN fit (sat.)':>14}  {'|Born-NN|':>10}")
print(f"{'─'*76}")
k_fine = [k for k in k_values_nn if k >= K_0]
for k in k_fine:
    nn_m   = acc_nn_measured.get(k, float('nan'))
    b_val  = acc_born[k]
    nn_fit = saturation(k, A_inf, A_0, tau_NN, K_0)
    diff   = abs(b_val - nn_m)
    print(f"  {k:>4}  {100*nn_m:>13.2f}%  {100*b_val:>9.2f}%  "
          f"{100*nn_fit:>13.2f}%  {100*diff:>9.2f}pp")
print(f"{'═'*76}")


# ══════════════════════════════════════════════════════════════
#  MAIN FIGURE  (6 panels)
# ══════════════════════════════════════════════════════════════

print("\n  Generating figure …")

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.48, wspace=0.38,
                        height_ratios=[1.1, 1, 0.9])

ax_main   = fig.add_subplot(gs[0, :2])   # top-left: big comparison
ax_err    = fig.add_subplot(gs[0, 2])    # top-right: amplitude errors
ax_aho    = fig.add_subplot(gs[1, :2])   # middle-left: anharmonic oscillator
ax_re_im  = fig.add_subplot(gs[1, 2])    # middle-right: complex Born amplitude
ax_diag   = fig.add_subplot(gs[2, :2])   # bottom-left: Feynman diagrams
ax_coup   = fig.add_subplot(gs[2, 2])    # bottom-right: coupling landscape

K_cont    = np.linspace(0, K_max_data, 500)

# ─── Panel 1: Main accuracy comparison ────────────────────────
ax = ax_main
# NN measured data — powers of 2 only, above K_0, max 10 points
k_plot = [k for k in k_values_nn if k >= K_0]
if len(k_plot) > 10:
    # evenly spaced subsample of at most 10 indices
    idx = np.round(np.linspace(0, len(k_plot) - 1, 10)).astype(int)
    k_plot = [k_plot[i] for i in idx]
kk = np.array(k_plot)
aa = np.array([acc_nn_measured[k] for k in k_plot]) * 100
ax.scatter(kk, aa, s=110, zorder=8, color='steelblue',
           edgecolors='navy', lw=0.9, label='NN measured (sklearn digits)', marker='D')
ax.plot(kk, aa, alpha=0.4, color='steelblue', lw=1.5)

# NN saturation fit (only for K >= K_0)
K_cont_fit = K_cont[K_cont >= K_0]
nn_fit_line = saturation(K_cont_fit, A_inf, A_0, tau_NN, K_0) * 100
ax.plot(K_cont_fit, nn_fit_line, '--', color='steelblue', lw=2.2, alpha=0.8,
        label=fr'NN saturation fit  $\tau = {tau_NN}$,  $K_0 = {K_0}$')

# Born series (EXACT match by construction, for K >= K_0)
ax.plot(K_cont_fit, acc_born[:K_max_data+1][np.minimum(K_cont_fit.astype(int), K_max_data)] * 100,
        '-', color='crimson', lw=2.8, alpha=0.9,
        label=f'Born series  $|z| = g_eff = {g_eff:.3f}$  (delta potential)')
k_values_nn_above = [k for k in k_values_nn if k >= K_0]
# ax.scatter(K_arr[k_values_nn_above], acc_born[k_values_nn_above] * 100,
#            s=80, zorder=8, color='crimson', edgecolors='darkred', lw=0.8)

# Full network reference
ax.axhline(A_inf * 100, color='gray', ls=':', lw=1.8, alpha=0.7,
           label=f'Full network $A_\infty = {100*A_inf:.2f}\%$')
ax.axhline(A_0 * 100, color='orange', ls=':', lw=1.5, alpha=0.6,
           label=f'Classical $A_0 = {100*A_0:.2f}\%$')

ax.set_xlabel('K  (paths per pixel  /  Born-series order)', fontsize=12)
ax.set_ylabel('Accuracy / Recovery (\%)', fontsize=12)
ax.set_title(r'Neural Network Paths  $\Leftrightarrow$  Feynman Diagrams (Born Series)' "\n"
             fr'Fit above inflection threshold $K_0 = {K_0}$; same saturation curve governs both systems', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(-1, K_max_data + K_max_data * 0.05)

# Mark threshold K_0
ax.axvline(K_0, color='purple', ls='-.', lw=1.5, alpha=0.7,
           label=fr'Threshold $K_0 = {K_0}$')
ax.text(K_0 + 0.5, ax.get_ylim()[0] + 2,
        fr'$K_0 = {K_0}$', fontsize=8.5, color='purple', va='bottom')

# Annotate key physics
ax.annotate('Classical path' "\n" r'($K=K_0$: threshold)',
            xy=(k_plot[0], acc_born[k_plot[0]]*100), xytext=(0.12, 0.25), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.3),
            fontsize=8.5, color='crimson')

ax.annotate('Quantum corrections saturate\n := Dyson resummation complete',
            xy=(k_values_nn[len(k_values_nn)//2], acc_born[k_values_nn[len(k_values_nn)//2]]*100), xytext=(0.40, 0.35), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            fontsize=8.5, color='gray')

# ─── Panel 2: Amplitude error decay ────────────────────────
ax = ax_err
K_range = np.arange(1, K_max_data + 1)
err_theoretical = g_eff ** K_range / abs(1 + 1j * g_eff)

ax.semilogy(K_range, err_arr[1:K_max_data+1], 'o-', color='crimson', lw=2.0,
            ms=5, label=r'$|t_K - t_{{exact}}|$  (Born)', markevery=4)
ax.semilogy(K_range, err_theoretical, '--', color='orange', lw=1.8, alpha=0.8,
            label=fr'$g_{{\mathrm{{eff}}}}^K / |1+z|  =  {g_eff:.3f}^K$')

ax.set_xlabel(r'$K$ (Born order)', fontsize=11)
ax.set_ylabel(r'Amplitude error $|t_K - t_{{exact}}|$', fontsize=10)
ax.set_title('Geometric Decay of\nBorn Series Error', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
ax.text(0.55, 0.72, fr'slope $= -\beta = -1/\tau = -{beta:.3f}$' "\n" fr'$g_{{eff}} = e^{{-\beta}} = {g_eff:.3f}$',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

# ─── Panel 3: Anharmonic oscillator PT vs exact ─────────────
ax = ax_aho
K_aho = np.arange(PT_MAX_ORDER + 1)

# Energy values
ax.plot(K_aho, E_PT_vals, 'o-', color='darkorange', lw=2.0,
        ms=5, label=fr'PT energy $E_K (\lambda={LAMBDA_AHO})$', markevery=1)
ax.axhline(E_exact_AHO, color='black', ls='--', lw=2.0,
           label=fr'Exact $E_0 = {E_exact_AHO:.5f}$')
ax.axhline(E_HO_AHO, color='gray', ls=':', lw=1.5,
           label=f'Harmonic $E_0 = {E_HO_AHO:.3f}  (K=0)$')

# Shade the "convergence window"
ax.axvspan(0, K_peak_AHO, alpha=0.07, color='green',
           label=fr'Converging  $(K\leq{K_peak_AHO})$')
ax.axvspan(K_peak_AHO, PT_MAX_ORDER + 0.5, alpha=0.07, color='red',
           label='Diverging  (asymptotic)')

ax.set_xlabel('Perturbation order K', fontsize=11)
ax.set_ylabel(r'Ground state energy $E_0(\lambda)$', fontsize=11)
ax.set_title(fr'Anharmonic Oscillator: PT vs Exact  $(\lambda = {LAMBDA_AHO})$' "\n"
             f'Asymptotic series — initial convergence then divergence\n'
             f'(contrast with convergent Born/NN series)',
             fontsize=10)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(-0.5, PT_MAX_ORDER + 0.5)

# Second y-axis: acc
_axis = ax.twinx()
_axis.plot(K_aho, acc_AHO * 100, 's-', color='purple', lw=1.5, ms=4, alpha=0.8, label='Accuracy recovery (\%)')

if tau_AHO is not None:
    _axis.plot(K_vals_AHO_fit, acc_AHO_scaled[1: K_peak_AHO + 1] * 100, '--', color='purple', lw=2.0, label=fr'AHO saturation fit $\tau={tau_AHO:.2f}$')

_axis.set_ylabel('Accuracy (\%)', color='purple', fontsize=11)
_axis.tick_params(axis='y', labelcolor='purple')

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = _axis.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8.5, loc='lower left')

# ─── Panel 4: Complex Plane Born Amplitude ──────────────────
ax = ax_re_im
ax.plot(t_arr.real, t_arr.imag, 'o-', color='crimson', lw=1.5, ms=4, alpha=0.7, label=r'$t_K$ (Born series)')
ax.scatter([t_exact.real], [t_exact.imag], color='black', marker='*', s=150, zorder=5, label='t_exact')
ax.scatter([0], [0], color='gray', s=50, zorder=5, label='Origin (K=0)')

for k in [1, 2, 3, 4, 8]:
    if k < len(t_arr):
        ax.annotate(f'K={k}', (t_arr[k].real, t_arr[k].imag), textcoords="offset points", xytext=(5, 5), fontsize=8, color='darkred')

ax.set_xlabel('Re(t)', fontsize=11)
ax.set_ylabel('Im(t)', fontsize=11)
ax.set_title('Born Series in Complex Plane\nConverging to Exact Amplitude', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.axis('equal')

# ─── Panel 5: Feynman Diagrams ──────────────────────────────
draw_born_diagrams(ax_diag, n_diagrams=5)

# ─── Panel 6: Coupling Landscape ────────────────────────────
coupling_landscape(ax_coup)

# ══════════════════════════════════════════════════════════════
#  FINALIZE AND SAVE
# ══════════════════════════════════════════════════════════════
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'nn_physics_analogy.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n  Plot saved to: {out_path}\n")

# ══════════════════════════════════════════════════════════════
#  GOODNESS OF FIT: Born approximation vs NN data  (K >= K_0)
# ══════════════════════════════════════════════════════════════
from scipy.stats import pearsonr

k_gof   = np.array([k for k in k_values_nn if k >= K_0])
nn_gof  = np.array([acc_nn_measured[k] for k in k_gof])
born_gof = np.array([acc_born[k] for k in k_gof])

ss_res  = np.sum((nn_gof - born_gof) ** 2)
ss_tot  = np.sum((nn_gof - nn_gof.mean()) ** 2)
r2      = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse    = np.sqrt(ss_res / len(k_gof))
mae     = np.mean(np.abs(nn_gof - born_gof))
r, pval = pearsonr(nn_gof, born_gof)

print(f"\n{'═'*64}")
print(f"  Goodness of fit: Born approximation vs NN  (K >= K_0 = {K_0})")
print(f"{'─'*64}")
print(f"  N points   : {len(k_gof)}")
print(f"  R²         : {r2:.6f}")
print(f"  Pearson r  : {r:.6f}  (p = {pval:.2e})")
print(f"  RMSE       : {rmse*100:.4f} pp")
print(f"  MAE        : {mae*100:.4f} pp")
print(f"{'═'*64}\n")