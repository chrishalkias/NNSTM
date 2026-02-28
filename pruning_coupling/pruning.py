#!/usr/bin/env python3
"""
Feynman Path Integral Analogy in Neural Networks
=================================================
Pure NumPy implementation — no PyTorch/TensorFlow required.
Uses sklearn's load_digits (8x8 handwritten digits, 10 classes, 1797 samples).

HYPOTHESIS:
In a trained fully-connected ReLU network one can enumerate "paths" from each
input pixel to the output by carrying its signal layer-by-layer and keeping only
the K strongest connections (beam width = K).  As K increases from 1 (dominant
diagram) to H (all neurons = full network), prediction accuracy should increase
monotonically — analogous to perturbation theory where adding more Feynman
diagrams improves the approximation until the full (non-perturbative) result is
recovered.

PATH INTEGRAL ANALOGY:
──────────────────────────────────────────────────────────────────────────────
The amplitude for path  π = (i → j₁ → j₂ → … → jL → k)  is:
    A(π) = xᵢ · W₀[j₁,i] · m₁[j₁] · W₁[j₂,j₁] · m₂[j₂] · … · WL[k,jL]

where mˡ[j] = 1 if neuron j in layer l passed the ReLU gate (on-shell), else 0.

Summing all paths (K=H) reproduces the weight-only linear part of the network:
    Σ_paths A(π) = [WL · diag(mL) · … · W₁ · diag(m₁) · W₀ · x]_k

Adding the bias offset (actual_logit - weight_logit) recovers the exact output.
K=1 is leading-order perturbation theory; K=H is the exact, non-perturbative result.

RELEVANT LITERATURE:
──────────────────────────────────────────────────────────────────────────────
1.  Feynman (1948) Rev. Mod. Phys. 20, 367
    — Original path integral: quantum amplitude = sum over all paths.

2.  Stoudenmire & Schwab (2016) NeurIPS
    "Supervised Learning with Tensor Networks"
    — Explicit path-sum / tensor-train ML; closest formal analogy to this work.

3.  Cohen & Shashua (2016) JMLR 18(18):1-50
    "Inductive Bias of Deep Convolutional Networks through the Lens of Rank"
    — Path counting and correlation order; paths ↔ Feynman diagrams.

4.  Lin, Tegmark & Rolnick (2017) J. Stat. Phys. 168, 1223
    "Why Does Deep and Cheap Learning Work So Well?"
    — Physics-inspired hierarchy; depth ↔ efficient factorisation of path sums.

5.  Baldi & Vershynin (2019) Neural Networks 116, 288-311
    "The Capacity of Feedforward Neural Networks"
    — Path counting and VC dimension upper bounds.

6.  Hanin & Rolnick (2019) NeurIPS
    "Deep ReLU Networks Have Surprisingly Few Activation Patterns"
    — ReLU masks select an exponentially small subset of paths (on-shell condition).

7.  Yaida (2020) "Non-Gaussian processes and neural networks at finite widths"
    — 1/H loop expansion of neural-network field theory; higher-order Feynman diagrams.

8.  Neal (1996) "Bayesian Learning for Neural Networks" Springer
    — Partition function / path integral connection to Bayesian neural nets.

9.  Domingos (2020) PNAS 117(47), 30043
    "Every Model Learned by Gradient Descent Is Approximately a Kernel Machine"
    — Gradient descent solutions as path-kernel expansions.

10. Vyas, Atanasov & Bordelon et al. (2022)
    "Feature-Learning Networks Are Consistent Across Widths"
    — Connections between path sums, NTK limit, and finite-width corrections.
"""

import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import curve_fit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
INPUT_SIZE        = 64    # 8×8 pixels from load_digits
HIDDEN_SIZE       = 64    # neurons per hidden layer  H
NUM_HIDDEN_LAYERS = 15     # L hidden layers
NUM_CLASSES       = 10
LR                = 1e-3
EPOCHS            = 300
BATCH_SIZE        = 64
SEED              = 42
ACTIVATION        = 'relu' # relu or linear
OUTPUT_DIR        = './figures'

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
#  1.  PURE-NUMPY NEURAL NETWORK
# ═══════════════════════════════════════════════════════════

def relu(z):           return np.maximum(0, z)
def lin(z):            return z
def relu_grad(z):      return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


class FCNetwork:
    """
    Fully-connected ReLU network: [INPUT] → [H]xL → [10]
    Implemented in pure NumPy with Adam optimizer.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_hidden_layers=NUM_HIDDEN_LAYERS, num_classes=NUM_CLASSES,
                 activation = ACTIVATION):
        self.H = hidden_size
        self.L = num_hidden_layers
        self.C = num_classes
        self.activation = relu if activation == 'relu' else lin

        # Xavier / He initialisation
        rng    = np.random.default_rng(SEED)
        sizes  = [input_size] + [hidden_size] * num_hidden_layers + [num_classes]
        self.W = []
        self.b = []
        for i_in, i_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / i_in)   # He init (good for ReLU)
            self.W.append(rng.normal(0, std, (i_out, i_in)).astype(np.float64))
            self.b.append(np.zeros(i_out, dtype=np.float64))

        # Adam state
        self._adam_init()

    # ── forward ────────────────────────────────────────────
    def forward(self, X):
        """
        X : [N, input_size]
        Returns logits [N, 10]
        """
        h = X
        for l in range(self.L):
            h = self.activation(h @ self.W[l].T + self.b[l])
        return h @ self.W[-1].T + self.b[-1]

    def forward_with_masks(self, X):
        """
        Forward pass that also records ReLU activation masks.
        Returns
        -------
        logits     : [N, 10]
        relu_masks : list of L arrays, each [N, H]  (1 = active, 0 = dead)
        """
        h = X
        relu_masks = []
        for l in range(self.L):
            z    = h @ self.W[l].T + self.b[l]   # pre-activation [N, H]
            mask = (z > 0).astype(np.float64)      # ReLU gate
            h    = z * mask                        # post-activation
            relu_masks.append(mask)
        logits = h @ self.W[-1].T + self.b[-1]
        return logits, relu_masks

    # ── backward ───────────────────────────────────────────
    def _forward_cache(self, X):
        """Forward pass caching pre- and post-activations for backprop."""
        zs, hs = [], [X]
        h = X
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            zs.append(z)
            h = self.activation(z)
            hs.append(h)
        logits = h @ self.W[-1].T + self.b[-1]
        return logits, zs, hs

    def compute_gradients(self, X, y_onehot):
        N       = X.shape[0]
        logits, zs, hs = self._forward_cache(X)
        probs   = softmax(logits)
        loss    = cross_entropy(probs, y_onehot)

        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # Output layer gradient
        delta = (probs - y_onehot) / N              # [N, C]
        dW[-1] = delta.T @ hs[-1]                   # [C, H]
        db[-1] = delta.sum(axis=0)                  # [C]

        # Hidden layer gradients (backprop)
        for l in range(self.L - 1, -1, -1):
            delta = (delta @ self.W[l + 1]) * relu_grad(zs[l])   # [N, H]
            dW[l] = delta.T @ hs[l]                               # [H, in]
            db[l] = delta.sum(axis=0)                             # [H]

        return loss, dW, db

    # ── Adam optimiser ─────────────────────────────────────
    def _adam_init(self):
        self._t  = 0
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]

    def adam_step(self, dW, db, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        for l in range(len(self.W)):
            self._mW[l] = beta1 * self._mW[l] + (1 - beta1) * dW[l]
            self._vW[l] = beta2 * self._vW[l] + (1 - beta2) * dW[l]**2
            self.W[l]  -= lr * (self._mW[l] / bc1) / (np.sqrt(self._vW[l] / bc2) + eps)

            self._mb[l] = beta1 * self._mb[l] + (1 - beta1) * db[l]
            self._vb[l] = beta2 * self._vb[l] + (1 - beta2) * db[l]**2
            self.b[l]  -= lr * (self._mb[l] / bc1) / (np.sqrt(self._vb[l] / bc2) + eps)

    # ── train ──────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        N = X_train.shape[0]
        C = self.C
        y_ohe = np.eye(C)[y_train]   # one-hot

        print(f"\n{'═'*62}")
        print(f"  Model: [{X_train.shape[1]}] → [{self.H}]x{self.L} → [{self.C}]")
        print(f"  Params: {sum(w.size+b.size for w,b in zip(self.W,self.b)):,}")
        print(f"  Training for {epochs} epochs …")
        print(f"{'═'*62}")

        for epoch in range(1, epochs + 1):
            # Mini-batch SGD
            idx = np.random.permutation(N)
            total_loss = 0.0
            for start in range(0, N, batch_size):
                sl = idx[start:start + batch_size]
                loss, dW, db = self.compute_gradients(X_train[sl], y_ohe[sl])
                self.adam_step(dW, db)
                total_loss += loss * len(sl)

            if epoch % 50 == 0 or epoch == 1:
                val_logits = self.forward(X_val)
                val_acc    = accuracy(val_logits, y_val)
                tr_logits  = self.forward(X_train)
                tr_acc     = accuracy(tr_logits, y_train)
                print(f"  Epoch {epoch:4d}/{epochs}  "
                      f"loss={total_loss/N:.4f}  "
                      f"train_acc={100*tr_acc:.1f}%  "
                      f"val_acc={100*val_acc:.1f}%")

        final_acc = accuracy(self.forward(X_val), y_val)
        print(f"\n  ► Final validation accuracy: {100*final_acc:.2f}%\n")
        return final_acc


# ═══════════════════════════════════════════════════════════
#  2.  DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_data():
    """
    Load sklearn's load_digits (8x8 images, 10 classes, 1797 samples).
    Returns (X_train, X_val, X_test, y_train, y_val, y_test) as float64 numpy arrays.
    """
    digits  = load_digits()
    X, y    = digits.data.astype(np.float64), digits.target

    # Split: 70% train / 15% val / 15% test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    # Standardise (fit on train only)
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_tr)
    X_val   = scaler.transform(X_val)
    X_te    = scaler.transform(X_te)

    print(f"  Dataset: sklearn digits  (8x8 pixels, 10 classes)")
    print(f"  Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ═══════════════════════════════════════════════════════════
#  3.  PATH TRACING  (the core algorithm)
# ═══════════════════════════════════════════════════════════

def sparsify_top_k(matrix, k):
    """
    Keep the K entries with largest |value| per row; zero the rest.

    This implements the 'beam search' / hierarchical pruning step:
    from each input pixel only the K strongest signal pathways survive.

    Parameters
    ----------
    matrix : ndarray [n_pixels, n_neurons]
    k      : int  — beam width (= paths_per_pixel)

    Returns
    -------
    ndarray [n_pixels, n_neurons]  (sparse; at most k non-zeros per row)
    """
    n_neurons = matrix.shape[1]
    if k >= n_neurons:
        return matrix                                  # no pruning needed

    kth      = n_neurons - k
    top_idx  = np.argpartition(np.abs(matrix), kth, axis=1)[:, kth:]   # [n_pixels, k]
    result   = np.zeros_like(matrix)
    rows     = np.arange(matrix.shape[0])[:, np.newaxis]                # [n_pixels, 1]
    result[rows, top_idx] = matrix[rows, top_idx]
    return result


def compute_path_logits(x_n, weights, masks_n, paths_per_pixel):
    """
    Compute path-integral-inspired logit estimate for one sample.

    Propagates each pixel's signal through the network layer-by-layer,
    keeping only the `paths_per_pixel` strongest connections at each layer.

    Parameters
    ----------
    x_n             : ndarray [input_size]  — one input sample
    weights         : list of ndarray       — [W₀, W₁, …, Wout], shape [out,in] each
    masks_n         : list of ndarray       — [m₀, …, mL], shape [H] each;
                      ReLU on-shell masks from the actual forward pass
    paths_per_pixel : int K ∈ [1, H]

    Returns
    -------
    ndarray [num_classes]  — linear path contribution (biases handled separately)

    MATH (see module docstring for physics analogy):
        current[i, j]  = contribution of pixel i to neuron j at current layer
        Init:   current = W₀ᵀ ⊙ x[:, None]     → [I, H],  current[i,j] = W₀[j,i]·xᵢ
        Per layer l:
            current  ←  (current @ W_l^T)            propagate amplitudes
                      ⊙  m^l[None, :]               on-shell / ReLU gate
                      →  sparsify_top_k(., K)       perturbative truncation
        Output:
            path_logits = current @ Wout^T          → [I, C]
            return path_logits.sum(axis=0)           → [C] (sum over pixels)
    """
    # ── First hidden layer ──────────────────────────────────
    # weights[0] : [H, I],  weights[0].T : [I, H]
    # current[i, j] = weights[0][j, i] * x_n[i]
    current = weights[0].T * x_n[:, np.newaxis]        # [I, H]
    current *= masks_n[0][np.newaxis, :]               # on-shell gate
    current  = sparsify_top_k(current, paths_per_pixel)

    # ── Remaining hidden layers ─────────────────────────────
    for l in range(1, len(weights) - 1):               # l = 1 … L-1
        current  = current @ weights[l].T              # [I, H_{l+1}]
        current *= masks_n[l][np.newaxis, :]           # on-shell gate
        current  = sparsify_top_k(current, paths_per_pixel)

    # ── Output layer (no ReLU, no pruning) ──────────────────
    path_logits = current @ weights[-1].T              # [I, C]
    return path_logits.sum(axis=0)                     # [C]


# ═══════════════════════════════════════════════════════════
#  4.  EVALUATION LOOP
# ═══════════════════════════════════════════════════════════

def evaluate_path_accuracy(model, X_test, y_test, k_values):
    """
    Evaluate prediction accuracy for each K = paths_per_pixel.

    Key design
    ──────────
    1. Run full forward pass → actual_logits, relu_masks
    2. Compute full_path_logits  (K = H, no pruning)
    3. bias_offset = actual_logits - full_path_logits  (exact per-sample correction)
       → full_path_logits + bias_offset ≡ actual_logits  ✓
    4. For each K:  final_logits = path_logits(K) + bias_offset
       As K → H:   path_logits(K) → full_path_logits  →  accuracy = normal_acc  ✓
    """
    N = X_test.shape[0]

    # ── Full forward pass ───────────────────────────────────
    actual_logits, relu_masks_list = model.forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    print(f"\n{'═'*62}")
    print(f"  Normal network accuracy on {N} test samples: {100*normal_acc:.2f}%")
    print(f"{'═'*62}")

    weights = model.W   # list of weight matrices

    # ── Full path logits (K = H) for bias correction ────────
    print(f"\n  [1/2] Computing full-path logits (K={HIDDEN_SIZE}) for bias correction …")
    t0 = time.time()
    full_path_logits = np.zeros((N, NUM_CLASSES))
    for n in range(N):
        masks_n = [relu_masks_list[l][n] for l in range(len(relu_masks_list))]
        full_path_logits[n] = compute_path_logits(X_test[n], weights, masks_n, HIDDEN_SIZE)
    print(f"     done in {time.time()-t0:.1f}s")

    # bias_offset captures all bias effects propagated through the network
    bias_offset = actual_logits - full_path_logits   # [N, C]

    # ── Sanity check: K=H + bias ≡ actual ──────────────────
    check_acc = accuracy(full_path_logits + bias_offset, y_test)
    assert np.isclose(check_acc, normal_acc, atol=1e-9), \
        f"FATAL: K=full reconstruction mismatch ({check_acc:.6f} vs {normal_acc:.6f})"
    print(f"  ✓ Sanity check: K={HIDDEN_SIZE}+bias_offset accuracy = {100*check_acc:.2f}%"
          f"  (matches normal: {100*normal_acc:.2f}%) ✓\n")

    # ── Evaluate each K ─────────────────────────────────────
    print(f"  [2/2] Evaluating {len(k_values)} K-values …")
    accuracies = {}
    for K in k_values:
        t1 = time.time()
        path_logits = np.zeros((N, NUM_CLASSES))
        for n in range(N):
            masks_n = [relu_masks_list[l][n] for l in range(len(relu_masks_list))]
            path_logits[n] = compute_path_logits(X_test[n], weights, masks_n, K)

        final_logits = path_logits + bias_offset
        acc = accuracy(final_logits, y_test)
        accuracies[K] = acc
        if (K & (K - 1)) == 0: # print only powers of 2
            print(f"     K={K:4d}  acc={100*acc:.2f}%  "
                f"({100*acc/normal_acc:.1f}% of full net)  "
                f"[{time.time()-t1:.2f}s]")

    return accuracies, normal_acc


# ═══════════════════════════════════════════════════════════
#  5.  FITTING  accuracy(K)
# ═══════════════════════════════════════════════════════════

def fit_accuracy_curve(k_values, accuracies, normal_acc):
    """
    Fit: acc(K) = A_0 + (A_inf - A_0) / (1 + exp(-beta * (K - K_0)))
    """
    k_arr   = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])

    def sigmoid_fit(K, A_inf, A_0, K_0, beta):
        K = np.array(K, dtype=float)
        exponent = np.clip(-beta * (K - K_0), -500, 500)
        return A_0 + (A_inf - A_0) / (1.0 + np.exp(exponent))

    popt = perr = None
    try:
        # Initial guesses: A_inf, A_0, K_0, beta
        p0     = [normal_acc, acc_arr[0], np.median(k_arr), 0.1]
        bounds = ([0.0, 0.0, 0.0, 0.0001], [1.0, 1.0, max(k_arr) * 2.0, 10.0])
        
        popt, pcov = curve_fit(sigmoid_fit, k_arr, acc_arr, p0=p0,
                               bounds=bounds, maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        A_inf, A_0, K_0, beta = popt
        g_eff = np.exp(-beta)

        # Calculate R-squared (Goodness of fit)
        residuals = acc_arr - sigmoid_fit(k_arr, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((acc_arr - np.mean(acc_arr))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Add to the existing print block

        
        print(f"\n{'═'*62}")
        print("  FITTED SIGMOID MODEL:")
        print("  acc(K) = A₀ + (A∞ - A₀) / (1 + exp(-β(K - K₀)))")
        print(f"{'─'*62}")
        print(f"  A∞   (asymptotic)      = {100*A_inf:.2f}% ± {100*perr[0]:.2f}%")
        print(f"  A₀   (baseline)        = {100*A_0:.2f}% ± {100*perr[1]:.2f}%")
        print(f"  K₀   (critical path)   = {K_0:.2f} ± {perr[2]:.2f} paths")
        print(f"  β    (growth rate)     = {beta:.4f} ± {perr[3]:.4f}")
        print(f"  g_eff (coupling const) = {g_eff:.4f} ± {(g_eff*perr[3]):.4f}")
        print(f"  R^2  (goodness of fit) = {r_squared:.4f}")
        print(f"{'═'*62}")
    except Exception as e:
        print(f"  ⚠ Curve fitting failed: {e}")

    return popt, sigmoid_fit, r_squared


# ═══════════════════════════════════════════════════════════
#  6.  PLOTTING
# ═══════════════════════════════════════════════════════════

def make_plot(k_values, accuracies, normal_acc, popt, fit_fn):
    k_arr   = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values]) * 100

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    k_fine = np.linspace(1, max(k_values), 500)
    fit_fine = fit_fn(k_fine, *popt) * 100 if popt is not None else None

    # Strict power-of-2 ticks
    max_k = int(max(k_arr))
    pow2_ticks = [2**i for i in range(int(np.log2(max_k)) + 1)]

    kw_sc  = dict(s=40, zorder=6, color='steelblue', edgecolors='navy', linewidths=0.5)
    kw_ln  = dict(alpha=0.45, color='steelblue', linewidth=1.8)
    kw_ref = dict(color='crimson', linestyle='--', linewidth=2.0,
                  label=f'Full network  {100*normal_acc:.1f}\%')
    kw_fit = dict(color='limegreen', linewidth=2.4, alpha=0.9,
                  label='Sigmoid fit')

    # Panel 1 – linear scale
    ax1.scatter(k_arr, acc_arr, **kw_sc, label='Path algorithm')
    ax1.plot(k_arr, acc_arr, **kw_ln)
    ax1.axhline(normal_acc * 100, **kw_ref)
    if fit_fine is not None:
        ax1.plot(k_fine, fit_fine, **kw_fit)
    ax1.set_xticks(pow2_ticks)
    ax1.set_xticklabels([str(t) for t in pow2_ticks], fontsize=9)
    ax1.set_xlabel('Paths per pixel  K', fontsize=11)
    ax1.set_ylabel('Accuracy (\%)', fontsize=11)
    ax1.set_title('Accuracy vs Paths per Pixel  (linear)', fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Panel 2 – log₂ scale
    ax2.scatter(k_arr, acc_arr, **kw_sc, label='Path algorithm')
    ax2.plot(k_arr, acc_arr, **kw_ln)
    ax2.axhline(normal_acc * 100, **kw_ref)
    if fit_fine is not None:
        ax2.plot(k_fine, fit_fine, **kw_fit)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(pow2_ticks)
    ax2.set_xticklabels([str(t) for t in pow2_ticks], fontsize=9)
    ax2.set_xlabel(r'Paths per pixel  K  ($log_2$)', fontsize=11)
    ax2.set_ylabel('Accuracy (\%)', fontsize=11)
    ax2.set_title(r'Accuracy vs Paths per Pixel  ($log_2$)', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3 – incremental gain over K=1
    baseline = acc_arr[0]
    delta    = acc_arr - baseline
    
    # Isolate data corresponding to powers of 2
    pow2_indices = [np.where(k_arr == t)[0][0] for t in pow2_ticks if t in k_arr]
    pow2_deltas  = [delta[i] for i in pow2_indices]
    pow2_labels  = [str(int(k_arr[i])) for i in pow2_indices]
    
    # Plot only the filtered data
    bar_positions = range(len(pow2_indices))
    bars = ax3.bar(bar_positions, pow2_deltas, color='steelblue',
                   edgecolor='navy', alpha=0.8)
    
    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels(pow2_labels, fontsize=9)
    ax3.set_xlabel('Paths per pixel  K', fontsize=11)
    ax3.set_ylabel('Accuracy gain over K=1  (pp)', fontsize=11)
    ax3.set_title('Incremental Gain from Adding Paths', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # Annotate all rendered bars
    for i, d in enumerate(pow2_deltas):
        ax3.text(i, d + 0.1, f'{d:.1f}', ha='center', va='bottom', fontsize=8)

    # Panel 4 – fraction of full-network accuracy
    frac = acc_arr / (normal_acc * 100) * 100
    ax4.scatter(k_arr, frac, **kw_sc, label='\% of full accuracy')
    ax4.plot(k_arr, frac, **kw_ln)
    ax4.axhline(100, **kw_ref)
    ax4.set_xscale('log', base=2)
    ax4.set_xticks(pow2_ticks)
    ax4.set_xticklabels([str(t) for t in pow2_ticks], fontsize=9)
    ax4.set_xlabel(r'Paths per pixel  K  ($log_2$)', fontsize=11)
    ax4.set_ylabel('\% of full-network accuracy', fontsize=11)
    ax4.set_title('Recovery of Full-Network Performance', fontsize=11)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Add fit annotation
    if popt is not None:
        A_inf, A_0, K_0, beta = popt
        g_eff = np.exp(-beta)
        ann = (fr"$\mathrm{{acc}}(K) = A_0 + \frac{{A_\infty - A_0}}{{1 + e^{{-\beta(K - K_0)}}}}$" "\n"
               fr"$A_\infty = {100*A_inf:.1f}\%$  $A_0 = {100*A_0:.1f}\%$" "\n"
               fr"$K_0 = {K_0:.1f}$  $\beta = {beta:.3f}$  $g_{{eff}} = {g_eff:.4f}$")
        ax1.annotate(ann, xy=(0.2, 0.05), xycoords='axes fraction',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.4',
                     fc='lightyellow', ec='gray', alpha=0.9))

    fig.suptitle(
        f'Path Integral Analogy in Neural Networks\n'
        f'Pure-NumPy FC-{NUM_HIDDEN_LAYERS}x{HIDDEN_SIZE} on sklearn Digits  |  '
        f'Full-network acc: {100*normal_acc:.2f}\%',
        fontsize=13, y=1.01
    )

    out_path = os.path.join(OUTPUT_DIR, 'path_fit.png')
    plt.show()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot → {out_path}")
    return out_path

# ═══════════════════════════════════════════════════════════
#  7.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════

def print_summary(k_values, accuracies, normal_acc):
    regimes = {
        1: 'Leading order  (dominant diagram)',
        2: '1st correction',
        4: 'Perturbative',
        8: 'Perturbative++',
        16: 'Intermediate',
        32: 'Near-complete',
        64: 'Full  (K=H)',
    }
    print(f"\n{'═'*72}")
    print(f"  {'K':>5}  {'Accuracy':>9}  {'% of Full':>10}  {'Gain vs K=1':>13}  Regime")
    print(f"{'─'*72}")
    acc1 = accuracies[k_values[0]]
    for k in k_values:
        acc = accuracies[k]
        print(f"  {k:>5}  {100*acc:>8.2f}%  {100*acc/normal_acc:>9.1f}%  "
              f"{100*(acc-acc1):>+12.2f}pp  {regimes.get(k,'')}")
    print(f"{'─'*72}")
    print(f"  {'Full':>5}  {100*normal_acc:>8.2f}%  {'100.0':>9}%  "
          f"{100*(normal_acc-acc1):>+12.2f}pp  True network output")
    print(f"{'═'*72}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()

    # 1. Load data
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()

    # 2. Train
    model     = FCNetwork()
    final_acc = model.train(X_tr, y_tr, X_val, y_val)

    # 3. K values: evaluate all integers from 1 to HIDDEN_SIZE
    k_values = list(range(1, HIDDEN_SIZE + 1))
    print(f"\n  K values to test: 1 through {HIDDEN_SIZE}")

    # 4. Evaluate path accuracy
    accuracies, normal_acc = evaluate_path_accuracy(model, X_te, y_te, k_values)

    # 5. Summary table
    print_summary(k_values, accuracies, normal_acc)

    # 6. Fit sigmoid curve
    popt, fit_fn, r_squared = fit_accuracy_curve(k_values, accuracies, normal_acc)

    # 7. Plot
    plot_path = make_plot(k_values, accuracies, normal_acc, popt, fit_fn)

    with open('pruning_coupling/accuracies.txt', 'w') as f:
        acc_formatted = {int(k): round(float(accuracies[k]), 4) for k in k_values}
        f.write(f"acc_nn_measured = {acc_formatted}\n")

    print(f"\n  Total runtime: {time.time()-t_start:.1f}s")
    print("\n  ✓ Done!")