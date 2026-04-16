"""Verify SNH energy gradient and Hessian against sympy symbolic and finite differences."""
import numpy as np
import sympy as sp


def sympy_verify():
    """Derive SNH gradient and Hessian symbolically and compare with energy.py."""
    F11, F12, F21, F22 = sp.symbols("F11 F12 F21 F22", real=True)
    mu_s, lam_s = sp.symbols("mu lam", positive=True)

    F_vec = sp.Matrix([F11, F12, F21, F22])
    Ic = F11**2 + F12**2 + F21**2 + F22**2
    J = F11 * F22 - F12 * F21
    alpha = 1 + mu_s / lam_s
    Psi = sp.Rational(1, 2) * (mu_s * (Ic - 2) + lam_s * (J - alpha) ** 2)

    grad = sp.Matrix([sp.diff(Psi, f) for f in F_vec])
    hess = sp.Matrix(4, 4, lambda i, j: sp.diff(grad[i], F_vec[j]))

    # Substitute numerical values and compare
    mu_val, lam_val = 100.0, 50.0
    F_val = [1.2, 0.3, -0.1, 0.9]
    subs = {F11: F_val[0], F12: F_val[1], F21: F_val[2], F22: F_val[3],
            mu_s: mu_val, lam_s: lam_val}

    psi_sym = float(Psi.subs(subs))
    grad_sym = np.array([float(g.subs(subs)) for g in grad], dtype=float)
    hess_sym = np.array([[float(hess[i, j].subs(subs)) for j in range(4)]
                         for i in range(4)], dtype=float)

    from . import energy
    F_np = np.array(F_val, dtype=float)
    psi_num = energy.psi(F_np, mu_val, lam_val)
    grad_num = energy.pk1(F_np, mu_val, lam_val)
    hess_num = energy.hessian(F_np, mu_val, lam_val)

    print("=== Sympy verification ===")
    print(f"  Psi:  sympy={psi_sym:.10f}  analytic={psi_num:.10f}  "
          f"err={abs(psi_sym - psi_num):.2e}")
    print(f"  grad: max err = {np.max(np.abs(grad_sym - grad_num)):.2e}")
    print(f"  hess: max err = {np.max(np.abs(hess_sym - hess_num)):.2e}")

    assert abs(psi_sym - psi_num) < 1e-10, "Energy mismatch"
    assert np.max(np.abs(grad_sym - grad_num)) < 1e-10, "Gradient mismatch"
    assert np.max(np.abs(hess_sym - hess_num)) < 1e-10, "Hessian mismatch"
    print("  PASS")

    # Also verify the symbolic expressions match our expected forms
    cof = sp.Matrix([F22, -F21, -F12, F11])
    expected_grad = mu_s * F_vec + lam_s * (J - alpha) * cof
    diff = sp.simplify(grad - expected_grad)
    assert diff.equals(sp.zeros(4, 1)), f"Gradient form mismatch: {diff}"
    print("  Gradient form matches expected cofactor formula: PASS")


def finite_diff_verify():
    """Verify gradient and Hessian via central finite differences."""
    from . import energy

    rng = np.random.default_rng(42)
    mu_val, lam_val = 100.0, 50.0

    for trial in range(10):
        F = rng.standard_normal(4) * 0.5 + np.array([1, 0, 0, 1])

        # Gradient check
        grad_analytic = energy.pk1(F, mu_val, lam_val)
        grad_fd = np.zeros(4)
        eps = 1e-7
        for i in range(4):
            Fp = F.copy(); Fp[i] += eps
            Fm = F.copy(); Fm[i] -= eps
            grad_fd[i] = (energy.psi(Fp, mu_val, lam_val) -
                          energy.psi(Fm, mu_val, lam_val)) / (2 * eps)

        grad_err = np.max(np.abs(grad_analytic - grad_fd))

        # Hessian check
        hess_analytic = energy.hessian(F, mu_val, lam_val)
        hess_fd = np.zeros((4, 4))
        for i in range(4):
            Fp = F.copy(); Fp[i] += eps
            Fm = F.copy(); Fm[i] -= eps
            hess_fd[:, i] = (energy.pk1(Fp, mu_val, lam_val) -
                             energy.pk1(Fm, mu_val, lam_val)) / (2 * eps)

        hess_err = np.max(np.abs(hess_analytic - hess_fd))

        status = "PASS" if grad_err < 1e-5 and hess_err < 1e-4 else "FAIL"
        if trial == 0 or status == "FAIL":
            print(f"  trial {trial}: grad_err={grad_err:.2e}  "
                  f"hess_err={hess_err:.2e}  {status}")

        assert grad_err < 1e-5, f"Gradient FD check failed: {grad_err:.2e}"
        assert hess_err < 1e-4, f"Hessian FD check failed: {hess_err:.2e}"

    print(f"  All 10 random F configs: PASS")


def verify_1d_reduction():
    """Verify that SNH gives d^2 Psi/ds^2 = k for uniaxial stretch F=diag(1,s)."""
    from . import energy

    k = 1000.0
    nu = 0.3
    mu_val, lam_val = energy.lame_from_k(k, nu)

    # Uniaxial stretch F = diag(1, s): Ic = 1 + s^2, J = s
    # d^2 Psi/ds^2 at s=1 should be mu + lam = k
    eps = 1e-6
    s = 1.0
    def psi_s(s_val):
        F = np.array([1.0, 0, 0, s_val], dtype=float)
        return energy.psi(F, mu_val, lam_val)

    d2 = (psi_s(s + eps) - 2 * psi_s(s) + psi_s(s - eps)) / eps**2
    print(f"\n=== 1D reduction check (uniaxial stretch) ===")
    print(f"  k={k}  nu={nu}  mu={mu_val:.4f}  lam={lam_val:.4f}")
    print(f"  d^2 Psi/ds^2 at s=1 = {d2:.4f}  (target: {k})")
    assert abs(d2 - k) < 1.0, f"1D reduction failed: {d2} vs {k}"

    # Also check uniform scaling for reference
    def psi_uni(s_val):
        F = np.array([s_val, 0, 0, s_val], dtype=float)
        return energy.psi(F, mu_val, lam_val)
    d2_uni = (psi_uni(1+eps) - 2*psi_uni(1) + psi_uni(1-eps)) / eps**2
    print(f"  uniform-scaling d^2 Psi/ds^2 = {d2_uni:.4f}  (= 4*lam = {4*lam_val:.4f})")
    print(f"  PASS")


def verify_spd_projection():
    """Test that hessian_spd always returns a positive semi-definite matrix."""
    from . import energy

    rng = np.random.default_rng(123)
    mu_val, lam_val = 100.0, 50.0
    n_fail = 0

    for trial in range(100):
        # Include some near-singular and inverted F
        F = rng.standard_normal(4) * 1.5
        if trial < 10:
            F = np.array([1, 0, 0, 1], dtype=float) * (0.1 * trial)

        H = energy.hessian_spd(F, mu_val, lam_val)
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) < -1e-10:
            print(f"  trial {trial}: min eigval = {np.min(eigvals):.2e}  FAIL")
            n_fail += 1

    print(f"\n=== SPD projection check ===")
    print(f"  {100 - n_fail}/100 passed SPD check")
    assert n_fail == 0, f"{n_fail} SPD failures"
    print(f"  PASS")


def main():
    print("Verifying 2D Stable Neo-Hookean energy implementation\n")
    sympy_verify()
    print()
    print("=== Finite difference verification ===")
    finite_diff_verify()
    verify_1d_reduction()
    verify_spd_projection()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
