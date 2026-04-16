"""CLI dispatcher for the 1D SI prototype tests."""

import argparse
import os

from .tests_basic import (test_single_body, test_stack, test_stiffness_sweep,
                          animate_sweep, animate_stack)
from .tests_cradle import (test_cradle, test_cradle_sweep, test_cradle_ideal,
                           animate_cradle_sweep, animate_cradle_ideal)
from .tests_plastic import (test_plasticity_single, test_plasticity_stack,
                            test_plasticity_sweep, animate_plasticity_stack)
from .tests_robustness import (test_fast_impact, test_mixed_ratios,
                               test_convergence_scaling, test_mass_ratio,
                               test_nonlinear_inversion)
from .tests_solver_val import (test_warm_start_benefit, test_long_chain_scaling,
                               test_dt_refinement, test_dt_refinement_nonlinear,
                               test_mu_frac_sensitivity, test_substep_tradeoff)
from .tests_material import (test_free_oscillation_drift, test_asymmetric_cubic,
                             test_cor_soft_contact)
from .tests_bouncy import (test_bouncy_balls, test_two_ball_bouncy,
                           test_two_ball_attract, animate_two_ball_attract,
                           animate_two_ball_attract_be_vs_exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="all",
                    choices=["single", "stack", "sweep", "animate", "animate_sweep",
                             "cradle", "cradle_sweep", "animate_cradle",
                             "cradle_ideal", "animate_cradle_ideal",
                             "plastic_single", "plastic_stack", "plastic_sweep",
                             "animate_plastic",
                             "fast_impact", "mixed_ratios", "convergence", "mass_ratio",
                             "nonlinear", "warm_start", "long_chain",
                             "dt_refine", "dt_refine_nl", "mu_frac", "substep",
                             "free_osc", "asym_cubic", "cor_soft",
                             "bouncy", "two_ball", "two_attract",
                             "anim_attract", "anim_attract_cmp",
                             "robustness", "all"])
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.test in ("single", "all"):
        test_single_body(args.outdir)
    if args.test in ("stack", "all"):
        test_stack(args.outdir)
    if args.test in ("sweep", "all"):
        test_stiffness_sweep(args.outdir)
    if args.test in ("animate", "all"):
        animate_stack(args.outdir)
    if args.test in ("animate_sweep", "all"):
        animate_sweep(args.outdir)
    if args.test in ("cradle", "all"):
        test_cradle(args.outdir)
    if args.test in ("cradle_sweep", "all"):
        test_cradle_sweep(args.outdir)
    if args.test in ("animate_cradle", "all"):
        animate_cradle_sweep(args.outdir)
    if args.test in ("cradle_ideal", "all"):
        test_cradle_ideal(args.outdir)
    if args.test in ("animate_cradle_ideal", "all"):
        animate_cradle_ideal(args.outdir)
    if args.test in ("plastic_single", "all"):
        test_plasticity_single(args.outdir)
    if args.test in ("plastic_stack", "all"):
        test_plasticity_stack(args.outdir)
    if args.test in ("plastic_sweep", "all"):
        test_plasticity_sweep(args.outdir)
    if args.test in ("animate_plastic", "all"):
        animate_plasticity_stack(args.outdir)
    if args.test in ("fast_impact", "robustness", "all"):
        test_fast_impact(args.outdir)
    if args.test in ("mixed_ratios", "robustness", "all"):
        test_mixed_ratios(args.outdir)
    if args.test in ("convergence", "robustness", "all"):
        test_convergence_scaling(args.outdir)
    if args.test in ("mass_ratio", "robustness", "all"):
        test_mass_ratio(args.outdir)
    if args.test in ("nonlinear", "robustness", "all"):
        test_nonlinear_inversion(args.outdir)
    if args.test in ("warm_start", "all"):
        test_warm_start_benefit(args.outdir)
    if args.test in ("long_chain", "all"):
        test_long_chain_scaling(args.outdir)
    if args.test in ("dt_refine", "all"):
        test_dt_refinement(args.outdir)
    if args.test in ("dt_refine_nl", "all"):
        test_dt_refinement_nonlinear(args.outdir)
    if args.test in ("mu_frac", "all"):
        test_mu_frac_sensitivity(args.outdir)
    if args.test in ("substep", "all"):
        test_substep_tradeoff(args.outdir)
    if args.test in ("free_osc", "all"):
        test_free_oscillation_drift(args.outdir)
    if args.test in ("asym_cubic", "all"):
        test_asymmetric_cubic(args.outdir)
    if args.test in ("cor_soft", "all"):
        test_cor_soft_contact(args.outdir)
    if args.test in ("bouncy", "all"):
        test_bouncy_balls(args.outdir)
    if args.test in ("two_ball", "all"):
        test_two_ball_bouncy(args.outdir)
    if args.test in ("two_attract", "all"):
        test_two_ball_attract(args.outdir)
    if args.test in ("anim_attract",):
        animate_two_ball_attract(args.outdir)
    if args.test in ("anim_attract_cmp",):
        animate_two_ball_attract_be_vs_exp(args.outdir)

    print(f"\nOutputs written to {args.outdir}")


if __name__ == "__main__":
    main()
