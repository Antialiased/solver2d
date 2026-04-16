"""Thin shim: the 1D SI prototype now lives in the `si1d/` package.

Run via:  python -m experiments.si1d.run --test ...
or:       python experiments/si_1d_stack.py --test ...
"""

from si1d.run import main

if __name__ == "__main__":
    main()
