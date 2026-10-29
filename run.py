"""Run the game by executing this file directly (e.g. PyCharm's Run).

The game code lives in the `rhombic_game` package and uses relative
imports, so the package's own modules cannot be run as scripts --
``python -m rhombic_game`` is the canonical entry point.  This
launcher sits at the project root, which puts the root on sys.path,
so ``import rhombic_game`` resolves and the package loads properly.
Equivalent PyCharm setup without this file: a run configuration with
"Module name" = ``rhombic_game`` instead of a script path.
"""

from rhombic_game.app import main

if __name__ == "__main__":
    raise SystemExit(main())
