"""Module-level fast-service manager.

Lives in its own module so ``request.py`` and ``runner.py`` can both
import ``agsbench_pi_autoresearch_fsm`` without circular dependencies.
"""

from fast_service import FastServiceManager

agsbench_pi_autoresearch_fsm = FastServiceManager()
