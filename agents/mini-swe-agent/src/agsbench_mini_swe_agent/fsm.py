"""Module-level fast-service manager.

Lives in its own module so ``request.py`` and ``runner.py`` can both import
``mini_swe_fsm`` without circular dependencies.
"""

from fast_service import FastServiceManager

mini_swe_fsm = FastServiceManager()
