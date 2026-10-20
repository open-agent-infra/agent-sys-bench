"""Module-level fast-service manager for the browsergym benchmark service.

Lives in its own module so ``request.py`` and ``runner.py`` can both import
``agsbench_webagent_fsm`` without circular dependencies.
"""

from fast_service import FastServiceManager

agsbench_webagent_fsm = FastServiceManager()
