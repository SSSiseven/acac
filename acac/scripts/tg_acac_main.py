"""TG entry point; configuration and environment setup are shared with ACAC."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    if not any(arg == "--alg_name" or arg.startswith("--alg_name=")
               for arg in sys.argv[1:]):
        sys.argv.extend(["--alg_name", "tg_acac_no_dependency"])
    runpy.run_path(str(Path(__file__).with_name("acac_main.py")), run_name="__main__")
