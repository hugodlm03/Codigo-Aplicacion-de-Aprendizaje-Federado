# hfedxgboost/run_hydra.py
import sys
from importlib import import_module

def strip_dashes(argv):
    """--param=val  ->  param=val   (Hydra-friendly)"""
    return [arg.lstrip("-") if arg.startswith("--") else arg for arg in argv]

def main() -> None:                           # entry-point
    sys.argv = [sys.argv[0]] + strip_dashes(sys.argv[1:])
    import_module("hfedxgboost.main").main()  # llama a tu Hydra main()

if __name__ == "__main__":
    main()
