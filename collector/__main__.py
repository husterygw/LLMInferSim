"""Allow `python -m collector ...` as alias for `python -m collector.cli ...`."""
import sys

from collector.cli import main

if __name__ == "__main__":
    sys.exit(main())
