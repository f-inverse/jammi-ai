"""``python -m jammi.session_journal -- <command>`` — run a command under the
session journal. Kept apart from the journal itself, which `import jammi`
loads: a module a package imports cannot also be that package's ``-m`` entry.
"""

import sys

from ._session_journal import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
