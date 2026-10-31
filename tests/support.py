"""Shared test plumbing for the shell scripts.

scripts/deploy.sh and scripts/fetch_model.sh are POSIX shell, and the tests
drive them by putting stub tools on PATH. That only works with a shell that
understands a Windows PATH: on this platform `bash` commonly resolves to WSL's,
which sees none of the paths the tests hand it and writes its output as UTF-16,
so every assertion fails for reasons unrelated to the script.

Nothing here skips a test. It finds the shell the scripts were written for, and
the callers skip only when no such shell is installed.
"""

import os
import shutil
import subprocess
from pathlib import Path


def posix_bash():
    """Path to a bash that can see Windows paths, or None if there is none."""
    candidates = []
    override = os.environ.get("GIT_BASH")
    if override:
        candidates.append(Path(override))
    git = shutil.which("git")
    if git:
        # <prefix>/cmd/git.exe or <prefix>/bin/git -> <prefix>/bin/bash.exe
        candidates.append(Path(git).parent / "bash.exe")
        candidates.append(Path(git).parent.parent / "bin" / "bash.exe")
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var]) / "Git" / "bin" / "bash.exe")
            candidates.append(Path(os.environ[var]) / "Git" / "usr" / "bin" / "bash.exe")
    for c in candidates:
        if c.is_file():
            return str(c)
    if os.name != "nt":
        return shutil.which("bash")
    return None

def run_script(shell, stub_bin, script, *args, env, cwd=None):
    """Run a shell script with `stub_bin` ahead of everything on PATH.

    Prepending it from the calling process is not enough on Windows: the PATH
    reaches bash already converted, with the shell's own /usr/bin in front, so
    a stub named mv or python3 shadows nothing and the real tool runs instead.
    cygpath produces the Windows directory in the form bash itself uses, and
    the prelude puts it first inside the shell.
    """
    prelude = 'export PATH="$(cygpath -u "$1")":$PATH; shift; exec "$@"'
    return subprocess.run([shell, "-c", prelude, "harness",
                           str(stub_bin), str(script), *args],
                          cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")

