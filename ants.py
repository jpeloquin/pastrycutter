"""Utility functions for working with ANTs

ANTsPy (https://github.com/ANTsX/ANTsPy) also provides a Python interface to ANTs.  This
module provides some supplemental utilities.

"""
import os
import psutil
from subprocess import CalledProcessError, Popen, PIPE, STDOUT


class AntsError(Exception):
    """Raise if ANTs error detected"""


def check_ants_log(pth_log):
    """Check ANTs log for errors

    ANTs sometimes catches and ignores ITK errors, which is a problem.  To be safe, use
    this function to check a log file consisting of redirected ANTs output.  (ANTs does
    not produce a log file by default.)

    """
    with open(pth_log, "r") as f:
        for ln in f:
            if ln.startswith("Exception caught"):
                while True:
                    ln = f.readlines(1)[0]
                    if not ln.strip():
                        break
                    if ln.startswith("Description: "):
                        return ln.rstrip().removeprefix("Description: ")
    return None


def env_ants(threads=None):
    """Return environment variables for calling ANTs in a subprocess"""
    if threads is None:
        threads = psutil.cpu_count(logical=False)
    ANTSPATH = os.getenv("ANTSPATH")
    if ANTSPATH is None:
        raise ValueError(f"The ANTSPATH environment variable is not set.")
    return {
        "ANTSPATH": ANTSPATH,
        "PATH": f"{ANTSPATH}:{os.getenv('PATH')}",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads),
    }


def run_ants(cmd, cwd, pth_log, threads=None):
    output = []
    with Popen(
        cmd,
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        text=True,
        cwd=cwd,
        env=env_ants(threads),
    ) as p, open(pth_log, "w") as logfile:
        for ln in p.stdout:
            logfile.write(ln)
            logfile.flush()
            output.append(ln)
    p.stdout.close()
    returncode = p.wait()
    if returncode != 0:
        raise CalledProcessError(
            returncode,
            cmd,
            "".join(output),
        )
    # Check ANTs log for exceptions; sometimes ANTs returns 0 even with an error
    error = check_ants_log(pth_log)
    if error:
        raise AntsError(error)
