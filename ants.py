"""Utility functions for working with ANTs

ANTsPy (https://github.com/ANTsX/ANTsPy) also provides a Python interface to ANTs.  This
module provides some supplemental utilities.

"""

import os
import re

import ants
import numpy as np
import psutil
from subprocess import CalledProcessError, Popen, PIPE, STDOUT

import nibabel

from pastrycutter.util import itk_directions_from_affine


class AntsError(Exception):
    """Raise if ANTs error detected"""


def antsimage_from_nibabel(nii: nibabel.Nifti1Image):
    """Return ANTsImage converted from a nibabel Nifti1Image"""
    direction, spacing, origin = itk_directions_from_affine(nii.affine)
    antsimage = ants.from_numpy(
        nii.get_fdata(), origin=tuple(origin), spacing=tuple(spacing)
    )
    antsimage.set_direction(direction)
    return antsimage


def env_ants(threads=None):
    """Return environment variables for calling ANTs in a subprocess"""
    if threads is None:
        threads = psutil.cpu_count(logical=False)
    ANTSPATH = os.getenv("ANTSPATH")
    if ANTSPATH is None:
        raise ValueError(f"The ANTSPATH environment variable is not set.")
    return {
        "ANTSPATH": ANTSPATH,
        "PATH": f"{ANTSPATH}{os.pathsep}{os.getenv('PATH')}",
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


###########################################
# Functions to read antsRegistration logs #
###########################################


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


def parse_ants_log(pth_log):
    """Read ANTs log file into structured data

    Reading the ANTs log file is useful for:

    1. Checking if the parameters specified in a script actually propagated to ANTs.

    2. Checking if ANTs converged within the allowed number of iterations.
    """
    d_setup = {}
    stage_lns = []
    with open(pth_log, "r") as f:
        for ln in f:
            if ln.startswith("Use histogram matching"):
                d_setup["match_histograms"] = {"true": True, "false": False}[
                    ln.split(" = ")[1].strip()
                ]
            if ln.startswith("Winsorize image intensities"):
                enabled = {"true": True, "false": False}[ln.split(" = ")[1].strip()]
                q0 = float(f.readline().split(" = ")[1])
                q1 = float(f.readline().split(" = ")[1])
                d_setup["winsorization"] = {"enabled": enabled, "quantiles": (q0, q1)}
            elif ln.startswith("Stage 0"):
                ln = f.readline().strip()
                while ln:
                    stage_lns.append(ln)
                    ln = f.readline().strip()
    stage_params = parse_ants_log_stage_lines(stage_lns)
    return {"setup": d_setup, "stages": [stage_params]}


def parse_ants_log_stage_lines(stage_lines):
    shrink_factor_lns = []

    params = {}
    params["metrics"] = []
    for ln in stage_lines:
        if ln.startswith("iterations"):
            params["max_iterations"] = tuple(
                int(x) for x in ln.split("=")[1].strip().split("x")
            )
        elif ln.startswith("convergence threshold"):
            params["convergence_threshold"] = float(ln.split("=")[1].strip())
        elif ln.startswith("convergence window size"):
            params["convergence_window"] = int(ln.split("=")[1].strip())
        elif ln.startswith("number of levels"):
            params["levels"] = int(ln.split("=")[1].strip())
        elif ln.startswith("Using") and "metricSamplingStrategy" in ln:
            m = re.match(r"Using (?P<strategy>.+) metricSamplingStrategy", ln)
            params["sampling"] = {"default NONE": None}[m.group("strategy")]
        elif ln.startswith("Shrink factors"):
            shrink_factor_lns.append(ln)
        elif ln.startswith("smoothing sigmas"):
            params["smoothing"] = tuple(
                int(i) for i in ln.split(": [")[1].removesuffix("]").split(",")
            )
        elif ln.startswith("using") and "metric" in ln:
            m = re.match(r"using the (?P<metric>.+) metric \((?P<params>.+)\)", ln)
            p = dict(eq.strip().split(" = ") for eq in m.group("params").split(","))
            d_metric = {
                "metric": {
                    "Mattes MI": "MI",
                    "CC": "CC",
                    "global correlation": "GC",
                    "MeanSquares": "MSQ",
                }[m.group("metric")],
                "weight": float(p["weight"]),
                "filter_gradient": {"1": True, "0": False}[p["use gradient filter"]],
            }
            if d_metric["metric"] in ("CC", "MI"):
                ants_k = {"CC": "radius", "MI": "number of bins"}[d_metric["metric"]]
                k = {"CC": "radius", "MI": "bins"}[d_metric["metric"]]
                d_metric[k] = int(p[ants_k])
            params["metrics"].append(d_metric)
    params["shrink_factors"] = parse_ants_log_shrink_factor_lines(shrink_factor_lns)
    assert len(params["shrink_factors"]) == params["levels"]
    assert len(params["smoothing"]) == params["levels"]
    return params


def parse_ants_log_shrink_factor_lines(shrink_lines):
    """Parse shrink factor lines within a stage

    E.g.,

    Shrink factors (level 1 out of 2): [2, 2, 2]
    Shrink factors (level 2 out of 2): [1, 1, 1]
    """
    shrinks = []
    for ln in shrink_lines:
        m = re.match(
            r"Shrink factors \(level (\d+) out of \d+\): \[(\d+), (\d+), (\d+)\]", ln
        )
        # group 0 is whole match
        shrinks.append((int(m.group(1)), tuple(int(m.group(i)) for i in (2, 3, 4))))
    shrinks.sort(key=lambda x: x[0])
    return tuple(t[1] for t in shrinks)
