#!/bin/bash
# Every path the CLARA cluster harness needs, resolved once, from THIS FILE's own location.
#
# Source it as the first thing a launcher does, BEFORE any `cd`. Scripts that are RUN (interactively
# or under srun) can source it directly:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"    # from scripts/clara/interactive/*
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/paths.sh"       # from scripts/clara/*
#
# Scripts SUBMITTED with sbatch cannot -- see the sbatch block below -- and carry a short locator
# instead. Copy it from any of the five sbatch_*/run_og391_* launchers rather than reinventing it.
#
# Sets and exports:
#   REALM_ROOT         this checkout's root, derived from this file's location
#   REALM_SHARED       shared, NOT-in-repo asset store: image, dataset, log tree, stock_patch
#   REALM_OGLITE_ROOT  the OG-lite fork for 3.9.1, bound over the image's own OmniGibson
#   REALM_SIF          the og391 Apptainer image
#   REALM_DATA         dataset root                    -> bound as /data
#   REALM_APPDATA      OMNIGIBSON_APPDATA_PATH backing -> bound as /cache
#   REALM_LOGS         results and logs                -> bound as /logs
#   REALM_STOCK_PATCH  patched stock OmniGibson files, for `MODE=stockfix rr`
#
# and defines realm_paths_show(), which prints what everything resolved to plus whether it exists.
# Run it when a path surprises you: `bash -c 'source scripts/clara/lib/paths.sh; realm_paths_show'`.
#
# EVERY caller must assert the canary after sourcing:
#
#   [ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: failed to source .../lib/paths.sh" >&2; exit 1; }
#
# because a failed `source` is NOT fatal on its own (set -e is off everywhere in this harness) and
# what a caller would then read out of $REALM_ROOT is the shell profile's value -- the pre-port
# 1.1.1 checkout. See the next block.
#
# SBATCH CALLERS CANNOT USE ${BASH_SOURCE[0]} to find this file. sbatch ships the script's TEXT to
# the node, which writes it to /var/spool/slurmd/job<N>/slurm_script and runs that copy. Verified
# 2026-08-14 with probe job 191043:
#
#     BASH_SOURCE[0]   = /var/spool/slurmd/job191043/slurm_script
#     SLURM_SUBMIT_DIR = <the cwd sbatch was called from>
#     scontrol Command = <the absolute path sbatch was handed>
#
# So the five sbatch_*/run_* launchers try BASH_SOURCE first (right for `bash <path>` and for srun),
# then `scontrol show job`, then $SLURM_SUBMIT_DIR, testing each candidate before using it.
#
# ==================================================================================================
# WHY NOTHING BELOW READS AN EXISTING $REALM_* VALUE
#
# The shell profile on this machine exports, from a block in ~/.bashrc that REALM's own setup.sh
# wrote and still manages:
#
#     REALM_ROOT=/home/sedlam56/projects/REALM/          <- the PRE-PORT OmniGibson 1.1.1 checkout
#     REALM_SIF=/home/sedlam56/apptainer/realm-dm.sif    <- the PRE-PORT 1.1.1 image
#     REALM_DATA_PATH=/home/sedlam56/projects/REALM/data/
#     REALM_LOGS=/home/sedlam56/projects/REALM/logs/
#
# So a `${REALM_ROOT:-<og391 default>}` written in an og391 script NEVER REACHES ITS DEFAULT -- it
# resolves to the 1.1.1 tree, and `${REALM_SIF:-...}` to the 1.1.1 image, where OmniGibson lives at
# /omnigibson-src instead of /behavior-src. That has already cost real time twice: once loudly
# ("rr: No such file or directory"), once as a failure that read as "the patch no longer applies"
# when it was actually the wrong container. The silent form is worse than either -- an og391 script
# quietly evaluating the old stack and reporting the numbers as the port's.
#
# So every name here is ASSIGNED, never defaulted-through. Overrides do exist, but only under names
# the profile does not manage: the *_OG391 suffix, following make_stock_patch.sh's REALM_SIF_OG391.
# If you add a path, keep that rule. `${REALM_ANYTHING:-...}` here is a trap, not a feature.
#
# REALM_DATA_PATH is deliberately NOT touched: scripts/clara/lib/apptainer.sh (the pre-3.9.1 eval
# drivers) reads it from the profile. og391 scripts use $REALM_DATA / $REALM_APPDATA instead.
# ==================================================================================================

#--- this checkout -------------------------------------------------------------------------------

# THIS FILE's location, not $0 and not the caller's cwd: a launcher can be invoked by any path from
# any directory, and a sourced file sees its own path in BASH_SOURCE[0]. This file is at
# <root>/scripts/clara/lib/paths.sh, so the root is three levels up.
#
# Deriving it is also what makes a git worktree actually isolated. A hardcoded root made a
# worktree's launcher bind the MAIN checkout at /app, so edits to realm/ in the worktree had no
# effect on the run while edits to the launchers did -- measured 2026-08-13, two agents' fixes were
# tested against code they had not edited.
REALM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

#--- the shared asset store ----------------------------------------------------------------------

# The 13 GB image, the 36 GB behavior-1k dataset, the log tree every REALM eval writes into, and the
# stock_patch dir. None of it is in this repo and none of it is duplicated per checkout.
#
# This is the one absolute path the harness still needs, and it is stated here ONCE instead of in
# fifteen launchers. It already names ~/projects/REALM -- both where those artifacts live today AND
# the destination of the pending REALM_og391 -> REALM rename -- so the rename does not invalidate it.
# Override with REALM_SHARED_OG391= (never REALM_ROOT=; see the header).
REALM_SHARED=${REALM_SHARED_OG391:-/mnt/home_lustre/sedlam56/projects/REALM}

# First candidate that exists, else the first candidate unchanged so that the CALLER's own
# `[ -f "$REALM_SIF" ] || exit 1` reports the path a reader would expect to see rather than the last
# thing tried. Never fails: resolution is not the place to exit, the caller's check is.
_realm_pick() {
  local flag=$1; shift
  local c
  for c in "$@"; do
    if [ -n "$c" ] && [ "$flag" "$c" ]; then printf '%s\n' "$c"; return 0; fi
  done
  printf '%s\n' "$1"
}

# In-repo first, shared store second. The repo exposes the big artifacts as untracked symlinks
# (realm_og391.sif -> shared, data/datasets -> shared, logs -> shared) precisely so the in-repo path
# is the normal answer, and after the rename they are real entries at the root. A fresh worktree has
# only some of those symlinks, which is what the fallback covers.
REALM_SIF=${REALM_SIF_OG391:-$(_realm_pick -f "$REALM_ROOT/realm_og391.sif" "$REALM_SHARED/realm_og391.sif")}
REALM_DATA=${REALM_DATA_OG391:-$(_realm_pick -d "$REALM_ROOT/data/datasets" "$REALM_SHARED/data/datasets_og391")}
# NOTE for whoever performs the rename: this checkout's `logs` is a symlink to $REALM_SHARED/logs,
# so once the tree IS ~/projects/REALM that symlink points at itself and neither candidate resolves.
# Delete the symlink then (the real log tree stays where it is) -- $REALM_ROOT/logs becomes the real
# directory and the first candidate wins. A loop makes the caller's `[ -d ]` check fail loudly,
# which is the intended outcome; nothing here can silently pick a wrong log tree.
REALM_LOGS=${REALM_LOGS_OG391:-$(_realm_pick -d "$REALM_ROOT/logs" "$REALM_SHARED/logs")}
REALM_STOCK_PATCH=${REALM_STOCK_PATCH_OG391:-$REALM_SHARED/stock_patch}

# Per-checkout, not shared: the Kit/USD shader caches this backs are written by the running sim and
# two concurrent jobs sharing one appdata dir corrupt each other's.
REALM_APPDATA=${REALM_APPDATA_OG391:-$REALM_ROOT/data/cache}

#--- the OG-lite fork ----------------------------------------------------------------------------

# Sibling of the checkout, then sibling of the shared store. The second candidate is what a worktree
# needs: worktrees live under projects/wt/<name>, whose sibling is another worktree, not OG-lite.
# OG-lite is one SHARED checkout on purpose -- it is bound read-mostly into every run and is not
# duplicated per worktree -- so resolving it relative to the shared store is the correct answer
# there, and the sibling candidate keeps it right for the main checkout and after the rename.
REALM_OGLITE_ROOT=${REALM_OGLITE_OG391:-$(_realm_pick -d \
  "$(dirname "$REALM_ROOT")/OG-lite_og391" "$(dirname "$REALM_SHARED")/OG-lite_og391")}

export REALM_ROOT REALM_SHARED REALM_OGLITE_ROOT REALM_SIF REALM_DATA REALM_APPDATA \
       REALM_LOGS REALM_STOCK_PATCH

#--- debugging -----------------------------------------------------------------------------------

realm_paths_show() {
  local v p
  printf '%-18s %s\n' "(cwd)" "$PWD"
  for v in REALM_ROOT REALM_SHARED REALM_OGLITE_ROOT REALM_SIF REALM_DATA REALM_APPDATA \
           REALM_LOGS REALM_STOCK_PATCH; do
    p=${!v}
    printf '%-18s %-64s %s\n' "$v" "$p" "$([ -e "$p" ] && echo ok || echo MISSING)"
  done
}

# The canary every caller checks. Set LAST so it cannot be true while resolution above was skipped,
# and deliberately NOT exported: an exported canary would be inherited by a child launcher (t*.sh ->
# rr, t9_sweep.sh -> rr) and would then read as "sourced fine" in a child where the source failed.
# Unexported, it is visible to the script that sourced this file and to nothing else.
REALM_PATHS_SH=1
