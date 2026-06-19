#!/bin/bash
#
# Download reference genome(s) for TESSERA into this script's own directory.
#
# This is a thin wrapper around `python -m tessera.ref_genome`, which is the
# single source of truth for the genome filenames and NCBI URLs (see
# tessera/ref_genome.py). Using TESSERA at all triggers the same download
# lazily on first featurisation; run this only to pre-provision (e.g. on an
# offline/shared machine, or to avoid first-call latency).
#
# Usage:
#   ./download_ref_genomes.sh [grch37 | grch38 | both]   (default: grch37)
#
# Files are written to this script's directory. An existing, complete FASTA is
# not re-downloaded. The FASTA index (.fai) is created by pyfaidx on first use.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

case "${1:-grch37}" in
    -h|--help|help)
        sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

exec python3 -m tessera.ref_genome "${1:-grch37}" --dir "$SCRIPT_DIR"
