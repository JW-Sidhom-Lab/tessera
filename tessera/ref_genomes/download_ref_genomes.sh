#!/bin/bash
#
# Download reference genome(s) for TESSERA.
#
# Usage:
#   ./download_ref_genomes.sh [grch37 | grch38 | both]
#
# Defaults to grch37 if no argument is given.
# Files are written to the script's own directory.
# If a FASTA already exists, it is not re-downloaded.
# A FASTA index (.fai) is created via samtools if available; otherwise
# pyfaidx will create one on first access.
#
# Examples:
#   ./download_ref_genomes.sh                # GRCh37 only (TESSERA default)
#   ./download_ref_genomes.sh grch38         # GRCh38 only
#   ./download_ref_genomes.sh both           # both genomes (~6 GB total)

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve script directory so the script works regardless of cwd
# -----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Genome version registry
# -----------------------------------------------------------------------------
GRCH37_FILE="GCF_000001405.25_GRCh37.p13_genomic.fna"
GRCH37_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/GCF_000001405.25_GRCh37.p13_genomic.fna.gz"

GRCH38_FILE="GCF_000001405.40_GRCh38.p14_genomic.fna"
GRCH38_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"

# -----------------------------------------------------------------------------
# Parse argument
# -----------------------------------------------------------------------------
TARGET="${1:-grch37}"

case "$TARGET" in
    grch37|GRCh37|GRCH37)
        TARGETS=(grch37)
        ;;
    grch38|GRCh38|GRCH38)
        TARGETS=(grch38)
        ;;
    both|all)
        TARGETS=(grch37 grch38)
        ;;
    -h|--help|help)
        sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "Error: unknown target '$TARGET'." >&2
        echo "Valid options: grch37 | grch38 | both" >&2
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Helper: download + (best-effort) index a single genome
# -----------------------------------------------------------------------------
download_genome() {
    local label="$1"
    local fasta="$2"
    local url="$3"

    if [ -f "$fasta" ]; then
        echo "[$label] FASTA already present at $fasta — skipping download."
    else
        echo "[$label] Downloading from $url ..."
        wget --show-progress -O "${fasta}.gz" "$url"
        echo "[$label] Extracting ..."
        gunzip "${fasta}.gz"
    fi

    if command -v samtools &> /dev/null; then
        if [ ! -f "${fasta}.fai" ]; then
            echo "[$label] Creating FASTA index with samtools ..."
            samtools faidx "$fasta"
        else
            echo "[$label] Index already present at ${fasta}.fai — skipping."
        fi
    else
        echo "[$label] samtools not found; skipping index creation."
        echo "[$label]   pyfaidx will create the index on first access (slower)."
    fi
}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
echo "TESSERA reference-genome setup"
echo "  target(s): ${TARGETS[*]}"
echo "  output dir: $SCRIPT_DIR"
echo

for t in "${TARGETS[@]}"; do
    case "$t" in
        grch37) download_genome "GRCh37" "$GRCH37_FILE" "$GRCH37_URL" ;;
        grch38) download_genome "GRCh38" "$GRCH38_FILE" "$GRCH38_URL" ;;
    esac
done

echo
echo "Done."
