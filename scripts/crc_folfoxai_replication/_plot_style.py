"""Shared Nature-style plotting configuration for the FOLFOXai replication.

Typography, line weights, spines, and palette follow Nature's figure
guidelines:
  - sans-serif fonts (Arial / Helvetica)
  - 7-8 pt labels, 0.5 pt spines, no top/right spines
  - single-column width 89 mm (3.5 in); double-column 183 mm (7.2 in)
  - colorblind-safe tertile palette
"""
import matplotlib as mpl


FIG_W_IN = 3.5
FIG_H_IN = 3.5
DPI = 600
TERTILE_COLORS = ['#2ca02c', '#1f77b4', '#d62728']


def apply_nature_style():
    mpl.rcParams.update({
        'font.family':        ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          8,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'axes.linewidth':     0.5,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'xtick.major.size':   2.0,
        'ytick.major.size':   2.0,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'legend.frameon':     False,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'savefig.dpi':        DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
    })


def save_panel(fig, out_path_no_ext, save_pdf=True):
    """Save a figure at DPI=600 PNG and optionally a vector PDF copy."""
    fig.savefig(f'{out_path_no_ext}.png', dpi=DPI)
    if save_pdf:
        fig.savefig(f'{out_path_no_ext}.pdf')
