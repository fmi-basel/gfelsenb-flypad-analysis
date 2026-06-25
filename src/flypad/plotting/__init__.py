"""Figures: theme, rasters, box plots, time courses, CDFs, export (design §10, M6).

Presentation layer — imports matplotlib, but the science core never imports this.
"""

from flypad.plotting.boxplots import (
    ci_plot,
    median_iqr_plot,
    my_errorbar,
    plot_spread,
    substrate_comparison,
    tilted_boxplot,
)
from flypad.plotting.cdf import ccdf_plot, cdf_plot, standalone_dashboard
from flypad.plotting.export import save_figure
from flypad.plotting.rasters import raster_plot
from flypad.plotting.theme import (
    condition_palette,
    distinguishable_colors,
    set_theme,
    suptitle,
    theme_context,
    tight_subplot,
)
from flypad.plotting.timecourses import (
    cumulative_timecourse_plot,
    jbfill,
    shaded_lines,
    shaded_plot,
)

__all__ = [
    "ccdf_plot",
    "cdf_plot",
    "ci_plot",
    "condition_palette",
    "cumulative_timecourse_plot",
    "distinguishable_colors",
    "jbfill",
    "median_iqr_plot",
    "my_errorbar",
    "plot_spread",
    "raster_plot",
    "save_figure",
    "set_theme",
    "shaded_lines",
    "shaded_plot",
    "standalone_dashboard",
    "substrate_comparison",
    "suptitle",
    "theme_context",
    "tight_subplot",
    "tilted_boxplot",
]
