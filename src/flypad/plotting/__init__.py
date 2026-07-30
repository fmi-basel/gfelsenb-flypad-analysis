"""Figures: theme, rasters, box plots, time courses, CDFs, export (design §10, M6).

Presentation layer — imports matplotlib, but the science core never imports this.
"""

from flypad.plotting.boxplots import (
    annotate_significance,
    ci_plot,
    median_iqr_plot,
    my_errorbar,
    plot_spread,
    significance_marker,
    substrate_comparison,
    tilted_boxplot,
)
from flypad.plotting.cdf import ccdf_plot, cdf_plot, standalone_dashboard
from flypad.plotting.export import save_figure
from flypad.plotting.facets import (
    faceted_boxplot,
    faceted_ccdf,
    faceted_dashboard,
    faceted_timecourse,
    resolve_facets,
    substrate_facets,
    with_file_labels,
)
from flypad.plotting.labels import METRIC_LABELS, metric_label, time_axis
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
    "METRIC_LABELS",
    "annotate_significance",
    "ccdf_plot",
    "cdf_plot",
    "ci_plot",
    "condition_palette",
    "cumulative_timecourse_plot",
    "distinguishable_colors",
    "faceted_boxplot",
    "faceted_ccdf",
    "faceted_dashboard",
    "faceted_timecourse",
    "jbfill",
    "median_iqr_plot",
    "metric_label",
    "my_errorbar",
    "plot_spread",
    "raster_plot",
    "resolve_facets",
    "save_figure",
    "set_theme",
    "shaded_lines",
    "shaded_plot",
    "significance_marker",
    "standalone_dashboard",
    "substrate_comparison",
    "substrate_facets",
    "suptitle",
    "theme_context",
    "tight_subplot",
    "tilted_boxplot",
    "time_axis",
    "with_file_labels",
]
