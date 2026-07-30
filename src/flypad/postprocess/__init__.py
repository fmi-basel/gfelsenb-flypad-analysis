"""Post-processing: feeding bursts, QC removal, transitions, metadata (design §5, M4)."""

from flypad.postprocess.alignment import (
    align_channels,
    arena_start_samples,
    available_duration,
    channel_start_samples,
)
from flypad.postprocess.bursts import (
    ChannelBursts,
    detect_feeding_bursts,
    feeding_ifi_threshold,
    group_feeding_bursts,
    mode_int,
    pooled_ifi_threshold,
)
from flypad.postprocess.metadata import (
    MAP_COLUMNS,
    BoardEntry,
    ExpSidecar,
    LogLabels,
    apply_label_overrides,
    build_channel_condition_map,
    channel_condition_map_for_dir,
    channel_map_from_filenames,
    parse_exp_file,
    parse_log_file,
)
from flypad.postprocess.quality import (
    QualityResult,
    assess_quality,
    flag_non_eaters,
    flag_spill_channels,
    flag_unconnected_channels,
    saturation_fraction,
    zero_fraction,
)
from flypad.postprocess.transitions import (
    ArenaTransitions,
    channel_transitions,
    classify_in_burst,
)

__all__ = [
    "MAP_COLUMNS",
    "ArenaTransitions",
    "BoardEntry",
    "ChannelBursts",
    "ExpSidecar",
    "LogLabels",
    "QualityResult",
    "align_channels",
    "apply_label_overrides",
    "arena_start_samples",
    "assess_quality",
    "available_duration",
    "build_channel_condition_map",
    "channel_condition_map_for_dir",
    "channel_map_from_filenames",
    "channel_start_samples",
    "channel_transitions",
    "classify_in_burst",
    "detect_feeding_bursts",
    "feeding_ifi_threshold",
    "flag_non_eaters",
    "flag_spill_channels",
    "flag_unconnected_channels",
    "group_feeding_bursts",
    "mode_int",
    "parse_exp_file",
    "parse_log_file",
    "pooled_ifi_threshold",
    "saturation_fraction",
    "zero_fraction",
]
