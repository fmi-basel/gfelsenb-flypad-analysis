"""Detection: activity bouts and sips (design §5.5-§5.7)."""

from flypad.detect.bouts import extract_bouts
from flypad.detect.results import ChannelBouts, ChannelSips, DetectResult
from flypad.detect.run import detect_recording
from flypad.detect.sips import detect_sips

__all__ = [
    "ChannelBouts",
    "ChannelSips",
    "DetectResult",
    "detect_recording",
    "detect_sips",
    "extract_bouts",
]
