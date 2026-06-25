"""Experiment metadata sidecars -> channel->condition map (design §17, M4).

The legacy MATLAB pipeline could only analyse a *single* condition, so recordings
were stored with ``Events.Condition`` collapsed to ``1`` and the real multi-condition
design was tracked externally in ``exp_*.txt`` board-layout sidecars. ``flypad`` must
therefore rebuild the channel->condition map from those sidecars rather than trusting
the ``.mat``.

Board layout
------------
Board position ``p`` (1-based) owns channels ``(p-1)*S .. (p-1)*S + S-1`` where
``S = channels_per_board_position`` (8). Within an arena the *even* channel is the
left substrate (substrate 1) and the *odd* channel the right substrate (substrate 2).

Sources
-------
``exp_<i>.txt``  board position -> condition (short label + sex), per recording file.
``LOG.txt``      ``Events.SubstrateLabel{k}`` strings (channel -> substrate label).
filenames        recording order / ``file_name`` column.

The canonical fixture ``data/sample/20240215/channel_condition_map.csv`` (192 rows,
2 files x 96 channels) is the regression target reproduced by
:func:`build_channel_condition_map`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flypad.io.discovery import CAP_PREFIX, find_capacitance_files

_EXP_INDEX_RE = re.compile(r"exp_(\d+)", re.IGNORECASE)
_MAT_LABEL_RE = re.compile(r"Events\.(\w+)\{(\d+)\}\s*=\s*'([^']*)'")

#: Tidy-table column order (matches the canonical CSV fixture).
MAP_COLUMNS = (
    "file_index",
    "file_name",
    "exp_file",
    "board_position",
    "channel",
    "condition",
    "condition_label",
    "condition_short",
    "sex",
    "substrate",
    "substrate_side",
    "substrate_label",
)


@dataclass(frozen=True)
class BoardEntry:
    """One ``## log`` line: a board position's condition short label and sex."""

    position: int
    condition_short: str
    sex: str


@dataclass(frozen=True)
class ExpSidecar:
    """Parsed ``exp_<i>.txt`` sidecar."""

    conditions: dict[int, str]  # condition number -> long label (## conditions)
    substrate_labels: dict[str, str]  # 'left'/'right' -> label (## substrates)
    board: tuple[BoardEntry, ...]  # ## log, in file order

    def condition_order(self) -> dict[str, int]:
        """Map each short label to a condition number by first-appearance order.

        The ``## conditions`` block is numbered ``1..N`` in the same order the short
        labels first appear in the ``## log`` block, so the *k*-th distinct short maps
        to condition *k* (and thus to ``conditions[k]``).
        """
        order: dict[str, int] = {}
        for entry in sorted(self.board, key=lambda e: e.position):
            if entry.condition_short not in order:
                order[entry.condition_short] = len(order) + 1
        return order

    def board_by_position(self) -> dict[int, BoardEntry]:
        return {e.position: e for e in self.board}


@dataclass(frozen=True)
class LogLabels:
    """Substrate / condition labels parsed from a MATLAB-style ``LOG.txt``."""

    condition_labels: dict[int, str]  # Events.ConditionLabel{k}
    substrate_labels: dict[int, str]  # Events.SubstrateLabel{k} (1-based)


def _split_kv(line: str, sep: str) -> tuple[str, str] | None:
    """Split ``line`` on the first ``sep`` and strip both halves (or ``None``)."""
    if sep not in line:
        return None
    key, _, value = line.partition(sep)
    return key.strip(), value.strip()


def parse_exp_file(path: str | Path) -> ExpSidecar:
    """Parse an ``exp_<i>.txt`` board-layout sidecar.

    The format is a small Markdown-ish document with ``## conditions`` (``- N:label``),
    ``## substrates`` (``left:`` / ``right:``) and ``## log`` (``pos: short,sex``)
    sections. Parsing is whitespace-tolerant (``8 : 44h refed, F`` is accepted).
    """
    conditions: dict[int, str] = {}
    substrate_labels: dict[str, str] = {}
    board: list[BoardEntry] = []
    section: str | None = None

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            section = line.lstrip("#").strip().lower()
            continue
        if line.startswith("#"):  # document title (single '#')
            continue

        if section == "conditions":
            kv = _split_kv(line.lstrip("-").strip(), ":")
            if kv and kv[0].isdigit():
                conditions[int(kv[0])] = kv[1]
        elif section == "substrates":
            kv = _split_kv(line, ":")
            if kv and kv[0].lower() in ("left", "right"):
                substrate_labels[kv[0].lower()] = kv[1]
        elif section == "log":
            kv = _split_kv(line, ":")
            if not kv or not kv[0].isdigit():
                continue
            short, _, sex = kv[1].partition(",")
            board.append(BoardEntry(int(kv[0]), short.strip(), sex.strip()))

    return ExpSidecar(
        conditions=conditions,
        substrate_labels=substrate_labels,
        board=tuple(board),
    )


def parse_log_file(path: str | Path) -> LogLabels:
    """Parse ``Events.ConditionLabel{}`` / ``Events.SubstrateLabel{}`` from ``LOG.txt``."""
    condition_labels: dict[int, str] = {}
    substrate_labels: dict[int, str] = {}
    for field, idx, value in _MAT_LABEL_RE.findall(Path(path).read_text(encoding="utf-8")):
        if field == "ConditionLabel":
            condition_labels[int(idx)] = value
        elif field == "SubstrateLabel":
            substrate_labels[int(idx)] = value
    return LogLabels(condition_labels=condition_labels, substrate_labels=substrate_labels)


def build_channel_condition_map(
    file_specs: Sequence[tuple[str, str | Path]],
    *,
    log_file: str | Path | None = None,
    n_channels: int = 96,
    channels_per_board_position: int = 8,
) -> pd.DataFrame:
    """Build the tidy channel->condition map for a set of recordings.

    Parameters
    ----------
    file_specs : ordered ``(capacitance_file_name, exp_file_path)`` pairs. The position
        in the sequence becomes ``file_index``.
    log_file : optional ``LOG.txt`` providing substrate labels. When absent the long
        labels from each sidecar's ``## substrates`` block are used instead.
    n_channels, channels_per_board_position : board geometry.

    Returns
    -------
    A :class:`pandas.DataFrame` with the columns in :data:`MAP_COLUMNS`, one row per
    ``(file, channel)``.
    """
    log = parse_log_file(log_file) if log_file is not None else None
    rows: list[dict[str, object]] = []

    for file_index, (file_name, exp_path) in enumerate(file_specs):
        exp = parse_exp_file(exp_path)
        short_to_condition = exp.condition_order()
        board = exp.board_by_position()
        exp_name = Path(exp_path).name

        for channel in range(n_channels):
            position = channel // channels_per_board_position + 1
            entry = board[position]
            condition = short_to_condition[entry.condition_short]
            is_left = channel % 2 == 0
            substrate = 1 if is_left else 2
            side = "left" if is_left else "right"
            if log is not None and substrate in log.substrate_labels:
                substrate_label = log.substrate_labels[substrate]
            else:
                substrate_label = exp.substrate_labels.get(side, "")

            rows.append(
                {
                    "file_index": file_index,
                    "file_name": file_name,
                    "exp_file": exp_name,
                    "board_position": position,
                    "channel": channel,
                    "condition": condition,
                    "condition_label": exp.conditions[condition],
                    "condition_short": entry.condition_short,
                    "sex": entry.sex,
                    "substrate": substrate,
                    "substrate_side": side,
                    "substrate_label": substrate_label,
                }
            )

    frame = pd.DataFrame(rows, columns=list(MAP_COLUMNS))
    int_cols = ["file_index", "board_position", "channel", "condition", "substrate"]
    frame[int_cols] = frame[int_cols].astype("int64")
    return frame


def _discover_exp_files(directory: Path) -> list[Path]:
    """Return ``exp_<i>.txt`` sidecars ordered by their integer index."""

    def index(path: Path) -> int:
        match = _EXP_INDEX_RE.search(path.name)
        return int(match.group(1)) if match else 0

    return sorted(directory.glob("exp_*.txt"), key=index)


def channel_condition_map_for_dir(
    data_dir: str | Path,
    *,
    n_channels: int = 96,
    channels_per_board_position: int = 8,
    log_name: str = "LOG.txt",
) -> pd.DataFrame:
    """Discover sidecars + recordings in ``data_dir`` and build the channel map.

    Pairs the *i*-th capacitance file (sorted by name) with ``exp_<i>.txt`` and uses
    ``LOG.txt`` for substrate labels when present.
    """
    directory = Path(data_dir)
    cap_files = find_capacitance_files(directory)
    exp_files = _discover_exp_files(directory)
    if not exp_files:
        raise FileNotFoundError(f"no exp_*.txt sidecars in {directory}")
    if cap_files and len(cap_files) != len(exp_files):
        raise ValueError(
            f"{len(cap_files)} {CAP_PREFIX}* files but {len(exp_files)} exp_*.txt sidecars"
        )

    names = (
        [p.name for p in cap_files] if cap_files else [f"file_{i}" for i in range(len(exp_files))]
    )
    specs = list(zip(names, exp_files, strict=True))
    log_path = directory / log_name
    return build_channel_condition_map(
        specs,
        log_file=log_path if log_path.exists() else None,
        n_channels=n_channels,
        channels_per_board_position=channels_per_board_position,
    )
