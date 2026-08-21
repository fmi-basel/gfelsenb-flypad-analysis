# Configuration

flypad has a single, fully-validated source of truth for every parameter. The resolution
order (later wins) is:

1. **Preset** — `matlab_compat` or `corrected` (chosen by `mode:`).
2. **Experiment YAML** — your file (`-c experiment.yaml`).
3. **CLI overrides** — `--set sip_detection.equality_factor=0.4`.

The merged mapping is validated into a Pydantic `Config`; unknown keys, wrong types, or
out-of-range values fail fast.

## Inspect the schema

```bash
flypad config schema            # full JSON-Schema (editor autocomplete)
flypad config show exp.yaml     # the fully-resolved config as YAML
flypad config validate exp.yaml # resolve + report validity
```

## Example experiment file

```yaml
mode: matlab_compat
hardware:
  configuration: "48 flies"
  n_channels: 96
  sampling_rate_hz: 100
acquisition:
  duration_samples: 425391
metadata:
  channels_per_board_position: 8
  substrates: ["10% yeast", "20 mM sucrose"]
  conditions: ["fully fed", "24h wet starved", "44h wet starved", "...", "..."]
alignment:
  enabled: true
output:
  formats: [parquet, csv]
```

## Condition and substrate labels

`metadata.conditions` and `metadata.substrates` name the experiment's groups. Both apply
**positionally** to the discovered channel map:

* `conditions[k - 1]` becomes the `condition_label` of condition number *k* — the numbering
  of the `## conditions` block in the `exp_*.txt` sidecars, or of the `C<cond>` filename
  tokens when there are no sidecars.
* `substrates[s - 1]` becomes the `substrate_label` of substrate *s* (1 = left, 2 = right).

Codes past the end of a list keep the label discovered from the data, so a short list
renames only its leading conditions. Because every table and figure derives its labels from
that one map, the names flow into `per_fly` / `events` / `per_condition` / `comparisons`
(CSV and Parquet alike), the box-plot tick labels, the facet titles, and the raster and
time-course legends.

`plotting.condition_labels` is different: it is an explicit `old: new` mapping applied to
figures **only**, so the exported tables keep their canonical labels. Use it for a display
rename (a poster figure) rather than to name the experiment.

## Faceting

`plotting.facet_by` splits the box plot, CCDF and time course into one axis per
`substrate` (default), per input `file`, or `none` for a single pooled axis. Two options
control how that strip is drawn:

| Option | Default | Effect |
|--------|---------|--------|
| `facet_layout` | `rows` | Facets stacked top-to-bottom; the shared condition axis is drawn once, under the bottom facet. `columns` puts them side by side. |
| `facet_share_y` | `false` | Each facet scales to its own data. Set `true` to tie every facet to one metric scale. |

Independent scales are the default because a substrate the flies barely touched is
otherwise flattened to a line beside one they fed on heavily. Turn `facet_share_y` on
when comparing magnitudes *across* facets matters more than reading the smaller one — the
significance brackets then align across facets too, since they anchor to the tallest.

On the CCDF the metric lives on the x-axis, so `facet_share_y` ties the x-axes there.

## Quality control

Per-channel spill (saturated-sample) and zero-sample fractions are **always** computed and
written to `per_fly` as `spill_fraction` / `zero_fraction`, alongside the `spill`,
`unconnected` and `non_eater` flags — so a removal is always auditable rather than silent.
Two toggles decide whether flagged channels are actually dropped before per-condition
aggregation:

| Option | `matlab_compat` | `corrected` | Effect |
|--------|-----------------|-------------|--------|
| `remove_spill_quality` | `false` | `true` | Drop channels whose saturated fraction exceeds `spill_quality_threshold` (0.5). |
| `remove_unconnected` | `false` | `true` | Drop channels whose zero fraction exceeds `unconnected_zero_fraction` (0.5). |

`matlab_compat` computes but does not auto-remove, matching v2.2. If you switch a dataset
from `matlab_compat` to `corrected`, expect per-condition counts to change: that is the
removal taking effect, and `per_fly` tells you exactly which channels went.

## Arena alignment

Arenas are loaded by hand, minutes apart, so a channel's experiment does not begin when the
recording does. With a manual fill-timestamp sidecar present, `alignment.enabled` (on in
`corrected`) shifts every channel to its own arena start and trims all channels to a common
duration, so time courses and rasters compare like with like.

| Option | Default | Effect |
|--------|---------|--------|
| `alignment.enabled` | `true` (`corrected`) | Align each arena to its fill timestamp. No-op when no sidecar is found. |
| `alignment.window_samples` | `null` | Analysis window per channel; `null` trims to the shortest aligned channel. |

Exported indices stay raw-file sample positions, and the raster shows the unaligned
recording next to the aligned one so you can see what the shift did.

## Key sections

| Section | What it controls |
|---------|------------------|
| `hardware` / `acquisition` | channel count, sampling rate, duration, dtype |
| `metadata` | board geometry, condition / substrate labels |
| `quality_control` | spill / unconnected thresholds and auto-removal toggles |
| `alignment` | per-arena alignment to manual fill timestamps |
| `preprocessing` | median kernel, baseline span, edge handling |
| `activity_bouts` | RMS window / threshold |
| `sip_detection` | threshold strategy, pairing, duration & amplitude gates |
| `feeding_bursts` | IFI criterion, minimum sips per burst |
| `non_eaters` | per-substrate / global removal thresholds |
| `stats` | permutation count, CI level, multiple-comparison method |
| `plotting` | `vector_format` (pdf/eps/svg/none), dpi, faceting, display-only labels |

Every field, with its type/default/bounds, is in the [API reference](api.md) (the Pydantic
models) and the exported JSON-Schema.
