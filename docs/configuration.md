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

## Key sections

| Section | What it controls |
|---------|------------------|
| `hardware` / `acquisition` | channel count, sampling rate, duration, dtype |
| `metadata` | board geometry, condition / substrate labels |
| `quality_control` | spill / unconnected thresholds |
| `preprocessing` | median kernel, baseline span, edge handling |
| `activity_bouts` | RMS window / threshold |
| `sip_detection` | threshold strategy, pairing, duration & amplitude gates |
| `feeding_bursts` | IFI criterion, minimum sips per burst |
| `non_eaters` | per-substrate / global removal thresholds |
| `stats` | permutation count, CI level, multiple-comparison method |
| `plotting` | `vector_format` (pdf/eps/svg/none), dpi, faceting, display-only labels |

Every field, with its type/default/bounds, is in the [API reference](api.md) (the Pydantic
models) and the exported JSON-Schema.
