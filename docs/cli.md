# CLI reference

Every command takes `-c/--config`, `--mode matlab_compat|corrected`, and repeatable
`--set key=value` overrides (later wins: preset → YAML → `--set`).

| Command | Purpose |
|---------|---------|
| `flypad run DATA_DIR -o OUT` | Full pipeline → tables + figures + provenance |
| `flypad detect DATA_DIR -o OUT` | Detection only → `events` + `per_fly` tables |
| `flypad stats RESULTS_DIR` | Recompute per-condition summaries from saved tables |
| `flypad plot RESULTS_DIR --kind … --metric …` | Render figures from saved tables |
| `flypad validate DATA_DIR --against GT.mat` | Score detections vs a MATLAB `Events.mat` |
| `flypad config show/validate/schema` | Inspect / validate / export config |
| `flypad gui` | Launch the desktop GUI (needs the `gui` extra) |
| `flypad version` | Print the installed version |

## Examples

```bash
# full run, corrected mode, override a parameter, skip figures
flypad run DATA_DIR -c exp.yaml --mode corrected \
    --set sip_detection.equality_factor=0.4 --no-plots -o results/

# figures only, a specific metric and kinds
flypad plot results/ --kind dashboard,boxplot,timecourse --metric n_feeding_bursts

# resolve and print the effective config
flypad config show exp.yaml --set hardware.n_channels=96
```

## Figure kinds (`--kind`)

`dashboard`, `boxplot`, `cdf`, `raster`, `timecourse`, `substrate` — comma-separated.
Outputs go to `RESULTS_DIR/figures/` as PNG plus the configured vector format (PDF by
default; set `plotting.vector_format` to `eps`/`svg`/`none`).
