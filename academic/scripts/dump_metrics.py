import json
from pathlib import Path

rows = json.loads(Path("results/run/metrics.json").read_text(encoding="utf-8"))
for row in rows:
    print(
        f"{row['agent']:20} {row['domain']:14} "
        f"{row['success_rate']:.3f} {row['success_ci_low']:.3f}-{row['success_ci_high']:.3f} "
        f"c={row['mean_cost']:.2f} mae={row['mean_prediction_mae']:.3f} "
        f"brier={row['mean_terminal_brier']:.3f} fs={row['mean_foresight_calls']:.2f}"
    )
