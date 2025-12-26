from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from pathlib import Path

LOGDIR = Path("tb_logs")          # change if needed
OUTDIR = Path("tb_exports")
OUTDIR.mkdir(exist_ok=True)

ea = event_accumulator.EventAccumulator(
    str(LOGDIR),
    size_guidance={
        event_accumulator.SCALARS: 0
    }
)
ea.Reload()

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=tag)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(tag)
    plt.legend()
    plt.grid(True)

    fname = tag.replace("/", "_") + ".jpg"
    plt.savefig(OUTDIR / fname, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved plots to {OUTDIR.resolve()}")
