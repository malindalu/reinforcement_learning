import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_tensorboard_scalars(logdir, output_dir):
    """
    Reads TensorBoard logs in `logdir`
    and saves each scalar plot as a PNG in `output_dir`.

    If it finds tags of the form:
        base, base + "_plus1std", base + "_minus1std"
    it will create a *band plot* for that group.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading TensorBoard logs from: {logdir}")
    ea = EventAccumulator(logdir, purge_orphaned_data=True)
    ea.Reload()

    # All scalar tags
    scalar_tags = ea.Tags()["scalars"]
    print(f"Found {len(scalar_tags)} scalar series:")
    for tag in scalar_tags:
        print("  -", tag)

    # Cache all scalar data
    scalar_data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalar_data[tag] = (steps, values)

    used = set()

    # ---- 1) Detect band groups: base, base_plus1std, base_minus1std ----
    band_groups = []
    for tag in scalar_tags:
        if tag.endswith("_rolling100"):
            base = tag
            plus = base + "_plus1std"
            minus = base + "_minus1std"
            if plus in scalar_tags and minus in scalar_tags:
                band_groups.append((base, plus, minus))
                used.update({base, plus, minus})

    # ---- 2) Plot band groups ----
    for base, plus, minus in band_groups:
        steps_base, vals_base = scalar_data[base]
        steps_plus, vals_plus = scalar_data[plus]
        steps_minus, vals_minus = scalar_data[minus]

        # Assume steps are aligned; if you want you can assert they match.
        plt.figure(figsize=(10, 5))
        # band
        plt.fill_between(
            steps_base,
            vals_minus,
            vals_plus,
            alpha=0.2,
            label=f"{base} ± 1 std",
        )
        # mean line
        plt.plot(steps_base, vals_base, label=base)

        plt.title(base + " (with band)")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        outfile = os.path.join(
            output_dir,
            f"{base.replace('/', '_')}_band.png"
        )
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
        print(f"Saved band plot: {outfile}")

    # ---- 3) Plot remaining tags normally ----
    for tag in scalar_tags:
        if tag in used:
            continue  # already covered in a band plot

        steps, values = scalar_data[tag]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)

        outfile = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outfile}")

    print("\nDone! All graphs saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True,
                        help="Path to 'runs/.../' where TensorBoard logs are stored")
    parser.add_argument("--out", type=str, default="tb_plots",
                        help="Output directory for PNG graphs")
    args = parser.parse_args()

    export_tensorboard_scalars(args.logdir, args.out)
