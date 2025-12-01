import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_tensorboard_scalars(logdir, output_dir):
    """
    Reads TensorBoard logs in `logdir`
    and saves each scalar plot as a PNG in `output_dir`.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load TB event file
    print(f"Loading TensorBoard logs from: {logdir}")
    ea = EventAccumulator(logdir, purge_orphaned_data=True)
    ea.Reload()

    # Get list of all scalar tags
    scalar_tags = ea.Tags()["scalars"]

    print(f"Found {len(scalar_tags)} scalar series:")
    for tag in scalar_tags:
        print("  -", tag)

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)

        outfile = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(outfile)
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