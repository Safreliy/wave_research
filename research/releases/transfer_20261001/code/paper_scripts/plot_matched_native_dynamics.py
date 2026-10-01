"""Plot frozen full-step matched-transfer errors against exact-input runs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    plt.rcParams.update({"font.size": 9})
    provenance = {}
    for label, colour in (("L9", "#1261a0"), ("L10", "#b54d13")):
        path = args.root / f"flat_native_{label}_matched" / "analysis_full_series.json"
        data = json.loads(path.read_text())
        provenance[label] = {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "native_archive_sha256": data["matched_archive_sha256"]}
        period = data["series"][-1]["time"] / 0.1
        for method, style, text in (("matched", "-", "matched"),
                                    ("practical_control", "--", "control")):
            for ax, key in zip(axes, ("global_liquid_velocity_relative_l2", "column_depth_wave_relative_l2")):
                rows = [r for r in data["series"] if r[method][key] is not None]
                ax.semilogy([r["time"]/period for r in rows], [r[method][key] for r in rows],
                            style, color=colour, marker="o" if method == "matched" else "s",
                            markersize=3, linewidth=1.5, label=f"{label} {text}")
    for ax, title in zip(axes, ("Global velocity error", "Wave-normalised surface error")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Time $t/T$")
        ax.set_xlim(0, .103)
        ax.set_xticks([0, .025, .05, .075, .1])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=.17)
        ax.set_ylabel("Relative error", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4,
               loc="outside upper center", frameon=False)
    fig.savefig(args.output, dpi=300)
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, metadata={"Title": "Matched native transfer-error propagation"})
    plt.close(fig)
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "scope": "matched-setting propagated transfer error against same-grid exact-input numerical trajectory",
        "source": provenance,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
