#!/usr/bin/env python3
"""
analyze_tx_samples.py

Reads the raw uint16 occupancy-sample files written by the DPDK TX worker
(one file per queue: "<outfile_base>_q<N>.bin", little-endian uint16 array)
and produces summary statistics, percentiles, and optional histogram plots.

Usage:
    # Auto-discover all "<prefix>_q*.bin" files in a directory
    python3 analyze_tx_samples.py --prefix results/run1

    # Or list files/queues explicitly
    python3 analyze_tx_samples.py --files run1_q0.bin run1_q1.bin

    # Add histogram PNGs (per-queue) and a combined summary CSV
    python3 analyze_tx_samples.py --prefix results/run1 --plot --csv summary.csv
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json

SAMPLE_FILE_MAGIC = 0x5458424D  # "TXBM"
HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u2"),
        ("queue_id", "<u2"),
        ("record_size", "<u4"),
        ("stats_size", "<u4"),
        ("num_records", "<u8"),
        ("nb_tx_desc", "<u4"),
        ("burst_size", "<u4"),
        ("timer_hz", "<u8"),
    ],
    align=True,
)


SAMPLE_DTYPE = np.dtype(
    [
        ("used", "<u4"),
        ("space", "<u4"),
        ("requested", "<u2"),
        ("to_send", "<u2"),
        ("sent", "<u2"),
        ("occupancy_gated", "<u2"),
        ("tx_not_accepted", "<u2"),
        ("reserved0", "<u2"),
        ("cycles_count", "<u8"),
        ("cycles_tx", "<u8"),
        ("cycles_total", "<u8"),
    ],
    align=True,
)


# matplotlib is optional; only needed if --plot is used
try:
    import matplotlib

    matplotlib.use("Agg")  # no display needed, just save PNGs
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

PERCENTILES = [50, 90, 95, 99, 99.9, 99.99]


def discover_files_from_dir(directory, pattern):
    """Recursively find sample binary files under directory."""
    return sorted(
        str(path) for path in Path(directory).rglob(pattern) if path.is_file()
    )


def discover_files(prefix):
    """Find files matching '<prefix>_q<N>.bin', sorted by queue number."""
    pattern = f"{prefix}_q*.bin"
    files = glob.glob(pattern)
    if not files:
        return []

    def qnum(fname):
        m = re.search(r"_q(\d+)\.bin$", fname)
        return int(m.group(1)) if m else -1

    files.sort(key=qnum)
    return files


def parse_json(fname):
    metadata_json = None
    with open(fname, "r") as json_log:
        metadata_json = json.load(json_log)

    return metadata_json


def parse_stdout(fname):
    with open(fname, "r") as stdout_log:
        text = stdout_log.read()

        # Match each Queue benchmark block up to the next queue block or EOF.
        block_re = re.compile(
            r"=+\s*Queue\s+(\d+)\s+benchmark\s*=+\s*"
            r"(.*?)(?==+\s*Queue\s+\d+\s+benchmark\s*=+|\Z)",
            re.DOTALL,
        )

        # Match generic "key : value" lines.
        field_re = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            re.MULTILINE,
        )

        histogram_re = re.compile(
            r"^\s*queue\s+(\d+)\s+used\s*=\s*(\d+)\s*:\s*(\d+)\s*"
            r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\)\s*$",
            re.MULTILINE,
        )

        results = []

        for matchs in block_re.finditer(text):
            queue_id = int(matchs.group(1))
            block = matchs.group(2)

        # Everything before histogram heading is the main stats section.
        stats_text = block.split("TX queue-count histogram:", 1)[0]

        stats = {}
        for key, value in field_re.findall(stats_text):
            if "." in value:
                stats[key] = float(value)
            else:
                stats[key] = int(value)

                histogram = []

            for queue_id, used, count, percent in histogram_re.findall(block):
                histogram.append(
                    {
                        "queue_id": int(queue_id),
                        "used": int(used),
                        "count": int(count),
                        "percent": float(percent),
                    }
                )

            results.append(
                {
                    "queue": queue_id,
                    "stats": stats,
                    "histogram": histogram,
                }
            )

        return results


def queue_id_from_filename(fname):
    m = re.search(r"_q(\d+)\.bin$", fname)
    return m.group(1) if m else os.path.basename(fname)


def load_samples(fname):
    with open(fname, "rb") as f:
        # Header
        hdr_arr = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
        if hdr_arr.size != 1:
            raise RuntimeError(f"{fname}: truncated header")

        hdr = hdr_arr[0]

        if int(hdr["magic"]) != SAMPLE_FILE_MAGIC:
            raise RuntimeError(f"{fname}: bad magic 0x{int(hdr['magic']):08x}")

        if int(hdr["record_size"]) != SAMPLE_DTYPE.itemsize:
            raise RuntimeError(
                f"{fname}: sample_record size mismatch: "
                f"file={int(hdr['record_size'])}, "
                f"python={SAMPLE_DTYPE.itemsize}"
            )

        # Skip queue_stats for now.
        stats_size = int(hdr["stats_size"])
        stats_raw = f.read(stats_size)

        if len(stats_raw) != stats_size:
            raise RuntimeError(f"{fname}: truncated queue_stats")

        # Sample records
        n = int(hdr["num_records"])

        data = np.fromfile(
            f,
            dtype=SAMPLE_DTYPE,
            count=n,
        )

        if data.size != n:
            raise RuntimeError(f"{fname}: expected {n} records, got {data.size}")

    return hdr, stats_raw, data


def compute_stats(data):
    if data.size == 0:
        return None
    stats = {
        "count": int(data.size),
        "min": int(data.min()),
        "max": int(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }
    pct_values = np.percentile(data, PERCENTILES)
    for p, v in zip(PERCENTILES, pct_values):
        stats[f"p{p}"] = float(v)
    return stats


def print_stats(qid, stats, hdr, data):
    print(
        f"\n[q{qid}] "
        f"records={int(hdr['num_records'])} "
        f"tx_desc={int(hdr['nb_tx_desc'])} "
        f"burst={int(hdr['burst_size'])} "
        f"timer_hz={int(hdr['timer_hz'])}"
    )

    def mean_field(data, field):
        return float(data[field].mean()) if data.size else 0.0

    print(f"  avg requested       = {mean_field(data, 'requested'):.2f}")
    print(f"  avg to_send         = {mean_field(data, 'to_send'):.2f}")
    print(f"  avg sent            = {mean_field(data, 'sent'):.2f}")

    print(f"  occupancy_gated     = {int(data['occupancy_gated'].sum())}")
    print(f"  tx_not_accepted     = {int(data['tx_not_accepted'].sum())}")

    print(f"  avg cycles_count    = {mean_field(data, 'cycles_count'):.2f}")
    print(f"  avg cycles_tx       = {mean_field(data, 'cycles_tx'):.2f}")
    print(f"  avg cycles_total    = {mean_field(data, 'cycles_total'):.2f}")

    gated_samples = np.count_nonzero(data["occupancy_gated"])
    rejected_samples = np.count_nonzero(data["tx_not_accepted"])

    print(
        f"  samples gated       = {gated_samples} "
        f"({100.0 * gated_samples / data.size:.6f}%)"
    )

    print(
        f"  samples not accepted= {rejected_samples} "
        f"({100.0 * rejected_samples / data.size:.6f}%)"
    )


def plot_histogram(qid, data, out_dir, bins):
    if not HAVE_MPL:
        print("  (skipping plot: matplotlib not installed)")
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(data, bins=bins, color="#3b6ea5", edgecolor="black", linewidth=0.3)
    plt.title(f"TX queue occupancy distribution — queue {qid}")
    plt.xlabel("Queue occupancy (descriptors used)")
    plt.ylabel("Sample count")
    plt.grid(axis="y", alpha=0.3)
    out_path = os.path.join(out_dir, f"hist_q{qid}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  wrote {out_path}")


def write_csv(csv_path, all_stats):
    all_stats.to_csv(csv_path)
    print(f"\nWrote combined summary CSV to {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--prefix",
        help="outfile_base prefix used by the C program "
        "(auto-discovers <prefix>_q*.bin)",
    )
    ap.add_argument("--files", nargs="+", help="explicit list of .bin files to process")
    ap.add_argument(
        "--dir",
        help="recursively process all samples_q*.bin files under this directory",
    )
    ap.add_argument(
        "--plot", action="store_true", help="save a histogram PNG per queue"
    )
    ap.add_argument(
        "--bins", type=int, default=64, help="number of histogram bins (default 64)"
    )
    ap.add_argument("--out-dir", default=".", help="directory to write PNGs/CSV into")
    ap.add_argument("--csv", help="path to write a combined summary CSV")

    args = ap.parse_args()

    inputs = sum(x is not None for x in (args.prefix, args.files, args.dir))
    if inputs != 1:
        ap.error("must supply exactly one of --prefix, --files, or --dir")

    if args.files:
        files = args.files
    elif args.dir:
        files = discover_files_from_dir(args.dir, "samples_q*.bin")
        stdout_files = discover_files_from_dir(args.dir, "stdout.log")
        metadata = discover_files_from_dir(args.dir, "metadata.json")
    else:
        files = discover_files(args.prefix)

    if not files:
        print("No sample files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} sample files.")

    os.makedirs(args.out_dir, exist_ok=True)

    all_stats = []
    all_data = []

    rows = []
    histogram_rows = []

    for fnames in files:
        for sample_fname in files:
            run_dir = os.path.dirname(sample_fname)

            stdout_fname = os.path.join(run_dir, "stdout.log")
            metadata_fname = os.path.join(run_dir, "metadata.json")

            qid = queue_id_from_filename(sample_fname)
            hdr, qs_raw, data = load_samples(sample_fname)

            occupancy = data["used"]
            stats = compute_stats(occupancy)

            stdout_data = parse_stdout(stdout_fname)
            metadata_data = parse_json(metadata_fname)

            # Find stdout entry corresponding to this queue.
            stdout_entry = next(
                (entry for entry in stdout_data if entry["queue"] == qid), None
            )
            stdout_entry = next(
                (
                    entry
                    for entry in reversed(stdout_data)
                    if int(entry.get("queue", -1)) == int(qid)
                ),
                None
            )

            if stdout_entry is None:
                stdout_stats = {}
                stdout_histogram = []
            else:
                stdout_stats = stdout_entry.get("stats", {})
                stdout_histogram = stdout_entry.get("histogram", [])

            row = {
                "queue_id": qid,
                # metadata.json
                **metadata_data,
                # stdout stats
                **{f"stdout_{k}": v for k, v in stdout_stats.items()},
                # stats calculated from samples_q*.bin
                **{f"occupancy_{k}": v for k, v in stats.items()},
            }

            # -------------------------
            # Histogram table
            # -------------------------
            for h in stdout_histogram:
                histogram_rows.append({
                    "experiment": metadata_data.get("experiment"),
                    "repeat": metadata_data.get("repeat"),
                    "queue_id": qid,
                    "used": h["used"],
                    "count": h["count"],
                    "percent": h["percent"],
                })

            rows.append(row)

    total_df = pd.DataFrame(rows)
    histogram_df = pd.DataFrame(histogram_rows)
    if args.csv:
        filename = args.csv + ".csv"
        write_csv(filename, total_df)
        histogram_filename = args.csv + "_histogram.csv"
        write_csv(histogram_filename, histogram_df)


def load_csv():
    df = pd.read_csv("csv.csv")
    print(df.columns)



if __name__ == "__main__":
    main()
    # load_csv()
