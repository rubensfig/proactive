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

SAMPLE_FILE_MAGIC = 0x5458424D   # "TXBM"
HEADER_DTYPE = np.dtype([
    ("magic",       "<u4"),
    ("version",     "<u2"),
    ("queue_id",    "<u2"),
    ("record_size", "<u4"),
    ("stats_size",  "<u4"),
    ("num_records", "<u8"),
    ("nb_tx_desc",  "<u4"),
    ("burst_size",  "<u4"),
    ("timer_hz",    "<u8"),
], align=True)


SAMPLE_DTYPE = np.dtype([
    ("used",              "<u4"),
    ("space",             "<u4"),

    ("requested",         "<u2"),
    ("to_send",           "<u2"),
    ("sent",              "<u2"),
    ("occupancy_gated",   "<u2"),
    ("tx_not_accepted",   "<u2"),
    ("reserved0",         "<u2"),

    ("cycles_count",      "<u8"),
    ("cycles_tx",         "<u8"),
    ("cycles_total",      "<u8"),
], align=True)



# matplotlib is optional; only needed if --plot is used
try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save PNGs
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

PERCENTILES = [50, 90, 95, 99, 99.9, 99.99]


def discover_files_from_dir(directory):
    """Recursively find sample binary files under directory."""
    return sorted(
        str(path)
        for path in Path(directory).rglob("samples_q*.bin")
        if path.is_file()
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
            raise RuntimeError(
                f"{fname}: bad magic 0x{int(hdr['magic']):08x}"
            )

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
            raise RuntimeError(
                f"{fname}: expected {n} records, got {data.size}"
            )

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
    import csv
    fieldnames = ["queue", "count", "min", "max", "mean", "std"] + \
                 [f"p{p}" for p in PERCENTILES]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for qid, stats in all_stats:
            if stats is None:
                continue
            row = {"queue": qid}
            row.update(stats)
            w.writerow(row)
    print(f"\nWrote combined summary CSV to {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", help="outfile_base prefix used by the C program "
                                      "(auto-discovers <prefix>_q*.bin)")
    ap.add_argument("--files", nargs="+", help="explicit list of .bin files to process")
    ap.add_argument(
        "--dir",
        help="recursively process all samples_q*.bin files under this directory",
    )
    ap.add_argument("--plot", action="store_true", help="save a histogram PNG per queue")
    ap.add_argument("--bins", type=int, default=64, help="number of histogram bins (default 64)")
    ap.add_argument("--out-dir", default=".", help="directory to write PNGs/CSV into")
    ap.add_argument("--csv", help="path to write a combined summary CSV")

    args = ap.parse_args()

    inputs = sum(x is not None for x in (args.prefix, args.files, args.dir))
    if inputs != 1:
        ap.error("must supply exactly one of --prefix, --files, or --dir")

    if args.files:
        files = args.files
    elif args.dir:
        files = discover_files_from_dir(args.dir)
    else:
        files = discover_files(args.prefix)

    if not files:
        print("No sample files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} sample files.")

    os.makedirs(args.out_dir, exist_ok=True)

    all_stats = []
    all_data = []

    for fname in files:
        qid = queue_id_from_filename(fname)
        hdr, qs_raw, data = load_samples(fname)
        occupancy = data["used"]
        stats = compute_stats(occupancy)
        print_stats(qid, stats, hdr, data)
        all_stats.append((qid, stats))
        if data.size:
            all_data.append(occupancy)

        if args.plot and data.size:
            plot_histogram(qid, occupancy, args.out_dir, args.bins)

    # Combined (all queues pooled) stats
    if len(all_data) > 1:
        combined = np.concatenate(all_data)
        combined_stats = compute_stats(combined)
        print_stats("ALL", combined_stats, hdr, data)
        all_stats.append(("ALL", combined_stats))
        if args.plot:
            plot_histogram("ALL", combined, args.out_dir, args.bins)

    if args.csv:
        write_csv(args.csv, all_stats)


if __name__ == "__main__":
    main()
