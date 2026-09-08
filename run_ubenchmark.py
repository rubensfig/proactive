#!/usr/bin/env python3
"""
run_tx_occupancy_probe.py

Repeated experiment runner for tx_occupancy_probe.

Example:
    sudo ./run_tx_occupancy_probe.py \
        --app ./tx_occupancy_probe \
        --lcores 1,2,3,4 \
        --descs 256,512,1024 \
        --rates 125000000,250000000,500000000,1000000000 \
        --bursts 32 \
        --samples 5000000 \
        --repeats 5 \
        --output results

Each experiment is placed under a directory such as:

    results/
      desc_0256_rate_125000000_burst_032/
        repeat_001/
          samples_q0.bin
          samples_q1.bin
          samples_q2.bin
          samples_q3.bin
          stdout.log
          stderr.log
          metadata.json
        repeat_002/
          ...

The command executed for each run is approximately:

    ./tx_occupancy_probe \
        -l 1,2,3,4 \
        -- \
        -p 0 \
        -n 1024 \
        -b 125000000 \
        -m 32 \
        -s 0 \
        -c 5000000 \
        -o /absolute/path/to/run/samples
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone


def comma_separated_ints(value: str) -> list[int]:
    try:
        return [int(x.strip(), 0) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated integer list: {value}"
        ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeated experiment runner for tx_occupancy_probe"
    )

    # Program / EAL configuration.
    p.add_argument(
        "--app",
        type=Path,
        default=Path("./tx_occupancy_probe"),
        help="Path to tx_occupancy_probe executable",
    )
    p.add_argument(
        "--lcores",
        default="1,2,3,4",
        help='DPDK EAL lcore list passed as "-l" (default: %(default)s)',
    )

    # Probe arguments.
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="DPDK port ID (default: %(default)s)",
    )
    p.add_argument(
        "--descs",
        type=comma_separated_ints,
        default=[256],
        help="Comma-separated Tx descriptor counts, e.g. 256,512,1024",
    )
    p.add_argument(
        "--rates",
        type=comma_separated_ints,
        default=[125_000_000],
        help=(
            "Comma-separated rte_tm shaping rates in bps, "
            "e.g. 125000000,250000000,1000000000; 0 disables shaping"
        ),
    )
    p.add_argument(
        "--bursts",
        type=comma_separated_ints,
        default=[32],
        help="Comma-separated requested tx_burst sizes",
    )
    p.add_argument(
        "--sleep-ns",
        dest="sleep_values",
        type=comma_separated_ints,
        default=[0],
        help="Comma-separated busy-loop delays in ns",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=5_000_000,
        help="Samples to collect per Tx queue",
    )

    # Repetition / experiment management.
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions of every parameter combination",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tx_probe_results"),
        help="Top-level output directory",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Seconds to wait between runs",
    )

    # Optional command-line additions.
    p.add_argument(
        "--eal-args",
        default="",
        help=(
            "Additional EAL arguments inserted before '--'. "
            'Example: --eal-args="-a 0002:01:00.0 --file-prefix=txprobe"'
        ),
    )
    p.add_argument(
        "--app-args",
        default="",
        help="Additional application arguments appended after standard args",
    )

    # Execution behavior.
    p.add_argument(
        "--sudo",
        action="store_true",
        help="Run application via sudo",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with subsequent runs if one run fails",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    return p.parse_args()


def build_command(
    args: argparse.Namespace,
    nb_desc: int,
    rate_bps: int,
    burst: int,
    sleep_ns: int,
    sample_base: Path,
) -> list[str]:
    cmd: list[str] = []

    if args.sudo:
        cmd.append("sudo")

    cmd.extend(
        [
            str(args.app),
            "-l",
            args.lcores,
        ]
    )

    if args.eal_args:
        cmd.extend(shlex.split(args.eal_args))

    cmd.append("--")

    cmd.extend(
        [
            "-p",
            str(args.port),
            "-n",
            str(nb_desc),
            "-b",
            str(rate_bps),
            "-m",
            str(burst),
            "-s",
            str(sleep_ns),
            "-c",
            str(args.samples),
            "-o",
            str(sample_base),
        ]
    )

    if args.app_args:
        cmd.extend(shlex.split(args.app_args))

    return cmd


def experiment_name(
    nb_desc: int,
    rate_bps: int,
    burst: int,
    sleep_ns: int,
) -> str:
    return (
        f"desc_{nb_desc:04d}_"
        f"rate_{rate_bps:010d}_"
        f"burst_{burst:03d}_"
        f"sleep_{sleep_ns}"
    )


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    app = args.app.expanduser()

    # Resolve executable before changing/creating anything.
    if not args.dry_run:
        if not app.exists():
            raise SystemExit(f"application does not exist: {app}")
        if not os.access(app, os.X_OK):
            raise SystemExit(f"application is not executable: {app}")

    app = app.resolve()

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(
            args.descs,
            args.rates,
            args.bursts,
            args.sleep_values,
        )
    )

    total_runs = len(combinations) * args.repeats

    print(f"Application : {app}")
    print(f"Lcores      : {args.lcores}")
    print(f"Experiments : {len(combinations)} parameter combinations")
    print(f"Repeats     : {args.repeats}")
    print(f"Total runs  : {total_runs}")
    print(f"Output      : {output_root}")
    print()

    run_number = 0
    failures = 0

    campaign_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app": str(app),
        "lcores": args.lcores,
        "port": args.port,
        "descs": args.descs,
        "rates_bps": args.rates,
        "bursts": args.bursts,
        "sleep_ns": args.sleep_values,
        "samples_per_queue": args.samples,
        "repeats": args.repeats,
        "cooldown_seconds": args.cooldown,
        "eal_args": args.eal_args,
        "app_args": args.app_args,
        "total_runs": total_runs,
    }
    write_json(output_root / "campaign.json", campaign_metadata)

    for nb_desc, rate_bps, burst, sleep_ns in combinations:
        exp_name = experiment_name(
            nb_desc=nb_desc,
            rate_bps=rate_bps,
            burst=burst,
            sleep_ns=sleep_ns,
        )

        exp_dir = output_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for repeat in range(1, args.repeats + 1):
            run_number += 1

            run_dir = exp_dir / f"repeat_{repeat:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Absolute path is intentional. It avoids ambiguity if the probe or
            # DPDK changes its current working directory.
            sample_base = run_dir / "samples"

            cmd = build_command(
                args=args,
                nb_desc=nb_desc,
                rate_bps=rate_bps,
                burst=burst,
                sleep_ns=sleep_ns,
                sample_base=sample_base,
            )

            print(
                f"[{run_number:03d}/{total_runs:03d}] "
                f"desc={nb_desc} "
                f"rate={rate_bps} "
                f"burst={burst} "
                f"sleep={sleep_ns}ns "
                f"repeat={repeat}/{args.repeats}"
            )
            print("  " + shlex.join(cmd))

            metadata = {
                "experiment": exp_name,
                "repeat": repeat,
                "run_number": run_number,
                "nb_tx_desc": nb_desc,
                "rate_bps": rate_bps,
                "burst": burst,
                "sleep_ns": sleep_ns,
                "samples_per_queue": args.samples,
                "port": args.port,
                "lcores": args.lcores,
                "sample_base": str(sample_base),
                "command": cmd,
                "command_shell": shlex.join(cmd),
                "started_utc": None,
                "finished_utc": None,
                "elapsed_seconds": None,
                "returncode": None,
                "success": None,
            }

            if args.dry_run:
                metadata["success"] = None
                metadata["dry_run"] = True
                write_json(run_dir / "metadata.json", metadata)
                print()
                continue

            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"

            start_wall = datetime.now(timezone.utc)
            start_mono = time.monotonic()
            metadata["started_utc"] = start_wall.isoformat()
            write_json(run_dir / "metadata.json", metadata)

            with (
                stdout_path.open("wb") as stdout_file,
                stderr_path.open("wb") as stderr_file,
            ):
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        check=False,
                    )
                    returncode = result.returncode

                except KeyboardInterrupt:
                    metadata["finished_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    metadata["elapsed_seconds"] = (
                        time.monotonic() - start_mono
                    )
                    metadata["success"] = False
                    metadata["interrupted"] = True
                    write_json(run_dir / "metadata.json", metadata)

                    print("\nInterrupted.", file=sys.stderr)
                    return 130

                except Exception as exc:
                    metadata["finished_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    metadata["elapsed_seconds"] = (
                        time.monotonic() - start_mono
                    )
                    metadata["success"] = False
                    metadata["exception"] = repr(exc)
                    write_json(run_dir / "metadata.json", metadata)

                    failures += 1
                    print(f"  ERROR: {exc}", file=sys.stderr)

                    if not args.continue_on_error:
                        return 1

                    continue

            elapsed = time.monotonic() - start_mono

            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["elapsed_seconds"] = elapsed
            metadata["returncode"] = returncode
            metadata["success"] = returncode == 0

            # Record the files actually produced by the application.
            sample_files = sorted(run_dir.glob("samples_q*.bin"))
            metadata["sample_files"] = [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sample_files
            ]

            write_json(run_dir / "metadata.json", metadata)

            if returncode == 0:
                print(
                    f"  completed in {elapsed:.2f}s; "
                    f"{len(sample_files)} sample file(s)"
                )
            else:
                failures += 1
                print(
                    f"  FAILED: return code {returncode}; "
                    f"see {stderr_path}",
                    file=sys.stderr,
                )

                if not args.continue_on_error:
                    return returncode or 1

            if run_number < total_runs and args.cooldown > 0:
                time.sleep(args.cooldown)

            print()

    print(
        f"Finished: {total_runs} run(s), "
        f"{failures} failure(s), "
        f"{total_runs - failures} successful."
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
