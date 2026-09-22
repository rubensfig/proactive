#!/usr/bin/env python3
"""
Repeated experiment runner for tx_occupancy_probe.

The runner sweeps:
  * lcore configurations
  * descriptor count / shaping rate / burst / pacing
  * transient type: -T 0, -T 1, -T 2 by default
  * mechanism-specific runtime controller parameters

Controller parameters:
  COMP: -N <near_steps>
  BQL : -I <interval_us> -G <grow_step>
  REJ : -A <add_step> -K <grow_streak>

Lcore sweep example:
  --lcore-sets '1,2;1,2,3;1,2,3,4'

Each semicolon-separated entry is passed verbatim to DPDK as "-l <entry>".
If --lcore-sets is omitted, the legacy single --lcores value is used.

By default, every other parameter combination is run once for each transient
mode 0, 1, and 2.  Use --transient-types to select a subset if needed.

Example:
  sudo ./run_ubenchmark_adapted.py \
      --app ./tx_occupancy_probe_rej \
      --mechanism rej \
      --lcore-sets '1,2;1,2,3;1,2,3,4' \
      --transient-types 0,1,2 \
      --rej-add-step 4,8,16 \
      --rej-grow-streak 8,16,32,64 \
      --descs 256 \
      --rates 125000000 \
      --bursts 32 \
      --samples 5000000 \
      --repeats 5 \
      --output results_rej
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone


def comma_separated_ints(value: str) -> list[int]:
    try:
        values = [int(x.strip(), 0) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated integer list: {value}"
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return values


def semicolon_separated_strings(value: str) -> list[str]:
    values = [x.strip() for x in value.split(";") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("lcore set list must not be empty")
    return values


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
        help=(
            'Single DPDK EAL lcore list passed as "-l". Used when '
            "--lcore-sets is not supplied (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--lcore-sets",
        type=semicolon_separated_strings,
        default=None,
        help=(
            "Semicolon-separated DPDK -l configurations to sweep. "
            "Quote the value in the shell, e.g. "
            "--lcore-sets '1,2;1,2,3;1,2,3,4'"
        ),
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
    p.add_argument(
        "--transient-types",
        type=comma_separated_ints,
        default=[0, 1, 2],
        help=(
            "Transient -T values to sweep. By default every combination is "
            "run with -T 0, -T 1, and -T 2 (default: 0,1,2)"
        ),
    )

    # Controller/mechanism parameter sweeps. The mechanism must match the
    # compile-time controller enabled in the probe binary.
    p.add_argument(
        "--mechanism",
        type=str.lower,
        choices=("none", "comp", "bql", "rej"),
        default="none",
        help=(
            "Controller compiled into the probe. Selects which runtime "
            "controller parameters are swept/passed (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--comp-near-steps",
        type=comma_separated_ints,
        default=[16],
        help="COMP -N values to sweep (default: %(default)s)",
    )
    p.add_argument(
        "--bql-interval-us",
        type=comma_separated_ints,
        default=[10],
        help="BQL -I cleanup/update intervals in us (default: %(default)s)",
    )
    p.add_argument(
        "--bql-grow-step",
        type=comma_separated_ints,
        default=[64],
        help="BQL -G grow-step values (default: %(default)s)",
    )
    p.add_argument(
        "--rej-add-step",
        type=comma_separated_ints,
        default=[8],
        help="REJ -A additive increase values (default: %(default)s)",
    )
    p.add_argument(
        "--rej-grow-streak",
        type=comma_separated_ints,
        default=[32],
        help="REJ -K success-streak values (default: %(default)s)",
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


def selected_lcore_sets(args: argparse.Namespace) -> list[str]:
    return args.lcore_sets if args.lcore_sets is not None else [args.lcores]


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.cooldown < 0:
        raise SystemExit("--cooldown must be >= 0")

    if any(v <= 0 for v in args.descs):
        raise SystemExit("all --descs values must be > 0")
    if any(v < 0 for v in args.rates):
        raise SystemExit("all --rates values must be >= 0")
    if any(v <= 0 for v in args.bursts):
        raise SystemExit("all --bursts values must be > 0")
    if any(v < 0 for v in args.sleep_values):
        raise SystemExit("all --sleep-ns values must be >= 0")

    if any(v not in (0, 1, 2) for v in args.transient_types):
        raise SystemExit("all --transient-types values must be one of: 0,1,2")

    lcore_sets = selected_lcore_sets(args)
    if not lcore_sets or any(not value.strip() for value in lcore_sets):
        raise SystemExit("at least one non-empty lcore configuration is required")

    # COMP near_steps=0 is deliberately allowed as a useful baseline that
    # effectively removes the near-watermark region.
    if any(v < 0 for v in args.comp_near_steps):
        raise SystemExit("all --comp-near-steps values must be >= 0")
    if any(v <= 0 for v in args.bql_interval_us):
        raise SystemExit("all --bql-interval-us values must be > 0")
    if any(v <= 0 for v in args.bql_grow_step):
        raise SystemExit("all --bql-grow-step values must be > 0")
    if any(v <= 0 for v in args.rej_add_step):
        raise SystemExit("all --rej-add-step values must be > 0")
    if any(v <= 0 for v in args.rej_grow_streak):
        raise SystemExit("all --rej-grow-streak values must be > 0")


def controller_parameter_sets(args: argparse.Namespace) -> list[dict[str, int]]:
    """Return only the parameter dimensions relevant to the selected binary."""
    if args.mechanism == "comp":
        return [
            {"comp_near_steps": near_steps}
            for near_steps in args.comp_near_steps
        ]

    if args.mechanism == "bql":
        return [
            {
                "bql_interval_us": interval_us,
                "bql_grow_step": grow_step,
            }
            for interval_us, grow_step in itertools.product(
                args.bql_interval_us,
                args.bql_grow_step,
            )
        ]

    if args.mechanism == "rej":
        return [
            {
                "rej_add_step": add_step,
                "rej_grow_streak": grow_streak,
            }
            for add_step, grow_streak in itertools.product(
                args.rej_add_step,
                args.rej_grow_streak,
            )
        ]

    return [{}]


def active_parameter_grid(args: argparse.Namespace) -> dict[str, list[int]]:
    if args.mechanism == "comp":
        return {"comp_near_steps": args.comp_near_steps}
    if args.mechanism == "bql":
        return {
            "bql_interval_us": args.bql_interval_us,
            "bql_grow_step": args.bql_grow_step,
        }
    if args.mechanism == "rej":
        return {
            "rej_add_step": args.rej_add_step,
            "rej_grow_streak": args.rej_grow_streak,
        }
    return {}


def build_command(
    args: argparse.Namespace,
    lcores: str,
    nb_desc: int,
    rate_bps: int,
    burst: int,
    sleep_ns: int,
    transient_type: int,
    controller_params: dict[str, int],
    sample_base: Path,
) -> list[str]:
    cmd: list[str] = []

    if args.sudo:
        cmd.append("sudo")

    cmd.extend(
        [
            str(args.app),
            "-l",
            lcores,
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
            "-T",
            str(transient_type),
            "-o",
            str(sample_base),
        ]
    )

    if args.mechanism == "comp":
        cmd.extend(["-N", str(controller_params["comp_near_steps"])])
    elif args.mechanism == "bql":
        cmd.extend(
            [
                "-I",
                str(controller_params["bql_interval_us"]),
                "-G",
                str(controller_params["bql_grow_step"]),
            ]
        )
    elif args.mechanism == "rej":
        cmd.extend(
            [
                "-A",
                str(controller_params["rej_add_step"]),
                "-K",
                str(controller_params["rej_grow_streak"]),
            ]
        )

    if args.app_args:
        cmd.extend(shlex.split(args.app_args))

    return cmd


def lcore_label(lcores: str) -> str:
    """Return a filesystem-safe label while preserving the lcore identity."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", lcores.strip())
    return label.strip("-") or "unknown"


def experiment_name(
    lcores: str,
    nb_desc: int,
    rate_bps: int,
    burst: int,
    sleep_ns: int,
    transient_type: int,
    mechanism: str,
    controller_params: dict[str, int],
) -> str:
    name = (
        f"cores_{lcore_label(lcores)}_"
        f"T_{transient_type}_"
        f"desc_{nb_desc:04d}_"
        f"rate_{rate_bps:010d}_"
        f"burst_{burst:03d}_"
        f"sleep_{sleep_ns}"
    )

    if mechanism == "comp":
        name += f"_comp_near_{controller_params['comp_near_steps']:04d}"
    elif mechanism == "bql":
        name += (
            f"_bql_int_{controller_params['bql_interval_us']:04d}us"
            f"_grow_{controller_params['bql_grow_step']:04d}"
        )
    elif mechanism == "rej":
        name += (
            f"_rej_add_{controller_params['rej_add_step']:04d}"
            f"_streak_{controller_params['rej_grow_streak']:04d}"
        )

    return name


def format_controller_params(mechanism: str, params: dict[str, int]) -> str:
    if mechanism == "comp":
        return f"near={params['comp_near_steps']}"
    if mechanism == "bql":
        return (
            f"interval={params['bql_interval_us']}us "
            f"grow={params['bql_grow_step']}"
        )
    if mechanism == "rej":
        return (
            f"add={params['rej_add_step']} "
            f"streak={params['rej_grow_streak']}"
        )
    return ""


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)

    app = args.app.expanduser()

    # Resolve executable before changing/creating anything.
    if not args.dry_run:
        if not app.exists():
            raise SystemExit(f"application does not exist: {app}")
        if not os.access(app, os.X_OK):
            raise SystemExit(f"application is not executable: {app}")

    app = app.resolve()
    args.app = app

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lcore_sets = selected_lcore_sets(args)
    controller_sets = controller_parameter_sets(args)

    combinations = [
        (
            lcores,
            nb_desc,
            rate_bps,
            burst,
            sleep_ns,
            transient_type,
            controller_params,
        )
        for lcores, nb_desc, rate_bps, burst, sleep_ns, transient_type
        in itertools.product(
            lcore_sets,
            args.descs,
            args.rates,
            args.bursts,
            args.sleep_values,
            args.transient_types,
        )
        for controller_params in controller_sets
    ]

    total_runs = len(combinations) * args.repeats

    print(f"Application : {app}")
    print(f"Lcore sets  : {', '.join(lcore_sets)}")
    print(f"Transient T : {','.join(str(x) for x in args.transient_types)}")
    print(f"Mechanism   : {args.mechanism.upper()}")
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
        "lcore_sets": lcore_sets,
        "transient_types": args.transient_types,
        "port": args.port,
        "descs": args.descs,
        "rates_bps": args.rates,
        "bursts": args.bursts,
        "sleep_ns": args.sleep_values,
        "samples_per_queue": args.samples,
        "mechanism": args.mechanism,
        "mechanism_parameter_grid": active_parameter_grid(args),
        "repeats": args.repeats,
        "cooldown_seconds": args.cooldown,
        "eal_args": args.eal_args,
        "app_args": args.app_args,
        "parameter_combinations": len(combinations),
        "total_runs": total_runs,
    }
    write_json(output_root / "campaign.json", campaign_metadata)

    for (
        lcores,
        nb_desc,
        rate_bps,
        burst,
        sleep_ns,
        transient_type,
        controller_params,
    ) in combinations:
        exp_name = experiment_name(
            lcores=lcores,
            nb_desc=nb_desc,
            rate_bps=rate_bps,
            burst=burst,
            sleep_ns=sleep_ns,
            transient_type=transient_type,
            mechanism=args.mechanism,
            controller_params=controller_params,
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
                lcores=lcores,
                nb_desc=nb_desc,
                rate_bps=rate_bps,
                burst=burst,
                sleep_ns=sleep_ns,
                transient_type=transient_type,
                controller_params=controller_params,
                sample_base=sample_base,
            )

            controller_text = format_controller_params(
                args.mechanism, controller_params
            )
            if controller_text:
                controller_text = " " + controller_text

            print(
                f"[{run_number:03d}/{total_runs:03d}] "
                f"mech={args.mechanism} "
                f"lcores={lcores} "
                f"T={transient_type} "
                f"desc={nb_desc} "
                f"rate={rate_bps} "
                f"burst={burst} "
                f"sleep={sleep_ns}ns"
                f"{controller_text} "
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
                "transient_type": transient_type,
                "samples_per_queue": args.samples,
                "mechanism": args.mechanism,
                "mechanism_parameters": controller_params,
                **controller_params,
                "port": args.port,
                "lcores": lcores,
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
                    metadata["elapsed_seconds"] = time.monotonic() - start_mono
                    metadata["success"] = False
                    metadata["interrupted"] = True
                    write_json(run_dir / "metadata.json", metadata)

                    print("\nInterrupted.", file=sys.stderr)
                    return 130

                except Exception as exc:
                    metadata["finished_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    metadata["elapsed_seconds"] = time.monotonic() - start_mono
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
