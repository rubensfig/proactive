#!/usr/bin/env python3
"""
Repeated experiment runner for the tx_shaper_baseline microbenchmark
(dpdk/examples/tx_shaper_baseline).

The runner sweeps:
  * lcore configurations
  * descriptor count / shaping rate / burst
  * transient type: -T 0, -T 1, -T 2 by default, and its period (-D)
  * controller-specific runtime parameters

The controller is selected when the binary is built (tx_controller in
meson.build). --mechanism must match it; after every run the runner checks
the "mechanism" field the binary writes into samples_q*.json.

Controllers (names as in the paper) and their runtime parameters:
  none : Uncoordinated                         (no parameters)
  pab  : Priority-Aware Backpressure           (no parameters)
  rej  : REJ, outcome-based window             -A <add_step> -K <grow_streak>
  cbc  : Completion-Based Capacity, Alg. 1     -I <poll_us>   (T_poll)
  qbc  : Queue Occupancy-Based Capacity, Alg. 2 (no parameters)

Lcore sweep example:
  --lcore-sets '1,2;1,2,3;1,2,3,4'

Each semicolon-separated entry is passed verbatim to DPDK as "-l <entry>".
If --lcore-sets is omitted, the single --lcores value is used.

Example:
  sudo ./run_ubenchmark.py \
      --app ./dpdk-tx_shaper_baseline \
      --mechanism cbc \
      --lcore-sets '1,2;1,2,3;1,2,3,4' \
      --transient-types 0,1 \
      --cbc-poll-us 1,10,100 \
      --descs 4096 \
      --rates 3125000000 \
      --bursts 512 \
      --samples 500000 \
      --repeats 5 \
      --output results_cbc
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


MECHANISMS = ("none", "pab", "rej", "cbc", "qbc")


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
        description="Repeated experiment runner for tx_shaper_baseline"
    )

    # Program / EAL configuration.
    p.add_argument(
        "--app",
        type=Path,
        default=Path("./dpdk-tx_shaper_baseline"),
        help="Path to the tx_shaper_baseline executable",
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
            "Comma-separated rte_tm shaping rates in BYTES/s (rte_tm "
            "convention), e.g. 3125000000 = 25 Gbit/s; 0 disables shaping"
        ),
    )
    p.add_argument(
        "--bursts",
        type=comma_separated_ints,
        default=[32],
        help="Comma-separated requested tx_burst sizes",
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
        default=[0, 1, 2, 3],
        help=(
            "Transient -T values to sweep. By default every combination is "
            "run with -T 0, -T 1, and -T 2 (default: 0,1,2,3)"
        ),
    )

    # Controller/mechanism parameter sweeps. The mechanism must match the
    # compile-time controller enabled in the probe binary.
    p.add_argument(
        "--mechanism",
        type=str.lower,
        choices=MECHANISMS,
        default="none",
        help=(
            "Controller compiled into the binary (tx_controller in "
            "meson.build). Selects which runtime parameters are swept "
            "(default: %(default)s)"
        ),
    )
    p.add_argument(
        "--cbc-poll-us",
        type=comma_separated_ints,
        default=[10],
        help="CBC -I completion polling intervals T_poll in us "
        "(default: %(default)s)",
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
    p.add_argument(
        "--duty-cycle",
        type=comma_separated_ints,
        default=[15],
        help=(
            "Comma-separated on/off periods in ms for transient type 1 "
            "(-D), e.g. 1,5,10,20 (default: %(default)s)"
        ),
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

    if any(v not in (0, 1, 2, 3) for v in args.transient_types):
        raise SystemExit("all --transient-types values must be one of: 0,1,2,3")
    if any(v <= 0 for v in args.duty_cycle):
        raise SystemExit("all --duty-cycle values must be > 0 (ms)")

    lcore_sets = selected_lcore_sets(args)
    if not lcore_sets or any(not value.strip() for value in lcore_sets):
        raise SystemExit("at least one non-empty lcore configuration is required")

    if any(v <= 0 for v in args.cbc_poll_us):
        raise SystemExit("all --cbc-poll-us values must be > 0")
    if any(v <= 0 for v in args.rej_add_step):
        raise SystemExit("all --rej-add-step values must be > 0")
    if any(v <= 0 for v in args.rej_grow_streak):
        raise SystemExit("all --rej-grow-streak values must be > 0")


def active_parameter_grid(args: argparse.Namespace) -> dict[str, list[int]]:
    """Runtime parameters of the selected controller, as name -> values."""
    if args.mechanism == "cbc":
        return {"cbc_poll_us": args.cbc_poll_us}
    if args.mechanism == "rej":
        return {
            "rej_add_step": args.rej_add_step,
            "rej_grow_streak": args.rej_grow_streak,
        }
    return {}


def controller_parameter_sets(args: argparse.Namespace) -> list[dict[str, int]]:
    """Cartesian product of the selected controller's parameter values."""
    grid = active_parameter_grid(args)
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


# Runtime option of the binary for each controller parameter.
PARAM_FLAGS = {
    "cbc_poll_us": "-I",
    "rej_add_step": "-A",
    "rej_grow_streak": "-K",
}


def build_command(
    args: argparse.Namespace,
    lcores: str,
    nb_desc: int,
    rate_bps: int,
    burst: int,
    transient_type: int,
    duty_cycle: int,
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
            "-c",
            str(args.samples),
            "-T",
            str(transient_type),
            "-D",
            str(duty_cycle),
            "-o",
            str(sample_base),
        ]
    )

    for name, value in controller_params.items():
        cmd.extend([PARAM_FLAGS[name], str(value)])

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
    transient_type: int,
    duty_cycle: int,
    mechanism: str,
    controller_params: dict[str, int],
) -> str:
    name = (
        f"cores_{lcore_label(lcores)}_"
        f"T_{transient_type}_"
        f"duty_{duty_cycle:03d}_"
        f"desc_{nb_desc:04d}_"
        f"rate_{rate_bps:010d}_"
        f"burst_{burst:03d}"
    )

    if mechanism == "cbc":
        name += f"_cbc_poll_{controller_params['cbc_poll_us']:04d}us"
    elif mechanism == "rej":
        name += (
            f"_rej_add_{controller_params['rej_add_step']:04d}"
            f"_streak_{controller_params['rej_grow_streak']:04d}"
        )

    return name


def format_controller_params(params: dict[str, int]) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def binary_mechanism(run_dir: Path) -> str | None:
    """Controller the binary reports in samples_q*.json, lower-case."""
    for path in sorted(run_dir.glob("samples_q*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                value = json.load(f).get("mechanism")
        except (OSError, ValueError):
            continue
        if value:
            return str(value).lower()
    return None


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
            transient_type,
            duty_cycle,
            controller_params,
        )
        for lcores, nb_desc, rate_bps, burst, transient_type, duty_cycle
        in itertools.product(
            lcore_sets,
            args.descs,
            args.rates,
            args.bursts,
            args.transient_types,
            args.duty_cycle,
        )
        for controller_params in controller_sets
    ]

    total_runs = len(combinations) * args.repeats

    print(f"Application : {app}")
    print(f"Lcore sets  : {', '.join(lcore_sets)}")
    print(f"Transient T : {','.join(str(x) for x in args.transient_types)}")
    print(f"Duty cycles : {','.join(str(x) for x in args.duty_cycle)}")
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
        "duty_cycle": args.duty_cycle,
        "port": args.port,
        "descs": args.descs,
        "rates_bps": args.rates,
        "bursts": args.bursts,
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
        transient_type,
        duty_cycle,
        controller_params,
    ) in combinations:
        exp_name = experiment_name(
            lcores=lcores,
            nb_desc=nb_desc,
            rate_bps=rate_bps,
            burst=burst,
            transient_type=transient_type,
            duty_cycle=duty_cycle,
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
                transient_type=transient_type,
                controller_params=controller_params,
                sample_base=sample_base,
                duty_cycle=duty_cycle,
            )

            controller_text = format_controller_params(controller_params)
            if controller_text:
                controller_text = " " + controller_text

            print(
                f"[{run_number:03d}/{total_runs:03d}] "
                f"mech={args.mechanism} "
                f"lcores={lcores} "
                f"T={transient_type} "
                f"D={duty_cycle}ms "
                f"desc={nb_desc} "
                f"rate={rate_bps} "
                f"burst={burst}"
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
                "transient_type": transient_type,
                "duty_cycle": duty_cycle,
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

            # Guard against running a binary built for another controller.
            built = binary_mechanism(run_dir)
            metadata["binary_mechanism"] = built
            if returncode == 0 and built is not None and built != args.mechanism:
                metadata["success"] = False
                metadata["error"] = (
                    f"binary was built for {built}, not {args.mechanism}"
                )
                returncode = 2
                print(
                    f"  ERROR: {metadata['error']} (tx_controller in "
                    "meson.build)",
                    file=sys.stderr,
                )

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
