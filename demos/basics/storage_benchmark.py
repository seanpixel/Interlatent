"""
Benchmark write/read time for SQLite vs HDF5 backends.

Usage:
  PYTHONPATH=. python demos/basics/storage_benchmark.py --events 2000 --channels 64
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from interlatent.api import LatentDB
from interlatent.schema import ActivationEvent


def build_events(num_steps: int, channels: int, layer: str, run_id: str):
    events = []
    step = 0
    for s in range(num_steps):
        for ch in range(channels):
            events.append(
                ActivationEvent(
                    run_id=run_id,
                    step=step,
                    layer=layer,
                    channel=ch,
                    prompt="benchmark prompt",
                    prompt_index=0,
                    token_index=s,
                    token=f"tok{s}",
                    tensor=[float(ch % 7) * 0.1],
                    context={"bench": True},
                )
            )
        step += 1
    return events


def time_write_read(db_uri: str, events, layer: str, batch_size: int):
    os.environ["LATENTDB_WRITE_BATCH_SIZE"] = str(batch_size)
    db = LatentDB(db_uri)

    t0 = time.perf_counter()
    for ev in events:
        db.write_event(ev)
    db.flush()
    t1 = time.perf_counter()

    rows = db.fetch_activations(layer=layer)
    t2 = time.perf_counter()

    # Force iter_activations path
    count_iter = 0
    for batch in db.iter_activations(layer=layer, batch_size=1000):
        count_iter += len(batch)
    t3 = time.perf_counter()

    db.close()
    return {
        "write_s": t1 - t0,
        "fetch_s": t2 - t1,
        "iter_s": t3 - t2,
        "rows": len(rows),
        "rows_iter": count_iter,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=1000, help="Number of steps (tokens) to simulate.")
    ap.add_argument("--channels", type=int, default=64, help="Channels per step.")
    ap.add_argument("--layer", type=str, default="llm.layer.0")
    ap.add_argument("--sqlite", type=Path, default=Path("bench_sqlite.db"))
    ap.add_argument("--hdf5", type=Path, default=Path("bench_hdf5.h5"))
    ap.add_argument("--batch-size", type=int, default=512, help="LATENTDB_WRITE_BATCH_SIZE")
    ap.add_argument(
        "--no-align-batch",
        action="store_true",
        help="Disable batch-size alignment (may split steps across flushes).",
    )
    ap.add_argument("--keep", action="store_true", help="Keep benchmark DB files.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["LATENTDB_MAX_CHANNELS"] = str(args.channels)
    events = build_events(args.events, args.channels, args.layer, run_id="bench")

    batch_size = args.batch_size
    if not args.no_align_batch:
        aligned = (batch_size // args.channels) * args.channels
        if aligned <= 0:
            aligned = args.channels
        if aligned != batch_size:
            print(f"[setup] aligning batch_size {batch_size} -> {aligned} (channels={args.channels})")
        batch_size = aligned

    sqlite_uri = f"sqlite:///{args.sqlite}"
    hdf5_uri = f"hdf5:///{args.hdf5}"

    if args.sqlite.exists():
        args.sqlite.unlink()
    if args.hdf5.exists():
        args.hdf5.unlink()

    print(f"[setup] events={len(events)} steps={args.events} channels={args.channels}")
    print(f"[setup] batch_size={batch_size}")

    print("\n[sqlite] timing...")
    sqlite_stats = time_write_read(sqlite_uri, events, args.layer, batch_size)
    print(f"write: {sqlite_stats['write_s']:.3f}s | fetch: {sqlite_stats['fetch_s']:.3f}s | iter: {sqlite_stats['iter_s']:.3f}s")
    print(f"rows(fetch)={sqlite_stats['rows']} rows(iter)={sqlite_stats['rows_iter']}")

    print("\n[hdf5] timing...")
    hdf5_stats = time_write_read(hdf5_uri, events, args.layer, batch_size)
    print(f"write: {hdf5_stats['write_s']:.3f}s | fetch: {hdf5_stats['fetch_s']:.3f}s | iter: {hdf5_stats['iter_s']:.3f}s")
    print(f"rows(fetch)={hdf5_stats['rows']} rows(iter)={hdf5_stats['rows_iter']}")

    if not args.keep:
        if args.sqlite.exists():
            args.sqlite.unlink()
        if args.hdf5.exists():
            args.hdf5.unlink()


if __name__ == "__main__":
    main()
