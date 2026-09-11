"""Record a drive to a replayable file. Run this BEFORE tuning anything.

    python examples/05_record.py personal/line_01.l1raw --duration 60 \
        --kind line --truth 5.0 --truth-method suitcase-at-0-mark \
        --speed 0.5 --environment bare-corridor

A 60 s drive is roughly 26 MB. The recording is the raw wire, so a later fix
to the parser retroactively applies to every file you have already captured.

DESCRIBE THE DRIVE AS YOU RECORD IT. The flags above write a sidecar JSON next
to the recording holding the truth value, the method, the speed, the
environment, and the software provenance. This is not bookkeeping for its own
sake -- two findings on this project were lost or inverted because the context
lived in a filename (``line_drive_8m`` was a 7 m drive) or in nobody's notes at
all. Thirty seconds at capture time, when you still know the answer.

The script WARNS if you record a line or loop with no truth value, because a
drive you cannot score is a drive you will re-record.
"""

import argparse
import logging
import time

from l1_stream.metadata import RecordingMeta
from l1_stream.recording import DatagramRecorder, raw_datagrams

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--port", type=int, default=12345)
parser.add_argument("--duration", type=float, default=None,
                    help="Seconds to record. Omit to run until Ctrl-C.")

meta_args = parser.add_argument_group("what this drive is (written to a sidecar)")
meta_args.add_argument("--kind", choices=["line", "loop", "pivot", "stationary"])
meta_args.add_argument("--truth", type=float, default=None,
                       help="Tape/laser distance in m. Use 0 for a closed loop.")
meta_args.add_argument("--truth-method", default=None,
                       help="e.g. suitcase-at-0-mark, laser, tape, closed-loop")
meta_args.add_argument("--speed", type=float, default=None, help="m/s, nominal")
meta_args.add_argument("--environment", default=None,
                       help="e.g. cluttered-room, bare-corridor, open-hall")
meta_args.add_argument("--notes", default="")
meta_args.add_argument("--no-sidecar", action="store_true",
                       help="Skip the sidecar. You will regret this.")
args = parser.parse_args()

if args.kind in ("line", "loop") and args.truth is None and not args.no_sidecar:
    print("WARNING: a --kind line/loop drive with no --truth cannot be scored "
          "later. Three of five pilot line drives died this way.\n")

started = time.monotonic()
last_report = 0.0

with DatagramRecorder(args.path) as rec:
    try:
        for recv_time, data in raw_datagrams(port=args.port, duration=args.duration):
            rec.write(recv_time, data)
            elapsed = time.monotonic() - started
            if elapsed - last_report >= 1.0:
                last_report = elapsed
                print(f"\r{elapsed:6.1f}s  {rec.records:7d} datagrams  "
                      f"{rec.bytes_written/1e6:6.1f} MB", end="", flush=True)
    except KeyboardInterrupt:
        print()

elapsed = time.monotonic() - started
print(f"\nWrote {rec.records} datagrams to {args.path}")

if rec.records == 0:
    print("Nothing arrived. Is the publisher running, and on this port?")
elif not args.no_sidecar:
    meta = RecordingMeta(
        kind=args.kind,
        truth_m=args.truth,
        truth_method=args.truth_method,
        speed_mps=args.speed,
        environment=args.environment,
        notes=args.notes,
        duration_s=round(elapsed, 2),
        datagrams=rec.records,
        bytes_written=rec.bytes_written,
        port=args.port,
    )
    sidecar = meta.save(args.path)
    print(f"Wrote {sidecar}")
    print(f"  {meta.summary()}")
