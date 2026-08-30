"""Record a drive to a replayable file. Run this BEFORE tuning anything.

    python examples/05_record.py drive_01.l1raw --duration 60

A 60 s drive is roughly 26 MB. The recording is the raw wire, so a later fix
to the parser retroactively applies to every file you have already captured.
"""

import argparse
import logging
import time

from l1_stream.recording import DatagramRecorder, raw_datagrams

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--port", type=int, default=12345)
parser.add_argument("--duration", type=float, default=None,
                    help="Seconds to record. Omit to run until Ctrl-C.")
args = parser.parse_args()

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

print(f"\nWrote {rec.records} datagrams to {args.path}")
if rec.records == 0:
    print("Nothing arrived. Is the publisher running, and on this port?")
