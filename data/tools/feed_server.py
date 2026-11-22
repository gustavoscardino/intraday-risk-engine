"""Serves a dataset as a live market-data feed over TCP: the rows of prices_15m.csv in arrival order,
one bar at a time, at a chosen pace. A risk engine connected to it receives the bars as it would
from a broker, so the same replay can be run live, with or without faults (serve a folder written
by inject_faults.py).

Usage:
    python data/tools/feed_server.py [--data b3_real] [--port 9000] [--bar-seconds 1] [--from DATE]
    python data/tools/feed_server.py --data data/faults/b3_real/delay --bar-seconds 0.2

Protocol (one line per message, UTF-8, '\\n'):
    timestamp,ticker,open,high,low,close,volume      header, sent once
    2026-11-19 10:15:00,PETR4,...                     a row, as in prices_15m.csv
    #clock 2026-11-19 10:15:00                        the feed has moved past this bar close
    #end                                              no more data

A #clock line follows the rows of each bar (when the next row belongs to a later bar), then the
server waits --bar-seconds: it is the exchange's clock, which a client needs to close a bar when
some stocks did not trade. Rows of a faulty feed that arrive late or out of order are sent where
they are in the file, after the #clock of their bar. --from skips the sessions before DATE (a
client calibrates on the history before it). One client at a time; each connection replays the
feed from the start. Standard library only.
"""
import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)


def resolve(dataset):
    for candidate in (dataset, os.path.join(DATA, dataset)):
        if os.path.isfile(os.path.join(candidate, "prices_15m.csv")):
            return candidate
    sys.exit(f"no prices_15m.csv in {dataset} or data/{dataset}")


def serve_one(conn, lines, header, bar_seconds):
    send = lambda text: conn.sendall((text + "\n").encode("utf-8"))
    send(header)
    clock = None  # latest bar close the feed has reached
    for line in lines:
        stamp = line[:19]
        if clock is not None and stamp > clock and stamp[:16] != clock[:16]:
            send(f"#clock {clock}")
            time.sleep(bar_seconds)
        if clock is None or stamp > clock:
            clock = stamp
        send(line)
    if clock is not None:
        send(f"#clock {clock}")
    send("#end")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="b3_real", help="dataset name (data/<name>) or folder")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--bar-seconds", type=float, default=1.0, help="wall-clock seconds per 15-minute bar")
    ap.add_argument("--from", dest="start", default="", help="first session to send, YYYY-MM-DD")
    ap.add_argument("--once", action="store_true", help="exit after the first client")
    args = ap.parse_args()
    folder = resolve(args.data)
    with open(os.path.join(folder, "prices_15m.csv"), encoding="utf-8") as f:
        header = f.readline().strip()
        lines = [l.rstrip("\r\n") for l in f if l.strip() and l[:10] >= args.start]
    print(f"serving {len(lines)} rows of {os.path.abspath(folder)} on {args.host}:{args.port} "
          f"({args.bar_seconds:g} s per bar)", flush=True)
    with socket.create_server((args.host, args.port)) as srv:
        while True:
            conn, addr = srv.accept()
            print(f"client {addr[0]}:{addr[1]} connected", flush=True)
            try:
                with conn:
                    serve_one(conn, lines, header, args.bar_seconds)
                print("feed finished", flush=True)
            except ConnectionError:  # reset, broken pipe or aborted (Windows) by the client
                print("client disconnected", flush=True)
            if args.once:
                break


if __name__ == "__main__":
    main()
