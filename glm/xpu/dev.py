#!/usr/bin/env python3
"""Client for xpu/devserver.py (host side, stdlib only).

  dev.py file.py            run a snippet file on the live model
  dev.py -c "code"          run inline code
  dev.py --quit             stop the server
"""
import os
import socket
import sys

SOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run", "dev.sock")
args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
if args[0] == "-c":
    src = args[1]
elif args[0] == "--quit":
    src = "__quit__"
else:
    src = open(args[0]).read()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(SOCK)
s.sendall(src.encode() + b"\0")
s.shutdown(socket.SHUT_WR)
while True:
    d = s.recv(1 << 16)
    if not d:
        break
    sys.stdout.write(d.decode(errors="replace"))
    sys.stdout.flush()
