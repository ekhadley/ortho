"""Runs a command while holding an org.freedesktop.ScreenSaver inhibit, which hypridle's suspend listener honours (systemd-inhibit is not): python3 scripts/awake.py ./run.py <names>. Laptop only."""
import subprocess
import sys

import dbus

ss = dbus.SessionBus().get_object("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver")
cookie = ss.Inhibit("ortho", "evals on the vast box", dbus_interface="org.freedesktop.ScreenSaver")
try: rc = subprocess.call(sys.argv[1:])
finally: ss.UnInhibit(cookie, dbus_interface="org.freedesktop.ScreenSaver")
sys.exit(rc)
