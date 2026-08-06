"""Keyboard control for the CARLA spectator (free-fly) camera.

Run this on the machine where the CARLA server lives (e.g. inside a remote
desktop session). A small pygame window pops up -- keep it focused while
pressing keys and it drives the spectator camera in the CARLA window next to
it. Reads keys via SDL (focused window), so it works over RDP/VNC without sudo.

Controls:
    Pitch   I / K
    Yaw     J / L
    Roll    U / O
    Move    W / A / S / D
    Up/Down Q / E
    Boost   hold Shift
    Snapshot P       (full-res image of the current view -> snapshots/)
    Quit    ESC

Usage:
    python spectator_control.py                  # connects to localhost:2000
    python spectator_control.py <ip> <port>
"""

import os
import sys
import datetime
import subprocess
import carla
import pygame

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

TURN_RATE = 60.0   # degrees per second
MOVE_RATE = 8.0    # meters per second
BOOST = 4.0        # multiplier while holding shift

SNAP_DIR = "snapshots"
WINDOW_NAME = "CarlaUE4"   # title substring of the CARLA render window


def _find_carla_window():
    """Return the X11 window id of the CarlaUE4 render window, or None."""
    try:
        tree = subprocess.check_output(
            ["xwininfo", "-root", "-tree"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in tree.splitlines():
        if WINDOW_NAME.lower() in line.lower() and "0x" in line:
            for tok in line.split():
                if tok.startswith("0x"):
                    return tok
    return None


def take_snapshot(*_):
    """Grab the exact pixels of the CarlaUE4 window via an X11 screen grab
    (matches what's on screen, including FOV and post-processing)."""
    win = _find_carla_window()
    if win is None:
        print(f"[snapshot] failed: no window matching '{WINDOW_NAME}' on "
              f"DISPLAY={os.environ.get('DISPLAY')}")
        return
    os.makedirs(SNAP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAP_DIR, f"carla_{stamp}.png")
    try:
        subprocess.run(["import", "-window", win, path], check=True,
                       stderr=subprocess.PIPE)
        print(f"[snapshot] saved -> {path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[snapshot] grab failed: {exc}")


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    spectator = world.get_spectator()

    pygame.init()
    pygame.display.set_mode((360, 120))
    pygame.display.set_caption("Spectator control - keep this window focused")
    print(f"Connected to {HOST}:{PORT}")
    print("Pitch I/K | Yaw J/L | Roll U/O | Move WASD | Up/Down Q/E | "
          "Shift=boost | P=snapshot | ESC=quit")

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # seconds since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                # Edge-triggered: one press -> one snapshot.
                take_snapshot(world, spectator)

        k = pygame.key.get_pressed()
        if k[pygame.K_ESCAPE]:
            running = False

        try:
            t = spectator.get_transform()
            rot, loc = t.rotation, t.location

            mult = BOOST if (k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]) else 1.0
            dr = TURN_RATE * dt * mult   # degrees this frame
            dm = MOVE_RATE * dt * mult   # meters this frame

            if k[pygame.K_i]: rot.pitch += dr
            if k[pygame.K_k]: rot.pitch -= dr
            if k[pygame.K_j]: rot.yaw   -= dr
            if k[pygame.K_l]: rot.yaw   += dr
            if k[pygame.K_u]: rot.roll  -= dr
            if k[pygame.K_o]: rot.roll  += dr

            fwd = t.get_forward_vector()
            right = t.get_right_vector()
            up = t.get_up_vector()

            if k[pygame.K_w]: loc += fwd * dm
            if k[pygame.K_s]: loc -= fwd * dm
            if k[pygame.K_d]: loc += right * dm
            if k[pygame.K_a]: loc -= right * dm
            if k[pygame.K_q]: loc += up * dm
            if k[pygame.K_e]: loc -= up * dm

            spectator.set_transform(carla.Transform(loc, rot))
        except RuntimeError as exc:
            # Transient RPC timeout (sim briefly busy). Skip this frame
            # instead of letting the C++ exception abort the process.
            print(f"[warn] RPC hiccup, skipping frame: {exc}")

    pygame.quit()


if __name__ == "__main__":
    main()
