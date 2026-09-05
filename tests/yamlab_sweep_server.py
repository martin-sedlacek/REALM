"""Reference YAMLab-policy server: checks the LeRobot observation contract, answers with a joint sweep.

    python tests/yamlab_sweep_server.py --port 8123

Speaks openpi's websocket protocol (metadata on connect, then msgpack-numpy request/response), which
is what `realm.inference.client._YamLabAdapter` connects to. Every observation is validated against
realm/inference/yamlab.py's contract -- a violation is answered with a text message, which the client
raises as `RuntimeError("Error in inference server: ...")`, so a wrong payload fails the eval loudly.
The "policy" is YAMLab's benchmark sweep: sinusoidal arm targets (left and right mirrored) and
fingers alternating open/closed every 60 steps, so a rollout driven by it visibly moves both arms.
"""
import argparse

import numpy as np
import websockets.exceptions
import websockets.sync.server
from openpi_client import msgpack_numpy

from realm.inference.yamlab import FINGER_IDX, IMAGE_KEYS, STATE_DIM, STATE_KEY

CHUNK = 8
OPEN, CLOSED = -0.0475, 0.0


def check(obs):
    problems = []
    if not isinstance(obs.get("prompt"), str) or not obs["prompt"]:
        problems.append("prompt missing or empty")
    st = obs.get(STATE_KEY)
    if st is None or np.asarray(st).shape != (STATE_DIM,):
        problems.append(f"{STATE_KEY}: expected shape ({STATE_DIM},), got {None if st is None else np.asarray(st).shape}")
    elif np.asarray(st).dtype != np.float32:
        problems.append(f"{STATE_KEY}: expected float32, got {np.asarray(st).dtype}")
    else:
        for i in FINGER_IDX:
            if not (OPEN - 1e-3 <= float(st[i]) <= CLOSED + 1e-3):
                problems.append(f"{STATE_KEY}[{i}] finger {float(st[i]):.4f} outside [{OPEN}, {CLOSED}]")
    for key in IMAGE_KEYS:
        im = obs.get(key)
        if im is None:
            problems.append(f"{key} missing")
        elif np.asarray(im).ndim != 3 or np.asarray(im).shape[2] != 3 or np.asarray(im).dtype != np.uint8:
            problems.append(f"{key}: expected uint8 (H, W, 3), got {np.asarray(im).dtype} {np.asarray(im).shape}")
    return problems


TOUR_SLOT = 45        # control steps per joint in --mode tour (1.5 s at 30 Hz)
TOUR_AMPLITUDE = 0.3  # rad, one full sine per slot, the SAME sign on both arms


def sweep(t0):
    """(CHUNK, 14) targets for steps t0 .. t0+CHUNK-1: sinusoids on all joints, arms mirrored."""
    out = np.zeros((CHUNK, STATE_DIM), dtype=np.float32)
    for k in range(CHUNK):
        t = t0 + k
        phase = 2 * np.pi * t / 120.0
        arm = 0.25 * np.sin(phase + np.arange(6) * np.pi / 6)
        out[k, 0:6] = arm
        out[k, 7:13] = -arm
        finger = OPEN if (t // 60) % 2 == 0 else CLOSED
        out[k, 6] = out[k, 13] = finger
    return out


def tour(t0):
    """(CHUNK, 14) targets: joints 1..6 one at a time, one +-TOUR_AMPLITUDE sine each over TOUR_SLOT steps,
    identically on both arms, then a gripper slot (close, open); repeats. For eyeballing every joint."""
    out = np.zeros((CHUNK, STATE_DIM), dtype=np.float32)
    for k in range(CHUNK):
        t = t0 + k
        slot, u = (t // TOUR_SLOT) % 7, (t % TOUR_SLOT) / TOUR_SLOT
        out[k, 6] = out[k, 13] = OPEN
        if slot < 6:
            out[k, slot] = out[k, 7 + slot] = TOUR_AMPLITUDE * np.sin(2 * np.pi * u)
        else:
            out[k, 6] = out[k, 13] = CLOSED if u < 0.5 else OPEN
    return out


def main(port, mode="sweep"):
    policy = {"sweep": sweep, "tour": tour}[mode]
    def handler(conn):
        conn.send(msgpack_numpy.packb({"server": "yamlab_sweep_stub", "action_dim": STATE_DIM}))
        packer = msgpack_numpy.Packer()
        t = 0
        try:
            for msg in conn:
                _serve_one(conn, packer, msgpack_numpy.unpackb(msg), t)
                t += CHUNK
        except websockets.exceptions.ConnectionClosed:
            # the eval process exits without a close frame (Isaac tears down with a segfault)
            print(f"[yamlab_sweep_server] client gone after {t} steps", flush=True)

    def _serve_one(conn, packer, obs, t):
        if True:
            problems = check(obs)
            if problems:
                print(f"[yamlab_sweep_server] BAD OBSERVATION: {problems}", flush=True)
                conn.send("yamlab observation contract violated: " + "; ".join(problems))
                return
            if t == 0:
                shapes = {k: tuple(np.asarray(v).shape) for k, v in obs.items() if k != "prompt"}
                print(f"[yamlab_sweep_server] first observation OK: prompt={obs['prompt']!r} shapes={shapes}", flush=True)
            conn.send(packer.pack({"actions": policy(t)}))

    print(f"[yamlab_sweep_server] listening on 0.0.0.0:{port}", flush=True)
    with websockets.sync.server.serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--mode", choices=["sweep", "tour"], default="sweep",
                        help="sweep: all joints, arms mirrored (the adapter test); tour: one joint at a time, both arms equally")
    a = parser.parse_args()
    main(a.port, a.mode)
