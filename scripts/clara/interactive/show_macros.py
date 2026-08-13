"""Print the OG-lite macros as REALM actually sets them, without booting a simulator.

Guards against the null test: if REALM_INCREMENTAL_CONTACT_CACHE never reaches gm, a clean run
proves nothing about the incremental fold. Runs set_sim_config() so this is the real value the
env would be constructed with, not the module default.
"""
import inspect
import os

from omnigibson.macros import gm

from realm.sim_config import set_sim_config

set_sim_config(robot=os.environ.get("REALM_ROBOT", "DROID"))

import omnigibson.utils.usd_utils as uu

print("=== macro state after set_sim_config() ===")
for k in ("INCREMENTAL_CONTACT_CACHE", "PROXIMITY_GATE_ENABLED", "PROXIMITY_GATE_RADIUS",
          "CONTACT_REPORTING_PATTERNS", "ENABLE_VISUAL_UPDATES", "OBJECT_STATE_UPDATE_WHITELIST",
          "RENDER_ON_STEP", "USE_GPU_DYNAMICS"):
    print(f"  gm.{k} = {getattr(gm, k, '<undefined>')}")
print("=== env vars ===")
for k in ("REALM_INCREMENTAL_CONTACT_CACHE", "REALM_PROXIMITY_GATE", "REALM_GPU_DYNAMICS"):
    print(f"  {k} = {os.environ.get(k, '<unset>')}")
print("=== live source ===")
print(f"  usd_utils: {inspect.getfile(uu)}")
src = inspect.getsource(uu)
print(f"  usd_utils has incremental fold branch: {'INCREMENTAL_CONTACT_CACHE' in src}")
print(f"  usd_utils has proximity gate:          {'PROXIMITY_GATE' in src}")
