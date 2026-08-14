#!/bin/bash
# Compare the REGISTERED PhysxSchema between the RoboLab (Isaac Lab 2.2.0) and REALM (og391)
# containers. If naturalFrequency/dampingRatio are missing from BOTH, PhysX ignores the authored
# mimic compliance on BOTH stacks and the attribute cannot be the RoboLab-vs-REALM difference.
for spec in "ROBOLAB /mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif /workspace/isaaclab/_isaac_sim/extscache" \
            "REALM /mnt/home_lustre/sedlam56/projects/REALM/realm_og391.sif /opt/conda/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache"; do
  set -- $spec
  echo "########## $1 ##########"
  apptainer exec "$2" bash -c 'f=$(ls -d '"$3"'/omni.usd.schema.physx-*/plugins/PhysxSchema/resources/generatedSchema.usda 2>/dev/null | head -1); echo SCHEMA=$f; echo "--- compliance tokens anywhere in registered physx schema ---"; grep -oE "naturalFrequency|dampingRatio|springStiffness|springDamping" "$f" | sort | uniq -c; echo "(no output above = none present)"; echo "--- md5 of whole registered physx schema ---"; md5sum "$f"; echo "--- naturalFrequency as a SYMBOL/STRING in the physx binaries ---"; for so in '"$3"'/omni.physx-*/bin/*.so '"$3"'/omni.usd.schema.physx-*/bin/*.so; do [ -f "$so" ] || continue; n=$(strings -a "$so" | grep -c "naturalFrequency"); d=$(strings -a "$so" | grep -c "dampingRatio"); echo "$(basename $so): naturalFrequency=$n dampingRatio=$d"; done'
done
