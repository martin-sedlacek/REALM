
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import create_joint


class SceneSetupMixin:

    def update_robot_physics(self):

        if not self.robot_name.startswith("DROID"):
            return

        friction = np.array(self.cfg["robots"][0]["friction"])
        armature = np.array(self.cfg["robots"][0]["armature"])
        joint_names = self.robot.arm_joint_names
        # OG 3.9.1 runs under the Fabric Scene Delegate, where raw USD edits are not automatically
        # propagated to Fabric; every USD write must happen inside this context (which synchronizes
        # on exit) or OmniGibson aborts. The context must not be nested, so all edits go in one block.
        with og.sim.editing_usd():
            self._author_arm_joint_physics(joint_names, friction, armature)
            self._convexify_dynamic_collision_meshes()

    def restore_authored_link_coms(self):
        """Put back the centre of mass the asset authors on every robot link.

        OmniGibson's ``RigidPrim.update_meshes()`` replaces each link's CoM with the volume-weighted
        centroid of its collision meshes composed only ONE level up (``frame="parent"``), which is wrong
        whenever a collision mesh sits under an intermediate Xform, and it writes that value over the
        authored one, so the live stage no longer knows what the asset said. On the YAM links the meshes
        sit under scaled Xforms and the loaded CoMs land metres away (link_1 at (1.56, -0.11, 6.45) m,
        Slurm 204612), inflating every joint's inertia ~100x: a 0.3 rad step took 0.8 s at the effort
        clamp with the wrist dragged 0.5 rad, where IsaacLab on the same asset settles in 0.4 s with
        nothing else moving (Slurm 204609). Authored values are read from the robot's source USD by link
        name and pushed to the physics view; links that author no ``physics:centerOfMass`` -- every DROID
        link -- are left exactly as OmniGibson made them, so DROID is untouched bit-for-bit.

        The primary fix is in the asset: scripts/build_yam_usd.py puts every collision Mesh directly under
        its link (``flatten_collision_xforms``), which makes the loader's composition exact, so on a
        current yam*.usd this finds nothing to restore and guards the next asset. It must run AFTER
        rebase_initial_file(), which re-initialises the prims and re-applies the override (Slurm 204613:
        a restore in bind_scene_handles had no effect; 204615: the live write is honoured and gives
        IsaacLab's step response).
        """
        stage = lazy.pxr.Usd.Stage.Open(self.robot.usd_path)
        root = stage.GetDefaultPrim().GetPath()
        restored, authored_n = [], 0
        for name, link in self.robot.links.items():
            prim = stage.GetPrimAtPath(f"{root}/{name}")
            attr = prim.GetAttribute("physics:centerOfMass") if prim else None
            if not attr or not attr.HasAuthoredValue():
                continue
            authored_n += 1
            authored = th.tensor(list(attr.Get()), dtype=th.float32)
            loaded = link.center_of_mass.to(th.float32)
            if th.allclose(loaded, authored, atol=1e-6):
                continue
            link.center_of_mass = authored
            after = link.center_of_mass.to(th.float32)
            restored.append(f"{name}: |loaded-authored| {float(th.linalg.norm(loaded - authored)):.3f} m, "
                            f"readback err {float(th.linalg.norm(after - authored)):.4f} m")
        # og.log.info is not emitted in REALM's logs; this is a physics correction, so say it loudly.
        self.link_com_restore_report = restored
        if authored_n:
            og.log.warning(f"[link CoM] {self.robot.name}: {authored_n} link(s) author physics:centerOfMass in "
                           f"{self.robot.usd_path}; restored {len(restored)} that OmniGibson's loader had "
                           f"displaced: {'; '.join(restored) if restored else 'none needed'}")

    def _author_arm_joint_physics(self, joint_names, friction, armature):
        for idx in range(7):
            prim_path = f"{self.robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
            joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            assert joint_prim.IsValid(), f"no joint prim at {prim_path}"
            # Create the attributes if the asset never authored them -- droid.usd ships them, the
            # robolab asset does not, and GetAttribute(...).Set() on a missing attribute is a silent
            # no-op, which would leave armature at zero without any error.
            lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            for attr_name, value in (("physxJoint:jointFriction", friction[idx]),
                                     ("physxJoint:armature", armature[idx])):
                attr = joint_prim.GetAttribute(attr_name)
                if not attr:
                    attr = joint_prim.CreateAttribute(attr_name, lazy.pxr.Sdf.ValueTypeNames.Float)
                attr.Set(float(value))

    def _convexify_dynamic_collision_meshes(self):
        for link_name, link in self.robot.links.items():
            for collision_mesh in link.collision_meshes.values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                if prim.IsValid() and prim.HasAttribute("physxMeshCollision:approximation"):
                    approx = prim.GetAttribute("physxMeshCollision:approximation").Get()
                    if approx in ["none", "meshSimplification"]:
                        prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")

    def restore_double_duty_render_purpose(self):
        """Restore render purpose when OmniGibson hides double-duty collision meshes.

        Dedicated visual meshes are left untouched. Asset-authored ``guide`` opinions are also
        preserved; PXR's property stack distinguishes them from OG's anonymous runtime override.
        """
        Usd, UsdGeom = lazy.pxr.Usd, lazy.pxr.UsdGeom

        def asset_authored_purpose(prim):

            attr = UsdGeom.Imageable(prim).GetPurposeAttr()
            if not attr:
                return None
            try:
                for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
                    if spec.layer is not None and not spec.layer.anonymous:
                        return str(spec.default)
            except Exception:
                return None
            return None

        restored, kept_guide = [], []
        objects = list(self.main_objects) + list(self.target_objects) + list(self.distractors)
        with og.sim.editing_usd():
            for obj in objects:
                for link_name, link in (getattr(obj, "links", None) or {}).items():
                    if link.visual_meshes:
                        continue
                    for mesh in link.collision_meshes.values():
                        prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(mesh.prim_path)
                        if not prim.IsValid():
                            continue
                        authored = asset_authored_purpose(prim)
                        if authored == UsdGeom.Tokens.guide:
                            kept_guide.append(mesh.prim_path)
                            continue
                        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(
                            authored or UsdGeom.Tokens.default_)
                        restored.append(mesh.prim_path)
        og.log.debug(
            f"Restored {len(restored)} double-duty geom purpose(s); kept "
            f"{len(kept_guide)} authored-guide geom(s) hidden"
        )
        return restored, kept_guide

    def apply_scene_fixes_from_cfg(self, manage_sim_state=True):

        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)

        if self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]:
            scene_data = spawn_cfg[self.scene_model][self.scene_part]
            if manage_sim_state:
                og.sim.stop()
            for obj in self.omnigibson_env.scene.objects:
                if obj.name in scene_data.get("to_fix", []):
                    obj.fixed_base = True
                    create_joint(
                        prim_path=f"{obj.prim_path}/rootJoint",
                        joint_type="FixedJoint",
                        body1=f"{obj.prim_path}/{obj._root_link_name}",
                    )
                elif obj.name in scene_data.get("to_remove", []):
                    obj_to_remove = self.omnigibson_env.scene.object_registry("name", obj.name)
                    self.omnigibson_env.scene.remove_object(obj_to_remove)

            if manage_sim_state:
                og.sim.play()

    def rebase_initial_file(self):

        self.omnigibson_env.scene.update_initial_file()

    def place_wrist_cameras(self):
        """Apply the robot config's REALM-only ``wrist_camera_pose`` (see env_config) to every wrist camera:
        each ``<robot>:<link>:Camera:<i>`` sensor gets ``set_local_pose(pos, quat_xyzw)`` in its parent link's
        frame. No-op when the key is absent (every DROID config, and the YAM configs that keep the USD pose)."""
        pose = getattr(self, "robot_wrist_camera_pose", None)
        if not pose:
            return
        pos = np.asarray(pose["pos"], dtype=float)
        w, x, y, z = pose["quat_wxyz"]
        quat_xyzw = np.array([x, y, z, w], dtype=float)
        cameras = {k: s for k, s in self.robot.sensors.items() if ":Camera:" in k}
        assert cameras, "wrist_camera_pose set but the robot has no Camera sensors"
        for key, sensor in cameras.items():
            sensor.set_local_pose(pos, quat_xyzw)
            og.log.info(f"[wrist_camera_pose] {key}: local pose set to {pos.tolist()} / wxyz {list(pose['quat_wxyz'])}")

    def disable_visual_toggles(self):
        for obj in self.omnigibson_env.scene.objects:
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False
