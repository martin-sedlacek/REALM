
import numpy as np
import yaml

import omnigibson as og

from realm.environments.foam_ball_reset import prepare_pour_proxy_physics
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

        # Hollowing the source and authoring its masses needs the simulator stopped, and this is
        # the one setup hook both paths enter in that state: the vector build stops once for every
        # member and calls in with manage_sim_state=False.
        prepare_pour_proxy_physics(self)

    def rebase_initial_file(self):

        self.omnigibson_env.scene.update_initial_file()

    def disable_visual_toggles(self):
        for obj in self.omnigibson_env.scene.objects:
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False
