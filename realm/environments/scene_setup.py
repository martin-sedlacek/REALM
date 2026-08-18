"""Fixes applied to a freshly loaded scene and robot, before any rollout runs.

Each of these runs exactly once per environment, between ``og.Environment`` building the scene and
the first ``reset()``. ``RealmEnvironmentDynamic.post_play_setup()`` orders them for a single env;
``RealmVectorEnvironment.__init__`` drives the same ones itself so it can batch the global stop/play
cycle ``apply_scene_fixes_from_cfg`` needs into one cycle for all members.
"""
import numpy as np
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import create_joint


class SceneSetupMixin:
    """Post-load corrections to the scene and the robot.

    Expects the host to provide ``config_path``, ``scene_model``, ``scene_part``, ``robot_name``,
    ``robot``, ``cfg`` and ``omnigibson_env``.
    """

    def update_robot_physics(self):
        """Author the arm's joint friction/armature and make its collision meshes convex.

        Matches on the DROID PREFIX, not on the config literally named "DROID". The robolab asset
        was silently skipped here, so its arm ran with zero armature -- no rotor inertia against a
        stiff impedance law -- and the wrist would not hold a commanded pose.
        """
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
        """Write config/robots/<robot>.yaml's friction and armature onto the seven arm joints."""
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
        """Triangle-mesh collision approximations are not valid for dynamic bodies."""
        for link_name, link in self.robot.links.items():
            for collision_mesh in link.collision_meshes.values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                if prim.IsValid() and prim.HasAttribute("physxMeshCollision:approximation"):
                    approx = prim.GetAttribute("physxMeshCollision:approximation").Get()
                    if approx in ["none", "meshSimplification"]:
                        prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")

    def restore_double_duty_render_purpose(self):
        """Un-hide geometry that is BOTH the collider and the visual mesh.

        WHAT BREAKS. ``RigidPrim._post_load`` classifies every geom under a link as collision or
        visual, and writes ``purpose = "guide"`` on the collision ones
        (``omnigibson/prims/rigid_prim.py:242``). ``guide`` never reaches the colour pass, which is
        correct for a dedicated collider sitting next to a separate visual mesh -- the shape every
        BEHAVIOR asset has, because ``convert_urdf_to_usd`` emits ``visuals/`` and ``collisions/``
        Xforms per link.

        REALM's own assets do not have that shape. ``custom_assets/impact_drawer/usd/cabinet.usd``
        applies ``UsdPhysics.CollisionAPI`` directly to its render meshes -- one mesh does both jobs
        -- so OmniGibson guides the only geometry the link has and the object becomes INVISIBLE
        while staying fully physical. Measured on og391, task 8, over the live stage: all 56 geoms
        `guide`, the ``default``+``render`` purpose world bound EMPTY, the ``guide`` bound
        (0.512, 0.669, 0.545); hiding the cabinet changed 27-182 px (the rt noise floor) where
        hiding the breakfast table changed 245,182; and with every other object hidden the cabinet
        still contributed 0 px on all three cameras.

        WHY 1.1.1 DREW IT ANYWAY, i.e. why this is a port regression and not a long-standing bug.
        Two independent changes, and this asset is hit by both:

          * DEPTH. 1.1.1 scanned exactly two levels below each link. 55 of the 56 geoms -- including
            all five drawer render meshes at ``drawer_blender_cut_0N/ObjectCapture/Geometry/Mesh/
            Mesh`` -- sit FOUR levels down, so 1.1.1 never visited them and never wrote a purpose.
            og391's ``_find_geom_prims`` recurses without a depth limit and reaches them.
          * INHERITANCE. og391 makes its ``is_collision`` flag sticky down the recursion, so a
            CollisionAPI on an ancestor Xform marks every mesh beneath it. That, not its own API, is
            what guides the cabinet BODY (``base_link/Geometry_01/Object_Geometry_02``, which has no
            CollisionAPI of its own and inherits from ``base_link``).

        THE CRITERION, and why it cannot un-hide a real collider. A link is only touched when
        OmniGibson left it with NO visual meshes at all -- meaning every geom it has was classified
        collision, so the link would render nothing and its colliders must be doing double duty.
        Any asset with dedicated visual geometry has a non-empty ``visual_meshes`` and is skipped
        outright, so this is a no-op on every BEHAVIOR object and on the robot. Within a qualifying
        link, a geom whose OWN layer authored ``guide`` keeps it: the asset's ``collider_guide`` /
        ``Cube`` helpers were meant to be hidden and stay hidden. The authored opinion is read off
        the property stack's first non-anonymous layer, so it survives OmniGibson having already
        written ``guide`` into the anonymous session layer above it -- and it is the asset's own
        value that gets restored, not a hardcoded ``default`` (the drawer handles author ``render``).

        RENDER-ONLY. ``purpose`` is a ``UsdGeom.Imageable`` attribute. The CollisionAPI, the
        contact/rest offsets, the collision approximation and the link centre-of-mass are all left
        exactly as OmniGibson set them; this writes nothing else. Verified bitwise: joint positions,
        joint velocities and every link pose of both the cabinet and the robot are identical across
        the writes, with no simulation step in between.
        """
        Usd, UsdGeom = lazy.pxr.Usd, lazy.pxr.UsdGeom

        def asset_authored_purpose(prim):
            """The `purpose` opinion from the asset's own (non-anonymous) layer, or None.

            OmniGibson's runtime write lands in an anonymous session layer, so a plain
            `GetPurposeAttr().Get()` returns `guide` for everything and cannot distinguish "the
            author hid this" from "OmniGibson hid this". The property stack still carries both.
            """
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
                    # Dedicated visual geometry exists -> OmniGibson's guiding is right. Skip.
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
        # One line per env, so a run's log says whether this fired and on what. Silence here would
        # make a regression (or an accidental no-op) invisible.
        print(f"[render_purpose] restored {len(restored)} double-duty geom(s) to their authored "
              f"purpose, kept {len(kept_guide)} authored-guide geom(s) hidden"
              + (f"; first restored: {restored[0]}" if restored else ""))
        return restored, kept_guide

    def apply_scene_fixes_from_cfg(self, manage_sim_state=True):
        """Pin or delete the scene objects that config/scenes/scenes.yaml lists for this scene part.

        Adding and removing objects needs a STOPPED sim. @manage_sim_state=False leaves the cycle to
        the caller, which is how RealmVectorEnvironment runs one cycle for all members instead of N.
        """
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
        """Make the scene AS FIXED the one reset() restores.

        og.Environment.post_play_load() captures scene._initial_file BEFORE
        apply_scene_fixes_from_cfg ever runs, so it still lists every object the scene config asked
        to REMOVE and the first reset undoes the removal. Re-capturing here makes the fixed scene
        the baseline, so restore() has nothing to add -- which also removes a per-member stop/play
        cycle from a vector env's first reset. Measured, and the one-reset pose transient that made
        the bug look member-dependent, are in docs/vector_env/README.md under "Second, independent
        bug: reset re-adds the removed chair".

        Separate from apply_scene_fixes_from_cfg's body rather than at its tail because Scene.save()
        asserts a non-stopped sim (it dumps joint state) -- the fixes themselves run stopped, and a
        vector env runs them for every member inside ONE stopped window and plays once afterwards.
        """
        self.omnigibson_env.scene.update_initial_file()

    def disable_visual_toggles(self):
        # TODO: (martin) for pre-baked OG switches on walls their rotation seems off so we cannot use those without the visual toggle...
        for obj in self.omnigibson_env.scene.objects:
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False
