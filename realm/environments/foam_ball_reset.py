"""pour_proxy: get N rigid foam balls to start each episode INSIDE the source bottle.

The pour task's success signal is "balls left the source and are now Inside the target". Fluid
particles would be the honest way to do that, but they need `USE_GPU_DYNAMICS` and
`ENABLE_HQ_RENDERING` on and `ENABLE_VISUAL_UPDATES` not off -- the exact opposite of the perf
defaults `realm/sim_config.py` now sets -- so `pour_liquid` stays unported and the proxy carries the
task. See `TaskProgressionMixin.check_pour` for the liquid checker that is kept but inert.

WHY THIS IS NOT A FEW LINES. Spawning a ball inside a bottle's collision geometry does not work:
the default convexHull approximation makes the bottle a solid blob, and even with a hollow
convexDecomposition PhysX resolves any spawn-frame penetration by ejecting the ball, which kicks
the bottle over. So the sequence is: hollow the bottle, make it heavy and bottom-heavy, settle it
ALONE with the balls parked out of the way, then drop the balls down its neck and check it is still
upright -- re-rolling the XY jitter if a ball squeezed through a decomposition seam and tipped it.
Finally `update_initial_file()` snapshots balls-in-bottle as the reset target, so the per-episode
cost is zero: `omnigibson_env.reset()` restores it.

WHY IT LIVES IN ITS OWN MODULE, batched, mirroring realm/environments/joint_reset.py. Every step in
that sequence is an `og.sim.step()`, and `og.sim.step()` advances EVERY scene in a vector env. A
per-member "settle, check, retry up to 8 times" loop would therefore step other members 8 times
over while they wait, and the pre-port branch's version -- which called `og.sim.stop()`,
`og.sim.play()` and `og.sim.step()` directly -- is exactly the pattern `_helpers.sim_stop`/`sim_play`
replaced for that reason. Here the retries run in LOCKSTEP instead: every member re-rolls, then ONE
shared batch of steps advances all of them, then each member judges its own bottle and drops out
when it is upright. Total simulator steps are therefore a function of the worst member, not of the
sum over members, and no member's retry perturbs another's settled scene.
"""
from collections import namedtuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import macros

from realm.environments.perturbations._helpers import sim_play, sim_stop
from realm.config.shared import (
    FOAM_BALL_COUNT,
    FOAM_BALL_DIAMETER,
    FOAM_BALL_MASS_KG,
    FOAM_BALL_MAX_ATTEMPTS,
    FOAM_BALL_PLACE_STEPS,
    FOAM_BALL_SOURCE_SETTLE_STEPS,
    FOAM_BALL_STASH_SCENE_Z,
    FOAM_BALL_UPRIGHT_TOLERANCE_DEG,
    FOAM_BALL_XY_JITTER,
    FOAM_BALL_Z_LIFT,
    POUR_SOURCE_COM_DROP_FRACTION,
    POUR_SOURCE_MASS_KG,
)

#: Task type this whole module exists for. `pour_liquid` is deliberately absent -- see the docstring.
POUR_PROXY_TASK_TYPE = "pour_proxy"

FoamBallPlan = namedtuple("FoamBallPlan", ["source", "balls"])
FoamBallPlan.__doc__ = """One member's placement work, excluding global simulator steps."""


def foam_ball_cfgs(cfg, mo_cfgs):
    """Object configs for the balls a pour_proxy task needs, or [] for any other task.

    Injected AFTER placement has run (see RealmEnvironmentDynamic), so they never enter
    `placement.place_within`'s random-placement bucket and never displace a distractor. The spawn
    positions here are provisional: `run_foam_ball_placements` parks and re-places every ball before
    anything settles. They are stacked more than a diameter apart so PhysX's contact offset does not
    report adjacent balls as touching on the spawn frame.
    """
    if cfg.get("task_type") != POUR_PROXY_TASK_TYPE:
        return []

    n_balls = int(cfg.get("n_foam_balls", FOAM_BALL_COUNT))
    source_pos = mo_cfgs[0]["position"]
    z_spacing = FOAM_BALL_DIAMETER * 1.5
    base_z = float(source_pos[2]) + FOAM_BALL_Z_LIFT

    return [
        {
            "type": "PrimitiveObject",
            "name": f"foam_ball_{i}",
            "primitive_type": "Sphere",
            "rgba": [0.85, 0.1, 0.1, 1.0],
            "bounding_box": [FOAM_BALL_DIAMETER] * 3,
            "scale": [FOAM_BALL_DIAMETER] * 3,
            "position": [float(source_pos[0]), float(source_pos[1]), base_z + i * z_spacing],
        }
        for i in range(n_balls)
    ]


def refresh_foam_ball_cfg_positions(env, cfgs):
    """Point each foam ball's cfg at where the ball actually IS, in scene frame.

    Any perturbation that ends with `set_scene_positions` writes every cfg position back to the
    scene, and a ball's cfg still holds the provisional spawn column from foam_ball_cfgs -- so
    without this the balls get teleported out of the settled bottle and spill. Call it before the
    placement pass in any perturbation that re-places objects.
    """
    if not env.foam_ball_names:
        return
    by_name = {name: env.omnigibson_env.scene.object_registry("name", name)
               for name in env.foam_ball_names}
    for cfg in cfgs:
        ball = by_name.get(cfg["name"])
        if ball is not None:
            cfg["position"] = ball.get_position_orientation(frame="scene")[0].tolist()


def _find_collider_meshes(prim):
    """Meshes under @prim that actually take part in collision.

    A mesh counts if it has a collision API applied or sits under an Xform whose name says
    "collider(s)". Visual-only and guide meshes contribute no physics, so distributing mass over
    them would put the source's centre of mass wherever the artist happened to put geometry.
    """
    found = []
    stack = [prim]
    while stack:
        current = stack.pop()
        for child in current.GetChildren():
            stack.append(child)
        if not current.IsA(lazy.pxr.UsdGeom.Mesh):
            continue
        has_api = current.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)
        under_collider = any(
            "collider" in ancestor.GetName().lower()
            for ancestor in [current.GetParent()] if ancestor and ancestor.IsValid()
        )
        if has_api or under_collider:
            found.append(current)
    return found


def set_object_mass_via_colliders(obj, total_mass_kg):
    """Spread @total_mass_kg over @obj's collider meshes, authoring MassAPI on each.

    Per-shape mass lets PhysX derive the body's inertia from where the collision shapes actually
    are in body-local space, which is what makes a bottle whose colliders cluster at the base
    behave bottom-heavy. Falls back to an equal split across links when no collider mesh is found.
    """
    meshes = _find_collider_meshes(obj.prim)
    if not meshes:
        n_links = max(len(obj._links), 1)
        for link in obj._links.values():
            link.mass = total_mass_kg / n_links
        og.log.info(f"[pour_proxy] {obj.name}: no collider meshes; split mass across {n_links} link(s)")
        return

    per_mesh = total_mass_kg / len(meshes)
    # Authoring MassAPI is a raw USD edit, and 3.9.1 guards those: a Tf.Notice listener records
    # any change made outside this context and re-raises it at the next entry to one, so an
    # unwrapped edit here surfaces as a RuntimeError from an unrelated later call.
    with og.sim.editing_usd():
        for mesh in meshes:
            mass_api = lazy.pxr.UsdPhysics.MassAPI.Apply(mesh)
            if not mass_api.GetMassAttr():
                mass_api.CreateMassAttr()
            mass_api.GetMassAttr().Set(float(per_mesh))
    # link.mass too, so PhysX has a consistent total to reference. OUTSIDE the block above: the
    # setter opens its own editing_usd(), and nesting them is an assertion error.
    for link in obj._links.values():
        link.mass = total_mass_kg / max(len(obj._links), 1)
    og.log.info(f"[pour_proxy] {obj.name}: {total_mass_kg} kg over {len(meshes)} collider mesh(es)")


def prepare_pour_proxy_physics(env):
    """Stopped-simulator half of the setup: hollow the source, then set the masses.

    Both of these need the simulator stopped. sim_stop/sim_play rather than og.sim.stop()/play()
    because those are global: the vector build already stopped the sim once for every member before
    it calls in here, and stopping again per member would thrash the whole simulator. Outside a
    vector env they do the real stop/play around this work.
    """
    if env.task_type != POUR_PROXY_TASK_TYPE:
        return

    sim_stop(env)
    for obj in env.main_objects:
        for link in obj._links.values():
            # convexHull would make the bottle a solid blob with no interior to pour from.
            #
            # On the link, not on link.collision_meshes.values(): in OG 1.1.1 (where the pouring
            # branch was written) the setter lived on each CollisionGeomPrim, and in 3.9.1 it moved
            # up to RigidPrim.set_collision_approximation, which applies it across that link's mesh
            # collision APIs. The 1.1.1 call raises AttributeError on a 3.9.1 GeomPrim.
            link.set_collision_approximation("convexDecomposition")
        set_object_mass_via_colliders(obj, POUR_SOURCE_MASS_KG)

    for ball in env.foam_balls:
        n_links = max(len(ball._links), 1)
        for link in ball._links.values():
            link.mass = FOAM_BALL_MASS_KG / n_links
    sim_play(env)


def run_foam_ball_placements(envs):
    """Place every pending member's balls inside its source, in lockstep. See module docstring."""

    pending = [env for env in envs if env.pending_foam_placement is not None]
    if not pending:
        return

    # HEADLESS does not disable rendering during simulator steps.
    with og.sim.render_on_step(False):
        # Park the balls in each member's OWN scene frame, high above it. Scene-frame, because a
        # shared world-space parking spot would pile every member's balls into one heap and they
        # would resolve that penetration against each other. Same idiom as
        # placement.get_default_objects_cfg.
        for env in pending:
            for i, ball in enumerate(env.pending_foam_placement.balls):
                ball.set_position_orientation(
                    position=[0.0, 0.0, FOAM_BALL_STASH_SCENE_Z + i * 0.1], frame="scene")
                ball.keep_still()

        # With the balls out of the way, every source settles alone -- so the pose captured next is
        # genuinely upright rather than one already nudged by an ejected ball.
        for _ in range(FOAM_BALL_SOURCE_SETTLE_STEPS):
            og.sim.step()
        for env in pending:
            env.capture_pristine_source_pose()

        unresolved = list(pending)
        for attempt in range(1, FOAM_BALL_MAX_ATTEMPTS + 1):
            for env in unresolved:
                env.reroll_foam_balls()
            for _ in range(FOAM_BALL_PLACE_STEPS):
                og.sim.step()
            still_tipped = [env for env in unresolved if not env.is_source_upright()]
            settled = len(unresolved) - len(still_tipped)
            if settled:
                og.log.info(f"[pour_proxy] {settled} source(s) upright with balls inside on "
                            f"attempt {attempt}/{FOAM_BALL_MAX_ATTEMPTS}")
            unresolved = still_tipped
            if not unresolved:
                break
            og.log.info(f"[pour_proxy] attempt {attempt}/{FOAM_BALL_MAX_ATTEMPTS} tipped "
                        f"{len(unresolved)} source(s); re-rolling jitter")

        # Judge every member on its FINAL state, not on the attempt it first passed: the shared
        # step batches keep running for members that already succeeded, so one could tip later.
        tipped = [env for env in pending if not env.is_source_upright()]
        if tipped:
            og.log.warning(
                f"[pour_proxy] {len(tipped)} of {len(pending)} source(s) still not upright after "
                f"{FOAM_BALL_MAX_ATTEMPTS} attempts -- those episodes start from a tipped bottle")

        for env in pending:
            env.make_source_bottom_heavy()
            # The settle may have drifted the source within tolerance; snapshot the canonical pose
            # rather than a borderline one.
            env.restore_pristine_source_pose()
            # Reset the arm too, so the snapshot does not bake in drifted joint positions and stale
            # controller targets -- that is how a gripper ends up stuck closed against warmup's
            # open command.
            env.robot.set_joint_positions(torch.tensor(env.reset_qpos, dtype=torch.float32))
            env.robot.keep_still()

        og.sim.step()  # let the pose writes and controller targets take effect

        for env in pending:
            env.omnigibson_env.scene.update_initial_file()
            env.refresh_foam_ball_init_poses()
            env.pending_foam_placement = None
            og.log.info(f"[pour_proxy] initial file recaptured with "
                        f"{len(env.foam_balls)} ball(s) inside the settled source")


class FoamBallMixin:
    """Per-member half of pour_proxy setup. The global stepping lives in run_foam_ball_placements."""

    pending_foam_placement = None
    foam_balls = ()
    _source_has_fillable_volume = False
    _pristine_source_pos = None
    _pristine_source_quat = None
    _initial_balls_in_source = None

    def bind_foam_balls(self):
        """Capture the balls once, by name, for the env's lifetime.

        Not read off `self.distractors` at use time: V-SC rebuilds that list, and the balls are
        deliberately excluded from what V-SC treats as clutter, so a later scan would come up empty
        and silently break success detection.
        """
        self.foam_balls = [obj for obj in self.distractors
                           if obj is not None and obj.name.startswith("foam_ball_")]
        self._source_has_fillable_volume = bool(self.foam_balls) and _has_container_volume(
            self.main_objects[0])

    @property
    def foam_ball_names(self):
        return tuple(ball.name for ball in self.foam_balls)

    def place_foam_balls(self):
        """Register this member's placement. Runs it immediately outside a vector env."""
        if self.task_type != POUR_PROXY_TASK_TYPE or not self.foam_balls:
            return
        assert len(self.target_objects) == 1, "pour_proxy needs exactly one target object"
        self.pending_foam_placement = FoamBallPlan(source=self.main_objects[0], balls=self.foam_balls)
        if not self.in_vec_env:
            run_foam_ball_placements([self])

    def capture_pristine_source_pose(self):
        pos, quat = self.main_objects[0].get_position_orientation()
        self._pristine_source_pos = np.asarray(_to_numpy(pos), dtype=float)
        self._pristine_source_quat = np.asarray(_to_numpy(quat), dtype=float)

    def restore_pristine_source_pose(self):
        source = self.main_objects[0]
        source.set_position_orientation(position=self._pristine_source_pos.tolist(),
                                        orientation=self._pristine_source_quat.tolist())
        source.keep_still()

    def reroll_foam_balls(self):
        """One attempt: source back to pristine, then balls down its neck with fresh XY jitter."""
        self.restore_pristine_source_pose()
        pos = self._pristine_source_pos
        z_spacing = FOAM_BALL_DIAMETER * 1.5
        for i, ball in enumerate(self.foam_balls):
            ball.set_position_orientation(position=[
                float(pos[0]) + np.random.uniform(-FOAM_BALL_XY_JITTER, FOAM_BALL_XY_JITTER),
                float(pos[1]) + np.random.uniform(-FOAM_BALL_XY_JITTER, FOAM_BALL_XY_JITTER),
                float(pos[2]) + FOAM_BALL_Z_LIFT + i * z_spacing,
            ])
            ball.keep_still()

    def is_source_upright(self, tolerance_deg=FOAM_BALL_UPRIGHT_TOLERANCE_DEG):
        """Is the source still within @tolerance_deg of the tilt it had when it settled alone?

        Compares against the pristine orientation rather than assuming the body's own Z axis points
        up: some custom bottle USDs carry an authored xform that rotates the body axes. Tilt only --
        rotation about the world vertical (VB-POSE's yaw noise, say) is not tipping.
        """
        if not self.main_objects or self._pristine_source_quat is None:
            return True
        _, quat = self.main_objects[0].get_position_orientation()
        quat = np.asarray(_to_numpy(quat), dtype=float)
        world_up = np.array([0.0, 0.0, 1.0])
        body_up = R.from_quat(self._pristine_source_quat).inv().apply(world_up)
        body_up_now = R.from_quat(quat).apply(body_up)
        cos_tilt = float(np.dot(body_up_now, world_up))
        return cos_tilt >= float(np.cos(np.deg2rad(tolerance_deg)))

    def make_source_bottom_heavy(self):
        """Drop the source's centre of mass toward its base so it resists tipping while poured.

        The offset is meant as "down in the world", so it is expressed in world terms and rotated
        into the link's local frame through the pristine orientation -- otherwise it would point
        somewhere else on any asset whose link xform is rotated.
        """
        source = self.main_objects[0]
        try:
            aabb_min, aabb_max = source.aabb
            z_extent = float(_to_numpy(aabb_max)[2] - _to_numpy(aabb_min)[2])
            drop_world = np.array([0.0, 0.0, -POUR_SOURCE_COM_DROP_FRACTION * (z_extent / 2.0)])
            offset_body = R.from_quat(self._pristine_source_quat).inv().apply(drop_world)
            with og.sim.editing_usd():
                mass_api = lazy.pxr.UsdPhysics.MassAPI.Apply(source.root_link.prim)
                if not mass_api.GetCenterOfMassAttr():
                    mass_api.CreateCenterOfMassAttr()
                mass_api.GetCenterOfMassAttr().Set(
                    lazy.pxr.Gf.Vec3f(*(float(v) for v in offset_body)))
        except Exception as e:
            # Non-fatal: the bottle is merely easier to tip. Loud, because that shows up as a
            # success-rate difference and not as a crash.
            og.log.warning(f"[pour_proxy] could not lower {source.name}'s centre of mass: {e}")

    def refresh_foam_ball_init_poses(self):
        """Point init_poses at where the balls ended up, not where they were spawned."""
        for ball in self.foam_balls:
            pos, quat = ball.get_position_orientation()
            self.init_poses[ball._relative_prim_path] = {"pos": pos, "rot": quat}

    def count_balls_inside(self, container):
        """Balls Inside @container, per OmniGibson's own state.

        Correct for the TARGET: a dataset receptacle carries a `fillable` meta link, which is the
        volume `Inside` tests points against. Use count_balls_in_source for the source instead --
        see that method for why this one cannot answer for it.
        """
        n = 0
        for ball in self.foam_balls:
            try:
                if bool(ball.states[og.object_states.Inside].get_value(container)):
                    n += 1
            except Exception:
                # Inside is a kinematic state evaluated on demand; a ball mid-flight between
                # containers can raise rather than answer. Treat that as "not inside".
                pass
        return n

    def count_balls_in_source(self):
        """Balls still in the source bottle.

        NOT `Inside`, unless the asset can support it. OG 3.9.1's `Inside._get_value` gates on the
        outer object having a link whose meta_link_type is in CONTAINER_META_LINK_TYPES
        (`fillable`/`openfillable`) and testing the point against THAT volume. The custom bottle
        USDs this task ships have exactly one link and no meta links, so `Inside(bottle)` is
        structurally False no matter where the ball is -- measured 2026-09-04: 15 of 15 balls
        demonstrably within the bottle's AABB, `Inside` returning 0 for every one. OG 1.1.1, where
        this task was authored, answered the same question by raycasting and so needed no meta link;
        that is the behaviour being restored here, not a threshold being loosened.

        The bounding-box test is sound for THIS role: it is only ever asked to notice that a ball
        LEFT the bottle, and a ball that has left is outside the bottle's box. The target side keeps
        the real `Inside` state, which is the half that decides success.
        """
        source = self.main_objects[0]
        if self._source_has_fillable_volume:
            return self.count_balls_inside(source)
        return sum(1 for ball in self.foam_balls if _centre_within_aabb(ball, source))

    def capture_foam_ball_reference(self):
        """How many balls the episode STARTS with in the source.

        check_pour_proxy needs it: balls arriving in the target only counts as pouring if they came
        out of the source, which a target-side count alone cannot tell you.
        """
        if self.task_type != POUR_PROXY_TASK_TYPE or not self.foam_balls:
            return
        self._initial_balls_in_source = self.count_balls_in_source()


def _to_numpy(value):
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _has_container_volume(obj):
    """Does @obj carry the meta link `Inside` needs to test containment against?"""
    container_types = macros.object_states.contains.CONTAINER_META_LINK_TYPES
    return any(getattr(link, "is_meta_link", False) and link.meta_link_type in container_types
               for link in obj.links.values())


def _centre_within_aabb(inner, outer):
    inner_lo, inner_hi = (_to_numpy(v) for v in inner.aabb)
    outer_lo, outer_hi = (_to_numpy(v) for v in outer.aabb)
    centre = (inner_lo + inner_hi) / 2.0
    return bool(np.all(centre >= outer_lo) and np.all(centre <= outer_hi))
