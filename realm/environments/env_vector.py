"""Vectorized REALM environments: N scenes in one simulator, stepped together.

OmniGibson 3.9.1 loads several scenes into one physics stage, tiled side by side with a margin
(Simulator.import_scene -> Scene.load(idx=..., last_scene_edge=...)). One og.sim.step() advances
every scene, so members must not be stepped individually -- doing so would advance all scenes while
applying only one member's action. That is what this class exists to prevent; it mirrors
OmniGibson's own VectorEnvironment, with REALM's per-member setup and task bookkeeping layered on.

Construction is three-phase because og.sim.play() and og.sim.stop() are global:

    1. build every member with in_vec_env=True   (loads scenes, does not play)
    2. og.sim.play() once, then post_play_load() + bind_scene_handles() per member
    3. one stop/play cycle around apply_scene_fixes_from_cfg() for all members, then finalize

Poses stay scene-relative throughout: REALM's external cameras use pose_frame "parent" and are
loaded into their own env's scene, and robots default to frame "scene", so each member's camera
extrinsics and spawn pose land correctly inside its own tile without knowing the tile offset.
"""
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy

from realm.environments.env_base import run_joint_resets
from realm.environments.env_dynamic import RealmEnvironmentDynamic, WARMUP_STEPS
from realm.environments.perturbations._helpers import (
    NEEDS_STOPPED_SIM,
    SETTLE_STEPS,
    settle_action,
)


class RealmVectorEnvironment:
    """N RealmEnvironmentDynamic members sharing one simulator.

    The members are independent scenes running the same task config; per-member variation comes
    from the sampling done while building each member's config (object placement is drawn per
    member), not from different task files.
    """

    def __init__(self, num_envs, on_first_env_built=None, **env_kwargs):
        """
        Args:
            num_envs (int): number of parallel environments
            on_first_env_built (None or callable): invoked once, right after member 0 exists.
                Isaac is not running until the first og.Environment is created, so carb settings
                cannot be touched before then -- but they must be set BEFORE the remaining scenes
                load if they are to affect how the renderer sizes its pools. That is the window
                this hook exists for: at 16 members the RTX descriptor/parameter-block pools run
                out while loading scene 10 ("Unable to allocate descriptor sets") and the process
                segfaults, long before GPU memory is a concern. Passing a lighter renderer profile
                here is what makes higher member counts reachable.
            **env_kwargs: forwarded verbatim to each RealmEnvironmentDynamic
        """
        assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
        self.num_envs = num_envs

        # OmniGibson requires a stopped sim to import a scene; a previous run may have left it playing.
        if og.sim is not None:
            og.sim.stop()

        og.log.info(f"Loading {num_envs} parallel environments...")
        self.envs = []
        for i in range(num_envs):
            og.log.info(f"  environment {i + 1}/{num_envs}")
            self.envs.append(RealmEnvironmentDynamic(in_vec_env=True, **env_kwargs))
            if i == 0 and on_first_env_built is not None:
                on_first_env_built()

        # Play once for every scene, then let each member finish the loading that needs a live sim.
        og.sim.play()
        for env in self.envs:
            env.omnigibson_env.post_play_load()
        for env in self.envs:
            env.bind_scene_handles()

        # Scene fixes stop and play the sim, so batch them: one cycle for all members instead of one
        # per member. Ordering within a member is unchanged from the single-env path.
        og.sim.stop()
        for env in self.envs:
            env.apply_scene_fixes_from_cfg(manage_sim_state=False)
        og.sim.play()

        for env in self.envs:
            env.finalize_setup()
        # finalize_setup() -> RealmEnvironmentBase.__init__ -> reset_joints(), which in a vector env
        # only records the plan. Drain it here so construction leaves the drawers where a single env
        # would, on one shared step loop rather than one per member.
        self._drain_joint_resets()
        og.log.info(f"{num_envs} environments ready.")

    # ============================== [ROLLOUT] ==============================
    def reset(self):
        """Reset every member, doing each GLOBAL simulator operation exactly once.

        A member's reset is not self-contained: perturbations stop, play and step the simulator, and
        every one of those acts on ALL scenes. Running `[env.reset() for env in self.envs]` therefore
        had each member tear down and rebuild its siblings' scenes mid-reset. Measured (job 190555,
        VB-POSE Vec=4): the main object dropped out of the contact view for scenes 1, 2 and 3, so 18
        of 25 rollouts logged zero environment collisions and never left REACH -- and the job still
        exited 0.

        So the per-member work is split into phases and the global operations are hoisted out:

            1. every member restores its own scene                    (no global state touched)
            2. ONE joint-reset loop for every member that asked for one (drawer tasks only)
            3. ONE og.sim.stop(), if any member's perturbation needs it
            4. every member's perturbations run
            5. ONE og.sim.play(), then repair of the sim's object-init queue
            6. work the perturbations deferred because it needs a playing sim
            7. ONE joint-reset loop again, for the perturbations that ask for one
            8. ONE settle loop driving all members together, if any asked for it
            9. every member re-takes its main-object scoring reference

        Mirrors the batching already used for scene fixes in __init__ and for the warmup loop.
        Returns a list of per-member (obs, info).
        """
        results = [env.reset_pre_perturbation() for env in self.envs]
        # reset_pre_perturbation() calls reset_joints(), which in a vector env only records a plan.
        # Drain it HERE, before the stop below: og.sim.step() asserts a playing sim.
        self._drain_joint_resets()

        # Only cycle the sim if a perturbation actually requires a stopped one (adding or removing
        # objects). Pose-only perturbations such as VB-POSE and V-VIEW write on a live sim, and
        # cycling for them would reintroduce the very disruption this method exists to avoid.
        needs_stop = any(
            p in NEEDS_STOPPED_SIM for env in self.envs for p in env.active_perturbations
        )
        if needs_stop:
            og.sim.stop()

        obss = [env.apply_perturbations(res[0]) for env, res in zip(self.envs, results)]

        if needs_stop:
            # BEFORE play(), not after. play() runs _non_physics_step(), which initializes whatever
            # is on the queue and THEN calls update() on every object's states -- and update()
            # asserts the state is initialized. So an object evicted from the queue makes play()
            # itself raise "Cannot update uninitialized state." before any repair afterwards could
            # run. Measured with VSB-NOBJ on task 4, which replaces the MAIN object with one
            # carrying updatable states; V-SC missed it only because it replaces distractors.
            # Re-queueing while still stopped lets play() do the initialization itself, in the right
            # order, instead of us repairing the damage afterwards.
            orphans = self._repair_init_queue()
            og.sim.play()
            # Loud rather than silent, kept from when the repair ran after play(): if re-queueing
            # ever stops working, the next symptom is the same opaque "Object must be initialized
            # before dumping state!" raised from an unrelated call site much later.
            failed = [f"scene{o.scene.idx}/{o.name}" for o in orphans if not o.initialized]
            assert not failed, f"objects still uninitialized after re-queue + play(): {failed}"

        for env in self.envs:
            for fn in env.deferred_post_play:
                fn()
            env.deferred_post_play.clear()

        # Perturbations call reset_joints() too: V-VIEW, VB-POSE and SB-NOUN inline during
        # apply_perturbations(), and V-SC, VB-MOBJ, VSB-NOBJ and SB-VRB from the _post_play blocks
        # just drained. Both land here -- after the shared play, so the sim is playing, and before
        # the shared settle, which is the order a single env runs them in too.
        self._drain_joint_resets()

        if any(env.wants_settle for env in self.envs):
            self._settle()
        for env in self.envs:
            env.wants_settle = False

        # 9. every member re-takes its lift/distance/rotation reference from the object it will
        #    actually be scored on. RealmEnvironmentDynamic.reset() does this at its own tail; a
        #    vector env drives the phases itself, so it has to make the call itself too -- same
        #    reason the settle and the deferred post-play work are hoisted up here. It goes LAST,
        #    after the shared play and settle, because a replaced object is not initialized (and a
        #    settling one has not stopped moving) before that. See capture_mo_reference().
        for env in self.envs:
            env.capture_mo_reference()

        # Nothing may still be pending. reset_joints() RECORDS in a vector env, so a call site that
        # lands outside the drain points above would leave a drawer at whatever openness the
        # previous rollout ended on and score the next one against it -- silently, since every other
        # check would still pass. Loud here, exactly as for the init-queue repair above.
        stuck = [i for i, env in enumerate(self.envs) if env.pending_joint_reset is not None]
        assert not stuck, (
            f"members {stuck} recorded a joint reset that was never run -- a reset_joints() call "
            f"site is outside RealmVectorEnvironment's drain points"
        )

        return [(obs, res[1]) for obs, res in zip(obss, results)]

    def _repair_init_queue(self):
        """Undo the damage a SIBLING member's remove_object() did to the sim's global init queue.

        Two symmetric repairs, both caused by the same upstream bug (see below):
          (a) a LIVE object was knocked off the queue and would never be initialized;
          (b) a REMOVED object kept its slot and would be initialized after its prim was deleted.

        Adding an object appends it to the GLOBAL og.sim._objects_to_initialize
        (Simulator._post_import_object), and Simulator._non_physics_step() initializes everything on
        that queue as soon as the sim is playing. og.sim.play() runs _non_physics_step() itself, so
        the single batched play() above should be enough for every member -- and it is what makes
        __init__ work, where all N scenes' objects are added while stopped and initialized by one
        play(). Yet V-SC still asserted "Object must be initialized before dumping state!" out of
        scene.dump_state().

        The reason is that Simulator._pre_remove_object() prunes that queue by NAME ALONE
        (omnigibson/simulator.py:1089-1093 in OG 3.9.1):

            for i, initialize_obj in enumerate(self._objects_to_initialize):
                if obj.name == initialize_obj.name:
                    self._objects_to_initialize.pop(i)
                    break

        Object names are unique per SCENE (scene.add_object asserts against that scene's registry),
        NOT per simulator, and every member of a vector env is built from the same task config -- so
        all N scenes contain a "corkscrew", a "wineglass", a "cube", and so on. The perturbations that
        swap an object do it as remove_object() + add_object() under the SAME name (see
        _helpers.replace_obj), and reset() runs all members' perturbations inside one stopped window,
        so the queue holds several members' pending objects at once. Member 1's
        remove_object("corkscrew") then matches member 0's freshly-added "corkscrew" and pops THAT
        instead: the object stays on the stage and in scene 0's registry, but is off the queue, so no
        play() or step() will ever initialize it. The last member is always fine, which is why the
        failure looks like "every scene but the last one".

        The SAME pop also leaves the object that was actually removed ON the queue -- repair (b).
        That only bites when the removed object was itself still pending, which needs a member to
        add an object and then remove it inside ONE stopped window. SB-VRB is the only perturbation
        that does that: on a task with no target (pick_spoon) it adds a "receiver" and then, if the
        new verb is put/stack, replace_obj()s it. Measured on task 4, Vec=2: member 1 removed its
        own brand-new "receiver", the pop took member 0's instead, and the batched play() then ran
        initialize() on a prim that had already been deleted from the stage:

            File "omnigibson/simulator.py", line 1273, in _non_physics_step
              obj.initialize()
            ...
            Exception: prim view ['/World/scene_1/receiver/base_link'] is not a valid view

        Single-env never hit either half: with one scene there is no sibling to collide with, so the
        pop always takes the right entry, and each perturbation's own play() drains the queue before
        the next thing runs.

        The real fix belongs upstream -- _pre_remove_object should match on identity, or at least on
        (scene, name), not on name -- but OG-lite is a shared checkout here, so we repair the queue
        instead. Re-queueing the orphans and running one _non_physics_step() sends them through
        exactly the path play() would have used (obj.initialize(), keep_still(), update_handles(),
        joint-break bookkeeping), so the resulting state is what it would have been had the eviction
        never happened. It is deliberately NOT og.Environment.post_play_load(): that also reloads the
        observation/action spaces, rebases the scene's initial file and calls reset(), none of which
        belongs in the middle of a reset.
        """
        our_scenes = {id(env.omnigibson_env.scene) for env in self.envs}

        def _is_dead(obj):
            """Is this queued object one that has been removed from its scene?

            Deliberately NOT "is there a prim at obj.prim_path": replace_obj re-creates the
            replacement at the SAME relative prim path, so the path is occupied again a moment
            later and says nothing about which instance owns it. (Tried; it silently disabled this
            whole repair and the crash came straight back.) Identity against the registry is what
            distinguishes them.
            """
            registered = obj.scene.object_registry("name", obj.name)
            if registered is obj:
                return False        # the live object, about to be initialized normally
            if registered is not None:
                return True         # a same-named REPLACEMENT holds the name; this one is the corpse
            # Registered under its name by nothing at all. Either it was removed without a
            # replacement, or it was never registered -- scene.add_object(..., register=False), which
            # OmniGibson uses for particle system templates and which must NOT be dropped. The stage
            # tells them apart: a removed object with no replacement leaves its prim path empty.
            return not lazy.isaacsim.core.utils.prims.is_prim_path_valid(obj.prim_path)

        # (b) first, so the queue is clean before (a) reads it. Restricted to our members' scenes,
        # so anything queued by code outside this vector env is left strictly alone.
        stale = [
            obj
            for obj in og.sim._objects_to_initialize
            if id(obj.scene) in our_scenes and _is_dead(obj)
        ]
        if stale:
            og.log.warning(
                "Dropping %d removed object(s) that a sibling scene's eviction left on the sim init "
                "queue: %s"
                % (len(stale), ", ".join(f"scene{obj.scene.idx}/{obj.name}" for obj in stale))
            )
            stale_ids = {id(obj) for obj in stale}
            og.sim._objects_to_initialize = [
                obj for obj in og.sim._objects_to_initialize if id(obj) not in stale_ids
            ]

        # (a) Anything still queued is not an orphan -- it is about to be initialized normally.
        # (After a play() the queue should already be empty; this only guards against a future
        # caller.)
        queued = {id(obj) for obj in og.sim._objects_to_initialize}
        orphans = [
            obj
            for env in self.envs
            for obj in env.omnigibson_env.scene.objects
            if not obj.initialized and id(obj) not in queued
        ]
        if not orphans:
            return []

        # warning, not info: OmniGibson pins the root logger to WARNING (simulator.py:294), so
        # og.log.info() is silently dropped and this line would never be seen. It is also genuinely
        # warning-worthy -- it reports that we are papering over an upstream bug -- and it only fires
        # on resets where an eviction actually happened.
        og.log.warning(
            "Re-queueing %d object(s) evicted from the sim init queue by a sibling scene: %s"
            % (len(orphans), ", ".join(f"scene{obj.scene.idx}/{obj.name}" for obj in orphans))
        )
        # Just re-queue. The caller invokes this while the sim is STOPPED, so the following
        # og.sim.play() performs the initialization itself via _non_physics_step() -- in that
        # method's own order, which initializes queued objects before updating any state. Calling
        # _non_physics_step() here instead would assert, because the sim is not playing yet.
        og.sim._objects_to_initialize.extend(orphans)
        return orphans

    def _drain_joint_resets(self):
        """Run every member's pending drawer reset on ONE shared step loop.

        reset_joints() issues ~55 og.sim.step()s on a drawer task and og.sim.step() advances every
        scene, so letting each member run its own loop costs 55*N global steps per reset and steps
        each member's scene N times over while driving only one member's joints -- the same defect
        the settle loop and the stop/play cycle were hoisted out of the per-member loop for. In a
        vector env reset_joints() therefore only records a plan; this drains them all at once. See
        env_base.run_joint_resets for the full write-up, including why it is UNVERIFIED
        (open_drawer/close_drawer do not load on this port, so nothing here executes today).

        Called at every point a member could have recorded a plan: after construction's
        finalize_setup() loop, after phase 1 of reset(), and after the perturbations. reset() then
        asserts nothing is left pending, so a future reset_joints() call site outside those points
        fails loudly instead of silently skipping the drawer reset.
        """
        run_joint_resets(self.envs)

    def _settle(self, steps=SETTLE_STEPS):
        """Let every member's scene come to rest, on one shared step loop.

        Deliberately goes through the underlying OmniGibson env rather than this class's step():
        post_step() calls recompute_task_progression(), which MUTATES task_progression, so settling
        through it would credit the rollout with progress made before the policy ever acted. The
        single-env path avoids this the same way, by calling omnigibson_env.step() rather than
        RealmEnvironmentDynamic.step().
        """
        actions = [settle_action(env) for env in self.envs]
        # Nothing reads a camera here, so skip the render pass on each step. gm.HEADLESS does NOT do
        # this -- og.sim.step() renders every call regardless; only this context suppresses it.
        with og.sim.render_on_step(False):
            for _ in range(steps):
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._pre_step(action)
                og.sim.step()
                for env, action in zip(self.envs, actions):
                    env.omnigibson_env._post_step(action)

    def step(self, actions, n_render_iterations=1):
        """Apply one action per member and advance the shared simulator once.

        Args:
            actions (list): one action per member, same format as RealmEnvironmentDynamic.step
            n_render_iterations (int): render passes before observations are read. Mirrors
                og.Environment.step: one render happens inside og.sim.step() when rendering is
                enabled, and this issues the remaining n-1 as explicit og.sim.render() calls. The
                flush is GLOBAL -- one pass refreshes every scene -- so it is done once here rather
                than per member.

        Returns:
            list: per-member (obs, task_progression, terminated, truncated, info)
        """
        assert len(actions) == self.num_envs, f"expected {self.num_envs} actions, got {len(actions)}"
        for env, action in zip(self.envs, actions):
            env.pre_step(action)
        og.sim.step()
        for _ in range(n_render_iterations - 1):
            og.sim.render()
        return [env.post_step(action) for env, action in zip(self.envs, actions)]

    def warmup(self):
        """Settle every member together: hold the arm still, open then close the gripper.

        Mirrors RealmEnvironmentDynamic.warmup() but drives all members off one shared step loop,
        so the 30 settle steps cost the same wall time for 1 member as for N.
        """
        og.log.info("Starting vector warmup...")
        for _ in range(30):
            og.sim.render()

        results = self.reset()
        ee_cmds = [env.warmup_ee_cmd() for env in self.envs]

        for t in range(WARMUP_STEPS):
            actions = [env.warmup_action(t, ee_cmd) for env, ee_cmd in zip(self.envs, ee_cmds)]
            results = self.step(actions)

        # Refine the reference now the arms have settled, mirroring the single-env warmup. reset()
        # above already took it from the right object; this only updates it to the settled pose.
        for env in self.envs:
            env.capture_mo_reference()
        og.log.info("Vector warmup finished.")
        return results

    def close(self):
        pass

    def __len__(self):
        return self.num_envs

    def __getitem__(self, idx):
        return self.envs[idx]
