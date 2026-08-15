# OmniGibson 3.9.1 vs Isaac Lab 2.2.0 — simulator, physics scene, lifecycle

What OmniGibson's simulator, physics-scene and lifecycle layer does that raw Isaac Sim / Isaac Lab
does not. Every row cites `file:line`.

- **OmniGibson paths** are relative to `/mnt/home_lustre/sedlam56/projects/OG-lite_og391`.
- **Isaac Sim paths** are relative to `/opt/conda/envs/behavior/lib/python3.1/site-packages/isaacsim`
  *inside* `/mnt/home_lustre/sedlam56/projects/REALM_og391/realm_og391.sif` — i.e. the Isaac Sim
  **5.1.0** that OmniGibson actually runs on here. Three files are cited often enough to shorten:
  - `physics_context.py` = `exts/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py`
  - `isaac simulation_context.py` = `exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py`
  - `simulation_manager.py` = `exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py`
- **Isaac Lab paths** are in-SIF absolute under `/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif`,
  rooted at `/workspace/isaaclab/source/isaaclab/isaaclab`.
- **RoboLab paths** are relative to `/mnt/home_lustre/sedlam56/projects/RoboLab`.
- **PhysX USD schema defaults** are read from `PhysxSchema/resources/generatedSchema.usda` in the
  same SIF (`omni.usd.schema.physx-107.3.26`).

This is the simulator/scene chapter of the same reference as
[`control_and_actuation.md`](control_and_actuation.md). It is a **read-only audit — findings are
flagged, nothing is fixed.**

## Runtime evidence

Every "value" claim about the live physics scene below is read back from a completed run, not
inferred. Probe: `tmp/physcene_probe.py` (transient, not committed) — bare `og.launch()` with no
arguments, no scene, `MODE=oglite`, on `srun --jobid=191206 --overlap`, 2026-08-15. It dumps every
attribute on `/physicsScene` with its authored flag, plus the derived device and the carb settings.
The full capture is quoted in [Appendix A](#appendix-a--physicsscene-as-actually-configured).
`math.isclose` / floor-division arithmetic in row 12 is a completed local computation, reproduced in
that row.

---

## The headline difference

**OmniGibson configures the physics scene by omission.** It writes ten attributes and leaves the
rest at PhysX's USD-schema defaults. Isaac Lab writes its whole `PhysxCfg`, plus min/max iteration
counts and a bound default physics material, explicitly.

Read back from the live scene (Appendix A), only these are authored by OmniGibson:
`gravityDirection`, `gravityMagnitude`, `broadphaseType`, `enableCCD`, `enableGPUDynamics`,
`enableStabilization`, `invertCollisionGroupFilter`, `solverType`, `timeStepsPerSecond`, and six
`gpu*` capacities. Everything else — **both solver iteration caps, both velocity iteration caps,
`bounceThreshold`, `frictionOffsetThreshold`, `frictionCorrelationDistance`,
`enableSceneQuerySupport`, `enableResidualReporting`, `maxBiasCoefficient`, `frictionType`,
`collisionSystem`** — is unauthored.

That is not automatically wrong; for iteration counts it is what makes the two stacks agree (see
"benign"). But it means **OmniGibson has no opinion** on several parameters Isaac Lab does have an
opinion on, and the difference only shows up by reading the schema.

The second structural difference: **`og.sim.stop()`, `play()`, `step()` and `render()` act on every
scene in the simulator at once.** That gets its own section below.

---

## Table

Class column: **deviation from Isaac defaults** / **override of authored values** / **architectural
constraint** / **latent bug**.

### A. Physics-scene configuration

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 1 | `simulator.py:643`, `macros.py:184` | `enable_ccd(gm.ENABLE_CCD)` with `ENABLE_CCD = False`. Authored `enableCCD=False`, verified live | Isaac Sim's own **CPU** default is CCD **on** — `PhysicsContext.__init__` with `set_defaults=True` calls `enable_ccd(True)` on the non-GPU branch (`physics_context.py:112-113`). Isaac Lab's `PhysxCfg.enable_ccd` default is `False` (`sim/simulation_cfg.py:86`), applied at `sim/simulation_context.py:753` — **but RoboLab overrides it to `True`** (`robolab/core/environments/base.py:175`), and that key lands | **yes** | deviation from Isaac defaults | unmeasured. CCD only bites at high relative speed; quasi-static manipulation is unaffected. **RoboLab runs with CCD on, OmniGibson with it off** — a genuine cross-stack difference | no |
| 2 | `simulator.py:644` | `enable_fabric(True)` unconditionally | Isaac Sim enables fabric only on the GPU branch (`physics_context.py:99`) or when `sim_params["use_fabric"]` is set (`:147-148`). Isaac Lab defaults `use_fabric = True` (`sim/simulation_cfg.py:308`); RoboLab keeps it (`base.py:162`) | no | architectural constraint | none directly — but it is the root of rows 3 and 22 | no |
| 3 | `simulator.py:644` → `simulation_manager.py:606` | enabling fabric sets `/physics/updateToUsd = False`. Verified live | same mechanism on both stacks | no | architectural constraint | **PhysX results never reach USD.** This is why every USD write must sit inside `og.sim.editing_usd()` (`simulator.py:1631-1675`) and why a `Tf.Notice` guard aborts on any write outside it (`simulator.py:1677-1729`) | no |
| 4 | `simulator.py:646-652`, `macros.py:177` | forces `enableGPUDynamics=False` + `broadphaseType=MBP` whenever `gm.USE_GPU_DYNAMICS` is False — **regardless of the requested device** | PhysX schema defaults are `enableGPUDynamics=1` and `broadphaseType="GPU"`. Isaac Sim picks by device (`physics_context.py:96-103`); `SimulationManager.set_physics_sim_device` re-picks (`simulation_manager.py:303-323`) | partly | deviation from Isaac defaults | **measured near-irrelevant to compliance**: RoboLab forced onto the same configuration (`--device cpu`, verified `is_gpu_dynamics_enabled=False, broadphase=MBP, TGS, dt=1/120`) keeps **94%** of its GPU compliance — mimic residual 0.299/1.155/2.125° vs GPU's 0.328/1.263/2.250 at 5/20/50 N | no |
| 5 | `simulator.py:2067-2073` → `simulation_manager.py:326-339` | `og.sim.device` is **not stored** — it is *derived* from `suppressReadback && broadphase=="GPU" && gpuDynamics`. Because row 4 forces MBP + GPU-dynamics-off, `og.sim.device` returns `"cpu"` no matter what was passed to `og.launch(device=…)`. Verified live: `og.sim.device = cpu` | Isaac Lab's `SimulationCfg.device` is a stored string, default `"cuda:0"` (`sim/simulation_cfg.py:271`) | **yes** | deviation from Isaac defaults | a config asking for `device: cuda` with `USE_GPU_DYNAMICS=False` silently runs entirely on CPU tensors. It surfaces only as `env_base.py:91`'s `"Device mismatch!"` assert, and only for the **second** `Environment` in a vector env — the first launch is silent | no |
| 6 | `simulator.py:654-660`, `macros.py:187-198` | writes six `gpu*` capacities onto the scene unconditionally. Three raise the schema default: `gpuFoundLostAggregatePairsCapacity` and `gpuTotalAggregatePairsCapacity` 1024 → **16 777 216** (16384×), `gpuMaxRigidContactCount` 524 288 → **2 097 152** (4×), `gpuMaxRigidPatchCount` 81 920 → **327 680** (4×). Three re-write the default unchanged (`gpuFoundLostPairsCapacity` 262 144, `gpuMaxParticleContacts` 1 048 576) | Isaac Lab writes its own set via `sim_params` (`sim/simulation_context.py:248-252` → `physics_context.py:154-175`); RoboLab raises three to 2³⁰ (`base.py:172-174`) | no | inert config | **none** — with `enableGPUDynamics=False` (row 4, and effectively mandatory per row 7) not one of these is read by PhysX. Do not attribute behaviour to them | no |
| 7 | `macros.py:177` | the CPU default is effectively **mandatory**, not a preference | n/a | no | architectural constraint | enabling GPU dynamics hits ~30 sites mixing CPU and cuda tensors; **35 are fixed in OG-lite and it still segfaults natively in Isaac's `_warm_start`.** So row 4's "configuration difference" cannot be closed by flipping the flag | fixes are OG-lite |
| 8 | grep-clean: OmniGibson never writes `physxScene:min/maxPositionIterationCount`. Verified unauthored live (1 / 255) | leaves the scene cap wide open, so each object's own request stands. Every `USDObject` requests 32 position / 1 velocity iterations (`objects/usd_object.py:64-65`, applied `:419-420`) | Isaac Lab writes all four caps explicitly (`sim/simulation_context.py:775-779`) from `PhysxCfg` defaults 1/255/0/255 (`sim/simulation_cfg.py:46,56,66,76`); **RoboLab overrides max position to 32 and max velocity to 1** (`base.py:180-181`) | no | **benign** — see below | both stacks solve at **32 position iterations** | no |
| 9 | unauthored: `bounceThreshold` = **0.0** (schema default), verified live | never sets a restitution bounce threshold, so restitution applies at *any* impact speed | Isaac Lab's `PhysxCfg.bounce_threshold_velocity` defaults to **0.5** m/s (`sim/simulation_cfg.py:110`), reaching PhysX via `sim_params` (`physics_context.py:183-184`); **RoboLab sets 0.2** (`base.py:182`), and the key lands | **yes** | deviation from Isaac defaults | unmeasured. Below 0.2 m/s of relative normal velocity RoboLab suppresses restitution and OmniGibson does not — relevant to settling / placement jitter, not to steady contact force | no |
| 10 | unauthored: `enableSceneQuerySupport` = **True** (schema default), verified live | leaves scene-query support on | Isaac Lab defaults it **off** — `SimulationCfg.enable_scene_query_support = False` (`sim/simulation_cfg.py:293`); RoboLab does not override | no | deviation from Isaac defaults | performance only; OmniGibson needs it for its raycast-based object states | no |
| 11 | no scene-level physics material anywhere in OmniGibson (grep: no `defaultMaterial`, no `bind_physics_material`; `sim_params` is never passed to `SimulationContext` — `simulator.py:478-483` — so `physics_context.py:192-201` never runs). The ground plane's friction arguments are **commented out** at `simulator.py:826-829` | colliders with no authored material fall back to PhysX's built-in default | Isaac Lab **creates and binds** `/physicsScene/defaultMaterial` on every launch (`sim/simulation_context.py:781-787`) from `RigidBodyMaterialCfg`: static 0.5 / dynamic 0.5 / restitution 0.0, `average`/`average` combine (`sim/spawners/materials/physics_materials_cfg.py:42,45,48,51,61`) | **yes** | deviation from Isaac defaults | unmeasured here. Belongs to the rigid/materials lane — flagged, not chased | no |

### B. Stepping and cadence

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 12 | `simulator.py:448` and `:891` (`int(sim_step_dt // rendering_dt)`) vs `simulator.py:732-740` (`math.isclose(ratio, round(ratio))`) | **validation rounds, the consumer floors.** For `action_frequency=6, rendering_frequency=30`: `(1/6)/(1/30)` evaluates to exactly `5.0` so `_validate_dts` passes, but `(1/6)//(1/30)` is `4.0` — Python's float floor-division is computed more accurately than the division and lands one below. `_n_steps_per_loop` becomes 4 instead of 5 | n/a — Isaac Lab's decimation is an integer `render_interval` (`sim/simulation_cfg.py:284`), never derived from a float ratio | **yes** | **latent bug** | the env advances **20% less physics per `step()` than the configured action rate claims** (7.5 Hz effective at a requested 6 Hz). Also hits `3/30` (9 instead of 10, −10%) and `12/60` (4 instead of 5, −20%). Exact, unaffected by `physics_frequency`. Common rates 30/30, 15/15, 15/30, 10/30, 5/30, 2/30, 1/30 are all correct. **Upstream** (stock `simulator.py:426`, `:851`) | no |
| 13 | `macros.py:270-272`, verified live | OmniGibson's own defaults are sim-step 30 Hz / rendering 30 Hz / physics 120 Hz → `_n_steps_per_loop=1`, `n_physics_timesteps_per_render=4` → **4 physics substeps per `og.sim.step()`** | Isaac Lab's defaults are `dt=1/60`, `render_interval=1` (`sim/simulation_cfg.py:281,284`) → 1 substep | no | context | REALM does not run the OmniGibson default: it passes `common_freq=15` into rendering + action frequency (`realm/environments/env_dynamic.py:116-118`), giving **8 substeps**, matching RoboLab's `dt=1/120, render_interval=8` (`base.py:159-160`). See "benign" | no |
| 14 | `simulator.py:1458-1469` | two different stepping paths. `render=True` → `SimulationContext.step(render=True)` → `app.update()`, physics driven by the Kit timeline; `render=False` → an explicit loop of `n_physics_timesteps_per_render` single physics steps | Isaac Lab drives decimation with an explicit counter, not two paths | no | **benign** — equal substep counts | the timeline path takes the same number of substeps because `set_simulation_dt` derives `substeps = max(int(rendering_dt/physics_dt), 1)` (isaac `simulation_context.py:437-440`) and `set_physics_dt` turns that into `minFrameRate = timeStepsPerSecond/substeps` (`physics_context.py:281-285`). **Verified live: `minFrameRate = 30` at 120 Hz physics / 4 substeps** | no |
| 15 | `simulator.py:1460`, `:1462` | when `_n_steps_per_loop > 1` (e.g. action 15 Hz with rendering 30 Hz), a single `og.sim.step()` calls `_sim_context.step(render=True)` **twice** — two rendered frames per env step | Isaac Lab renders once per `render_interval` physics steps | no | context | rendering cost scales with the action/rendering ratio, not just with the frame count you asked for | no |
| 16 | `simulator.py:1503`, `:1528` | both physics-step hooks are gated on `not SimulationManager._warmup_needed`, so during Isaac's warm start no controller output is computed or flushed and **no contacts are accumulated** | Isaac's warm start is `force_load_physics_from_usd()` + `start_simulation()` + `update_simulation(dt, 0.0)` (`simulation_manager.py:236-242`), triggered on each play | **yes** | architectural constraint | the first physics update after every `play()` runs with zero commanded control and its contacts never reach `RigidContactAPI` | no |
| 17 | `macros.py:262`, `simulator.py:451`, `:1430-1445` | `gm.RENDER_ON_STEP` and `sim.step(render=…, blind=…)` select between the two paths of row 14 and, with `blind=True`, skip object-state / visual / transition-rule updates (`simulator.py:1306-1307`) | stock 3.9.1's `step()` takes no arguments (`simulator.py:1360` in the stock tree) | no | new API surface | perf only; substep count is unchanged | **yes** (`04fc69b`, `368550e`) |

### C. `play()` / `stop()` lifecycle

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 18 | `simulator.py:1373-1388` | every `play()` after a `stop()` re-calls `robot.update_controller_mode()`, `robot.reset()` and `robot.keep_still()` for **every initialized robot in every scene** | Isaac Lab applies actuator gains **once**, in `_process_actuators_cfg` (`assets/articulation/articulation.py:1514`) inside `_initialize_impl`, driven by the timeline PLAY event and guarded by `_is_initialized` (`assets/asset_base.py:287-326`). **`reset()` does not re-apply them** (`articulation.py:172-182`; the implicit actuator's `reset` is a no-op, `actuators/actuator_pd.py:111-113`) | **yes** | lifecycle divergence | a runtime gain change survives `env.reset()` in Isaac Lab and is **silently reverted** here | no |
| 19 | `simulation_manager.py:236-242` (`force_load_physics_from_usd`) + `/persistent/physics/resetOnStop = True` (**verified live**) + `/physics/updateToUsd = False` (row 3) | the *mechanism* behind row 18, and it is broader than gains: on the first play after a stop, PhysX re-reads the **entire** physics scene from USD. Because nothing computed while playing is ever written back to USD, **all** runtime physics state is discarded — not just drive gains | same Isaac mechanism, but Isaac Lab never stops mid-run | **yes** | architectural constraint | anything OmniGibson does not explicitly re-apply (`robot.reset()`, `keep_still()`, `Scene.restore`/`load_state`) is gone after a stop/play | no |
| 20 | `simulator.py:1413-1415`, `object_states/attached_to.py:105`, `systems/micro_particle_system.py:736` | `stop()` fires a **simulator-global** callback dict keyed by `f"{obj.name}_detach"`. Object names are unique per **scene** only (`scenes/scene_base.py:703-705`); the module-level `NAMES` set (`utils/python_utils.py:18`) is defined, cleared, and **never populated** — there is no global name registry | Isaac Lab has no analogue; nothing detaches on reset | **yes** | **latent bug** | in a vector env whose members come from one config, all N same-named objects register the **same key**, so only the last survives. `stop()` then detaches attachments in **exactly one scene**. Inferred from code, not measured. Same family as OG-lite's `59af7c0` | no — **upstream** |
| 21 | `simulator.py:937-946` | `import_scene` does `play()` → `scene.initialize()` → `step()` → `stop()` **per scene** | Isaac Lab clones all envs before the first play | no | architectural constraint | loading N scenes performs **N global play/stop cycles**; each advances every already-loaded scene by a full `og.sim.step()` and re-reads all of them from USD (row 19). Upstream flags it itself at `simulator.py:1383-1386` | no |
| 22 | `simulator.py:562-564` | a play/stop cycle during `Simulator.__init__`, before any scene exists | n/a | no | harmless | none | no |
| 23 | `scenes/scene_base.py:948-967` | `Scene.restore()` calls **`og.sim.stop()` and `og.sim.play()`** — global — whenever the object set changed | Isaac Lab's `reset()` writes state to the sim; it never stops the timeline | **yes** | architectural constraint | reached from `Scene.reset(hard=True)` (`:769-770`), i.e. the default env reset path. **A per-scene restore that adds or removes one object resets every other scene from USD** and fires row 20's callbacks | no |
| 24 | `omnigibson/__init__.py:104-141` | `og.clear()` tears the simulator down and relaunches it, revoking the viewport menubar USD watcher first (`:119-128`) so prim deletions do not queue deferred callbacks onto an invalid stage | n/a | no | architectural constraint | the comment records that skipping the revoke corrupts CUDA/PhysX state | no |

### D. Object add / remove

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 25 | `simulator.py:1019-1032`, `:1069-1076`, `m.OBJECT_GRAVEYARD_POS` at `:66` | removing any object: dump the state of **every** scene → teleport the doomed objects to a fixed world position `(100, 100, 100)` → take a **global** `step_physics()` → remove → reload the state of **every** scene | Isaac Lab does not support removing prims from a running scene at all | **yes** | architectural constraint | one uncommanded global physics substep per removal. The dump/reload restores registered poses and velocities but not PhysX's internal contact/solver history. Reached by every transition rule (`transition_rules.py:192`) and by `Scene.restore` (`scene_base.py:955`) | no |
| 26 | `simulator.py:962-967` | `adding_objects` invalidates the **global** physics sim view before loading, then rebuilds every handle in every scene (`simulator.py:1224-1249`) | n/a | no | architectural constraint | `RigidContactAPI` and `ControllableObjectViewAPI` are simulator-wide singletons (`:1248-1249`) — there is no per-scene view | no |
| 27 | `simulator.py:1089-1103` | prunes the init queue by **identity**, not by name | n/a | no | fix of an upstream bug | stock matched on `obj.name`, which in a vector env pops a sibling scene's freshly added object instead. Rationale in the comment at `:1091-1099` | **yes** (`59af7c0`) |

### E. Scene offsets and tiling

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 28 | `simulator.py:927-932`, `scenes/scene_base.py:345-434`; `m.SCENE_MARGIN = 10.0` (`simulator.py:68`), `m.INITIAL_SCENE_PRIM_Z_OFFSET = -100.0` (`:69`) | scenes are tiled along **+X only**, at offsets derived per scene from `compute_path_world_bounding_box` — i.e. **data-dependent**, accumulated in `_last_scene_edge`. Scene `idx != 0` is parked at z = −100 while loading | Isaac Lab clones envs onto a **fixed grid** with `InteractiveSceneCfg.env_spacing`; RoboLab uses 2.0 m (`base.py:161`) | no | architectural constraint | scene origins are not predictable from config alone. All scenes share z = 0 because the ground plane is global (row 30) | no |
| 29 | `simulator.py:574-577`, `scenes/scene_base.py:714-726` | the only collision groups are **two simulator-global** ones (`fixed_base_fixed_links`, `structural_doors`) at `/World/collision_groups/…`; every scene's fixed links join the *same* group | Isaac Lab calls `filter_collisions` to give each cloned env its own collision isolation | **yes** | architectural constraint | there is **no per-scene collision isolation** — every scene shares one broadphase and one contact buffer. Objects in different scenes will interact if they ever overlap; only the 10 m margin keeps them apart | no |
| 30 | `simulator.py:809-814`, `:843-848` | the ground plane and the skybox are **simulator-global singletons** — `add_ground_plane` returns early once one exists | Isaac Lab spawns terrain per scene config | no | architectural constraint | one ground plane at z = 0 for all scenes. A per-scene floor height is expressed by moving the *scene prim* instead (`scenes/static_traversable_scene.py:113-129`, whose name `move_floor_plane` is misleading — it moves `self._scene_prim`) | no |
| 31 | `scenes/scene_base.py:405-428` | re-applies every object's pose after the scene prim reaches its final position | stock 3.9.1 does not, so the world-frame target is baked against the parked z = −100 transform | was **yes** | fix of an upstream bug | **measured**: at `num_envs=4`, **70 of 128** registered objects in each of scenes 1–3 sat at z ≈ +100, including the task's breakfast table | **yes** (`ef7442b`) |

### F. Logging and diagnostics

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 32 | `simulator.py:294`; verified live (`logging.getLogger().level = 30`) | pins the **root** logger to WARNING at launch | n/a | **yes** | deviation | `simulator.py:164` raises only `omnigibson.simulator`'s own level to INFO. `omnigibson/__init__.py:46-48` attaches a handler to the `omnigibson` logger and sets `propagate = False` but never sets its **level**, so it inherits root = WARNING. Net: **every `log.info` in OmniGibson except `omnigibson.simulator`'s own is dropped** | no |
| 33 | `simulator.py:199-202`, `:327-335`, `:1356-1358`, `:1125`; `simulator.py:97-160` | prunes carb log channels to error under `gm.NO_OMNI_LOGS`; permanently disables `carb.windowing-glfw.plugin`, `omni.hydra.scene_delegate.plugin`, `omni.kit.manipulator.prim.model`; suppresses `omni.usd`, `omni.physicsschema.plugin`, `omni.physx.plugin` around **every** `play()`, and `omni.physx.tensors.plugin` around every prim removal; `SuppressLogsUntilError` buffers all stdout/stderr through launch | Isaac Lab leaves carb logging alone | **yes** | deviation | the three channels muted around `play()` are exactly where PhysX reports scale/rigid-body/transform-hierarchy complaints. A genuinely broken asset can load without a visible message | no |
| 34 | `omnigibson_5_1_0.kit:23`, `:25-27` | `enableDeveloperWarnings = false`, and the Kit **crash reporter is disabled** | Isaac Lab ships the stock experience | **yes** | deviation | a Kit-level crash produces no report. Relevant given that Isaac exits 139 at teardown regardless of outcome | no |
| 35 | `simulator.py:688`, `:694`; unauthored `enableResidualReporting = False` (verified live) | solver residuals are published to neither USD nor Fabric, and residual reporting is never enabled | Isaac Sim exposes `enable_solver_residuals` via `sim_params` (`physics_context.py:124-125`) | **yes** | deviation | **"did the solver converge at 32 iterations?" is unanswerable** without a code change. Worth knowing before anyone tries to explain a compliance result by solver convergence | no |
| 36 | `simulator.py:280-281` | raises `/rtx/descriptorSets` to 360 000 and `/rtx/reservedDescriptors` to 900 000 at launch (verified live) | Kit's default is 10 000 | no | resource pool | rendering resources only, no physics effect. Without it, multi-scene vector envs segfault during scene load with GPU memory nearly empty — rationale at `simulator.py:262-279` | **yes** |

### G. State published to USD / Fabric

| # | site | what OmniGibson does | consequence |
|---|---|---|---|
| 37 | `simulator.py:682-695`, all verified live | `updateToUsd=False`, `updateVelocitiesToUsd=False` (it is `gm.ENABLE_HQ_RENDERING`, default False), `updateParticlesToUsd=True`, `fabricUpdateTransformations=True`, `fabricUpdateVelocities=False`, `fabricUpdateJointStates=False`, `fabricUpdateResiduals=False`, `outputVelocitiesLocalSpace=False`, `fabricUseGPUInterop=True` | **only transforms are published.** Velocities and joint states reach neither USD nor Fabric — any velocity read through a USD/Fabric path is stale, and only the PhysX tensor API is valid. Note the ordering: `enable_fabric` sets `updateParticlesToUsd=False` (`simulation_manager.py:607`) and `_set_renderer_settings` re-enables it at `simulator.py:683` |
| 38 | `simulator.py:666` | `/rtx/rendermode = "RealTimePathTracing"` | deviation from Kit's default raster path. Images only; belongs to the rendering lane |

### H. OG-lite-only behaviour changes in this domain

| # | site | what it does | default | why it matters |
|---|---|---|---|---|
| 39 | `macros.py:233-249`, applied at `simulator.py:1317-1322` | **proximity gate**: objects whose AABB is farther than `PROXIMITY_GATE_RADIUS` from every robot's AABB are dropped from the `RigidContactAPI` contact matrix (rows *and* columns) and skipped in the per-step object-state loop | **`PROXIMITY_GATE_ENABLED = True`, radius 1.5 m — on by default** | the single most behaviour-changing OG-lite default in this domain. Its own docstring says **"TURN THIS OFF FOR MOBILE MANIPULATION"**: membership is computed when the contact view is built (on play, and on handle rebuild), not per step, so a base that drives across the room keeps its initialization-time membership |
| 40 | `macros.py:217-231` | `CONTACT_REPORTING_PATTERNS` restricts which links get `PhysxContactReportAPI` at load time | `None` (upstream behaviour) | off by default; excluded links are invisible to *every* contact query |
| 41 | `macros.py:206-215` | `ENABLE_VISUAL_UPDATES`, `OBJECT_STATE_UPDATE_WHITELIST` | `True`, `None` | off by default. The whitelist split is at `simulator.py:530-547`; all state types are still globally initialized (`:566-570`, `:2035-2038`) |
| 42 | `macros.py:251-257` | `INCREMENTAL_CONTACT_CACHE` folds contacts per substep instead of batching | `False` | off by default; equivalence test at `tests/test_contact_cache_equivalence.py` |
| 43 | `envs/env_base.py:649-687`, `envs/vec_env_base.py:40-78` | `step_blind()` / `render_obs()` | new API | `Environment.step_blind` **asserts** it is not inside a vector env (`env_base.py:664-668`) — an explicit guard against the global-step hazard of the next section |
| 44 | `scenes/scene_base.py:1275-1281` | `Scene.serialize` anchors the concatenation on the registry's device | n/a | no-op on CPU; needed only if row 4/5 is ever flipped |

---

## Global: acts on every scene, not one

This class bit the vectorization work repeatedly, so it gets its own list. **`og.sim.stop()`,
`play()`, `step()`, `step_physics()` and `render()` all act on the whole simulator.** There is one
`/physicsScene`, one timeline, one broadphase, one contact buffer, and one set of tensor views for
every scene loaded (`simulator.py:1237-1249`).

Everything below is a *per-scene-looking* call that is in fact global:

| call | site | what actually happens |
|---|---|---|
| `Scene.reset()` | `scenes/scene_base.py:774` | takes a **global** `og.sim.step_physics()` |
| `Scene.restore()` | `scenes/scene_base.py:948-967` | **globally** stops and plays the simulator if the object set changed |
| `Environment.reset()` | `envs/env_base.py:730`, `:732-733` | one **global** `og.sim.step()` plus three **global** `og.sim.render()` |
| `VectorEnvironment.reset()` | `envs/vec_env_base.py:80-90` | loops the above per member |
| `Environment.step()` | `envs/env_base.py:640` | `og.sim.step()` advances every scene |
| `SB3VectorEnvironment.reset()` | `envs/sb3_vec_env.py:104-106` | deliberately runs **30 global** `og.sim.step()` calls to "settle" |
| `Simulator.import_scene()` | `simulator.py:937-946` | play/step/stop per scene, each global (row 21) |
| `Simulator.removing_objects()` | `simulator.py:1021`, `:1032`, `:1076` | global state dump, global physics step, global state reload (row 25) |
| `stop()` callbacks | `simulator.py:1413-1415` | one flat dict for the whole simulator, keyed by a per-scene-unique name (row 20) |

**The compounding case.** The reset chain is `Environment.reset` (`env_base.py:718`) →
`task.reset` (`:723`) → `BaseTask._reset_scene` (`tasks/task_base.py:183`) → `Scene.reset`
(`scene_base.py:753`) → `restore` (`:770`) + `step_physics` (`:774`); then back in
`Environment.reset`, `og.sim.step()` (`env_base.py:730`) + 3 renders. So one member's reset costs
`1 + S` global physics substeps, where `S` is row 13's substep count. With
`VectorEnvironment.reset()` looping N members, **scene 0 is advanced `N × (1 + S)` substeps after
its own restore, with no action applied.** At N = 4 and S = 8 that is 36 substeps = **0.30 s of
uncommanded physics**. *Inferred from the call graph, not measured* — but it is exactly why the
vectorized envs had to hoist `play`/`stop`/`step` out of per-member loops.

Upstream knows. `envs/sb3_vec_env.py:12-13` and `:49-61` carry a `last_stepped_env` guard whose
comment says it outright: *"When you step the eval env, the physics state of the train env also gets
stepped, despite the train env not taking new actions."*

---

## Benign / by design — checked, do not re-check

- **dt and decimation are equivalent.** Both stacks run physics at **1/120 s** with **decimation 8**.
  OmniGibson: `gm.DEFAULT_PHYSICS_FREQ = 120` (`macros.py:272`) with REALM's `common_freq = 15`
  (`realm/environments/env_dynamic.py:116-118`) → 8 substeps per `og.sim.step()`, verified live via
  `_n_steps_per_loop × n_physics_timesteps_per_render`. RoboLab: `sim.dt = 1/(60*2)`,
  `render_interval = 8` (`robolab/core/environments/base.py:159-160`). **Equivalent. Not a
  deviation.** (OmniGibson's *library* default without REALM's override is 30/30/120 → 4 substeps —
  worth knowing when reading a bare `og.launch()` capture, as in Appendix A.)
- **The 8 substeps are still a measurement trap, not a deviation.** An externally applied force
  lands on exactly one of the eight, so a naive external-force measurement reads **~8× too stiff**.
  Same on both stacks.
- **Both stacks solve at 32 position iterations.** OmniGibson leaves the scene caps at the schema
  defaults 1/255 (verified unauthored) and each object requests 32 (`objects/usd_object.py:64`,
  `:419`); RoboLab's articulation asks for 64 but its scene sets
  `physxScene:maxPositionIterationCount = 32` (`base.py:180`), which the schema documents as
  overriding actors that request more. Two opposite routes, same answer. **Velocity iterations
  differ (0 vs 1) and were measured to have no effect.**
- **Solver type.** TGS on both. OmniGibson never writes it; the schema default is `TGS` and Isaac's
  `PhysicsContext` sets it explicitly anyway (`physics_context.py:115`). Verified live: `TGS`.
- **GPU dynamics is a real configuration difference with a measured-small effect.** 94% of RoboLab's
  compliance survives being forced onto OmniGibson's exact CPU/MBP/TGS configuration (row 4). Do not
  spend more time on it; do spend time on row 7 (it cannot be flipped anyway).
- **Gravity.** −9.81 m/s² on both (`simulator.py:421`, `:639`; `sim/simulation_cfg.py:287`).
  Verified live: direction `(0,0,-1)`, magnitude 9.81.
- **`enableStabilization` = False, `enableEnhancedDeterminism` = False,
  `frictionOffsetThreshold` = 0.04, `frictionCorrelationDistance` = 0.025, `frictionType` = patch,
  `collisionSystem` = PCM, `maxBiasCoefficient` = inf** — identical on both stacks, whether authored
  or defaulted. Verified live.
- **`invertCollisionGroupFilter` = False** (`simulator.py:642`) is the schema default. No-op write.
- **The object graveyard does not collide with anything.** `m.OBJECT_GRAVEYARD_POS = (100, 100, 100)`
  (`simulator.py:66`) looks dangerous next to scene tiling that walks along +X past x = 100 for
  ~6 large scenes — but the graveyard is also at **y = 100 and z = 100**, while every scene sits at
  y ≈ 0, z ≈ 0. It cannot intersect a scene. Checked; benign.
- **The two stepping paths take the same number of substeps** (row 14). Checked against the Kit
  timeline's `minFrameRate` derivation and confirmed live.

---

## Traps that mislead but are not deviations

- **"Isaac Lab silently drops config keys" is the wrong lesson.** The `if hasattr(physx, attr_name)`
  guard that drops 10 of 18 PhysX settings is **RoboLab's own code** — the dict at
  `robolab/core/environments/base.py:171-190`, the guard at
  `robolab/core/environments/base.py:191-194`, not Isaac Lab's. RoboLab writes one dict covering two
  Isaac Lab schema generations and lets `hasattr` pick. Against the `PhysxCfg` in
  `isaac-lab-2.2.0.sif` (`sim/simulation_cfg.py:20-160`) the eight that **land** are
  `gpu_temp_buffer_capacity`, `gpu_heap_capacity`, `gpu_collision_stack_size`, `enable_ccd`,
  `max_position_iteration_count`, `max_velocity_iteration_count`, `bounce_threshold_velocity`,
  `solver_type`; the ten **dropped** are `contact_offset`, `rest_offset`,
  `num_position_iterations`, `num_velocity_iterations`, `max_depenetration_velocity`, `num_threads`,
  `relaxation`, `warm_start`, `shape_collision_distance`, `shape_collision_margin`. Do not attribute
  behaviour to any of those ten. Isaac Lab itself applies `PhysxCfg` through an explicit
  `if "<key>" in sim_params` chain (`sim/simulation_context.py:248-252` →
  `physics_context.py:118-201`) plus direct writes for the iteration counts, CCD, determinism and
  gravity (`sim/simulation_context.py:753-779`) — every field I could trace does land.
- **The two Isaac Sim versions are not the same.** OmniGibson runs Isaac Sim **5.1.0**
  (`realm_og391.sif`); the Isaac Lab 2.2 SIF pairs with a different build. Any claim about
  `PhysicsContext` behaviour must name which one it was read from.
- **`og.sim.device` is derived, not stored** (row 5). Reading it back does **not** tell you what was
  requested — it tells you whether GPU dynamics is on. `og.clear()`'s device assert
  (`omnigibson/__init__.py:139-141`) compares the derived value to itself and therefore always
  passes.
- **`gm.GPU_*` capacities are live-authored but inert** (row 6). They appear on the scene prim with
  large values; PhysX never reads them while `enableGPUDynamics=False`. Reading them back proves
  nothing.
- **`log.info` is invisible** (row 32). A diagnostic added with `log.info` in any module other than
  `omnigibson.simulator` will produce no output and can be mistaken for code that did not run.
- **`_validate_dts` passing does not mean the decimation is right** (row 12).
- **`move_floor_plane` does not move the floor plane** (`scenes/static_traversable_scene.py:113-129`)
  — it moves the scene prim. The floor plane is global (row 30).
- **`macros.py` also carries overrides outside this lane**, and nobody else will enumerate them:
  `FORCE_LIGHT_INTENSITY = 10000` (`:275`) and `FORCE_ROUGHNESS = 0.7` (`:278`) override
  USD-authored values for **all** dataset objects, and `FORCE_CATEGORY_MASS = True` (`:282`) uses a
  category-level mass instead of the per-model density × volume. Rendering and rigid/mass lanes
  respectively — flagged here, not chased.
- **Isaac Lab disables contact processing globally** (`sim/simulation_context.py:305`,
  `/physics/disableContactProcessing = True`, re-enabled when a contact sensor is created).
  OmniGibson leaves it on and applies `PhysxContactReportAPI` broadly. Not a physics-fidelity
  difference, but it makes contact-cost comparisons between the stacks meaningless.

---

## Appendix A — `/physicsScene` as actually configured

Read back from a completed run (see "Runtime evidence"). `authored=False` means the value is the
PhysX USD-schema default and **OmniGibson never touched it**. Launch was a bare `og.launch()`, so
the dt row shows OmniGibson's own defaults, not REALM's.

```
physics:gravityDirection                      authored=True  (0, 0, -1)
physics:gravityMagnitude                      authored=True  9.8100004196167
physxScene:bounceThreshold                    authored=False 0.0
physxScene:broadphaseType                     authored=True  MBP
physxScene:collisionSystem                    authored=False PCM
physxScene:enableCCD                          authored=True  False
physxScene:enableEnhancedDeterminism          authored=False False
physxScene:enableGPUDynamics                  authored=True  False
physxScene:enableResidualReporting            authored=False False
physxScene:enableSceneQuerySupport            authored=False True
physxScene:enableStabilization                authored=True  False
physxScene:frictionCorrelationDistance        authored=False 0.025
physxScene:frictionOffsetThreshold            authored=False 0.04
physxScene:frictionType                       authored=False patch
physxScene:gpuFoundLostAggregatePairsCapacity authored=True  16777216   (schema default 1024)
physxScene:gpuFoundLostPairsCapacity          authored=True  262144     (schema default 262144)
physxScene:gpuMaxParticleContacts             authored=True  1048576    (schema default 1048576)
physxScene:gpuMaxRigidContactCount            authored=True  2097152    (schema default 524288)
physxScene:gpuMaxRigidPatchCount              authored=True  327680     (schema default 81920)
physxScene:gpuTotalAggregatePairsCapacity     authored=True  16777216   (schema default 1024)
physxScene:invertCollisionGroupFilter         authored=True  False
physxScene:maxBiasCoefficient                 authored=False inf
physxScene:maxPositionIterationCount          authored=False 255
physxScene:maxVelocityIterationCount          authored=False 255
physxScene:minPositionIterationCount          authored=False 1
physxScene:minVelocityIterationCount          authored=False 0
physxScene:solverType                         authored=True  TGS
physxScene:timeStepsPerSecond                 authored=True  120
physxScene:updateType                         authored=False Synchronous

og.sim.device                       = cpu          (derived — see row 5)
SM.get_backend()                    = torch
sim.get_physics_dt()                = 1/120
sim.get_rendering_dt()              = 1/30         (bare og.launch(); REALM uses 1/15)
sim.get_sim_step_dt()               = 1/30         (bare og.launch(); REALM uses 1/15)
sim._n_steps_per_loop               = 1
sim.n_physics_timesteps_per_render  = 4
substeps per og.sim.step()          = 4            (8 under REALM's common_freq=15)

/persistent/simulation/minFrameRate   = 30         (= timeStepsPerSecond / substeps)
/persistent/physics/resetOnStop       = True       (mechanism behind rows 18-19)
/persistent/physics/numThreads        = 8
/physics/suppressReadback             = False
/physics/updateToUsd                  = False
/physics/updateVelocitiesToUsd        = False
/physics/updateParticlesToUsd         = True
/physics/fabricUpdateTransformations  = True
/physics/fabricUpdateVelocities       = False
/physics/fabricUpdateJointStates      = False
/physics/outputVelocitiesLocalSpace   = False
/app/runLoops/main/rateLimitEnabled   = False
/rtx/descriptorSets                   = 360000     (OG-lite, simulator.py:280)
/rtx/reservedDescriptors              = 900000     (OG-lite, simulator.py:281)

logging.getLogger().level             = 30 (WARNING)
```

---

## Not covered

- **`transition_rules.py` itself.** Its lifecycle coupling is covered (rows 25, 21 and
  `simulator.py:1338-1340`, `:1390-1392`), but the rule semantics — what fires, when, and what each
  recipe changes — are not audited here. `gm.ENABLE_TRANSITION_RULES` defaults to **True**
  (`macros.py:204`), so the engine runs every step for every scene by default.
- **Rendering settings** beyond their effect on physics readback (rows 37-38).
- **Per-object physics** (mass, inertia, collision approximation, materials) — the rigid lane. Row 11
  touches its boundary: OmniGibson binds **no** scene-level default physics material where Isaac Lab
  binds 0.5/0.5/0.0.
- **Vector-env correctness beyond the global-call inventory.** The compounding-reset estimate in the
  global section is **inferred from the call graph and has not been measured.**
