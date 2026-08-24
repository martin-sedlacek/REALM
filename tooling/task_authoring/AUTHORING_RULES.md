# REALM task-authoring geometry rules

These rules apply to prompt-generated drafts, batch-generated task families, and manual review of
saved configs.

## Support-plane clearance

`relative_bbox_position[2]` is added directly to the scene support Z. Never author a generated
object at relative Z `0`: that can place its origin or lower geometry inside the support and produce
large contact forces on the first simulation step. The authoring tools use:

```text
relative Z = authored bbox height / 2 + 0.050 m
```

Because REALM adds relative Z to the scene support Z, this guarantees:

```text
world bbox bottom = scene support Z + relative Z - authored bbox height / 2
                  = scene support Z + 0.050 m
```

The 50 mm clearance gives the object a short settling drop and prevents it from spawning in contact with the support because of visual
support thickness, bounds tolerances, or numerical error. A model whose origin is known not to be
at its bbox centre may need a larger hand-reviewed offset.

## Proportion-preserving resize and rotation

- Treat dataset metadata bounds as the natural dimensions.
- If an object must be resized to fit the spawn region or receiver, apply one uniform scale factor
  to X, Y, and Z. Never squeeze axes independently.
- Prefer no rotation. If footprint alignment is needed, use yaw (Z-axis rotation), normally 90°.
- Do not introduce roll or pitch unless the task instruction explicitly requires a non-upright
  initial state and the placement has been reviewed in simulation.
- Record automatic resizing with the original bbox, authored bbox, scale factor, and reason.

## Receiver and stack-support capacity

Before accepting a two-object draft, compare the main object's authored XY footprint against the
target after considering both 0° and 90° yaw:

- `put`: target XY must cover the oriented main XY with a 1.15 multiplier. This is a conservative
  outer-bbox proxy for partial containment; real interior volume still needs simulation review.
- `stack`: target XY must support at least 0.65 of the oriented main XY footprint.

If the check fails, first try a 90° main-object yaw. If it still fails, uniformly shrink the main
object until the proxy passes. Never distort either object's proportions to force a match.

## Instruction closure and initial predicates

Every object required to make an instruction meaningful must be grounded in the scene, even when
REALM's task type only requires one `main_object`. Treat prepositions as state constraints, not as
disposable language:

- `remove/take X from Y` requires both X and Y. X is the main object and Y is an immutable source.
- `take X off Y` requires X to start supported by Y.
- A lid removed from a pot or pan must start centered above that vessel, with its lower bbox face
  10 mm above the vessel's upper bbox face. This avoids initial interpenetration while preserving a
  short settling drop.
- `take X from/out of Y` requires X to start at least partially inside Y. The source footprint must
  first pass the same conservative capacity check as a `put` receiver.

The generated-config audit records the inferred predicate, main object, source object, and applied
clearance. A draft is semantically invalid if a required source is absent or its bbox relationship
does not satisfy the inferred predicate.

## Final validation

Bounding-box checks cannot prove mesh clearance, container interior volume, stability, reachability,
or collision-free settling. Generated configs must still pass an OmniGibson scene-correctness run
(`SUITE_MODE=oglite`) before they are treated as benchmark-ready.
