# Licensing and third-party notices

The root `LICENSE` applies to source code authored by REALM contributors. Copyright
in third-party material remains with its respective owners. Model weights, datasets,
simulator assets, and external runtime components retain their own applicable terms.

## OmniGibson and BEHAVIOR-1K code

REALM builds on [BEHAVIOR-1K / OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K).
The OmniGibson revision used to derive the patches in `.docker/patches/` is
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`, as recorded in that directory's
`PROVENANCE`. Its MIT copyright and permission notice are preserved in
[LICENSES/OmniGibson-MIT.txt](LICENSES/OmniGibson-MIT.txt) and alongside the patches
in [.docker/patches/LICENSE](.docker/patches/LICENSE). These notices also apply to
any upstream-derived source included in REALM.

Upstream copyright is (c) 2023 Stanford Vision and Learning Group.

## BDDL

[BDDL](https://github.com/StanfordVL/bddl) is supplied by the BEHAVIOR runtime.
Its MIT notice, copyright (c) 2021 Stanford Vision and Learning Lab, is preserved
in [LICENSES/BDDL-MIT.txt](LICENSES/BDDL-MIT.txt) for upstream-derived task material.

## Vendored OpenPI client

`packages/openpi-client/` contains code from
[Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client).
It retains Apache License 2.0, included in
[packages/openpi-client/LICENSE](packages/openpi-client/LICENSE).
Preserve upstream attribution and mark modifications when redistributing modified
files, as required by that license. The Apache terms continue to govern this code.

## Three.js

The bundled task-authoring viewer uses Three.js. Its MIT notice is retained in
[tooling/task_authoring/THREE-LICENSE.txt](tooling/task_authoring/THREE-LICENSE.txt).

## BEHAVIOR datasets and simulator assets

BEHAVIOR's dataset agreement is separate from its software license. It restricts
use to non-commercial academic research within OmniGibson, restricts extraction
and reverse engineering, and prohibits redistribution of the data or decryption
key. Users must accept the applicable dataset agreement when obtaining assets.
See the [upstream installer and dataset agreement](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/setup.sh)
and [asset documentation](https://behavior.stanford.edu/getting_started/important_concepts.html).
The root source-code license grants no additional rights to these assets.

`custom_assets/`, robot USDs, textures, and other binary or media assets need
asset-specific provenance and licensing. Their presence in this repository does
not establish redistribution rights. The provenance of the currently bundled
custom assets has not been fully established. Consult the original rights holder
and applicable asset terms before redistributing them.

## External runtimes, models, and other dependencies

NVIDIA Isaac Sim / Omniverse, downloaded models, and installed dependencies retain
their respective licenses and agreements. Obtain those components through their
supported distribution channels and comply with their terms. This document covers
the components identified above and is not a complete transitive dependency audit.
