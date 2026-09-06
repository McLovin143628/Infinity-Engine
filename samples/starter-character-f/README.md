# Starter character (female)

**The engine's second committed body** - the same New Character wizard, the
same 161-bone rig and the same eight assets as `../starter-character/`, built
from a different set of MEASURED proportions.

Every number in the spec was read off `SKM_Quinn`'s exported glTF on
2026-09-05 (joint world bind translations + the mesh's own vertex bounds):
1.8017 m tall, hips at 0.5480, shoulders at 0.7746, head at 0.9021, a
0.3211 m shoulder span and a 0.2231 m hip span. Against Manny's own
measurements she is 15.5% narrower across the shoulders and 12.1% wider
across the hips as a fraction of height.

**Nothing from Unreal is in these files.** The mannequin was MEASURED; the
geometry is this engine's own generator, vertex for vertex.

Both bodies publish the same 161 joint names in the same order, which is
what lets one clip play on either — see `char1a_gate.rs`.

Generated - do not hand-edit. Regenerate with:

```sh
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
