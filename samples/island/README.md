# The Vancouver island (wave I7)

**Fifty square kilometres of real North Shore elevation, carved into an island.**
This folder is the *generator*, not the world: the recipe, the designed
coastline, the biome masks, the road network and the derived water layers —
**287 679 bytes** (281 KB) over eleven tracked files. The world it describes is
**549 879 456 B** (549.9 MB) of terrain and is not committed. (342.7 MB was wave
I7's figure; wave TER2b's detail band moved it, and the number here is a fresh
`island build` on the committed recipe — wave GTA1's audit.)

```sh
inf island build --recipe samples/island/island.toml
```

That is the whole command. It plans the source tiles, fetches the ones the cache
lacks, samples real elevation onto the world grid, carves the coastline, derives
the water and the biomes, drapes and audits the roads, builds the pyramid, and
writes the heavy halves into a project at `<checkout>/../island-build/project`.
About forty seconds on a warm cache (24.7 s of build, and the cook after it is
another 40.7 s), and **156 tiles / 12 MB** of elevation on a cold one. Run
`inf island plan` first to see exactly what it will ask the network for.

## What is committed and what is not

| committed | why |
|---|---|
| `island.toml` | every decision: where on Earth, how fine, which source, where the sea is, where the settlements are |
| `layers/coast.geojson` | **the designed coastline** — 43 vertices that turn a piece of a mountain range into a landmass |
| `layers/biomes.geojson` | the design masks: the farmland belts and the meadows the classifier is never allowed to invent |
| `layers/roads.geojson` | the road network, routed once under an 8 % grade ceiling and committed as the design |
| `layers/streams.geojson`, `layers/lakes.geojson` | derived from flow accumulation over the carved ground, then committed — the derivation is re-runnable, the layer is the artifact |

| NOT committed | why |
|---|---|
| the `.inf_terrain` | **549 879 456 B** (549.9 MB) |
| the road mesh | 517 086 vertices |
| the `.inf_biomes` set | derived from the palette |
| the tile cache | 12 MB of somebody else's bytes |

Everything in the second table is rebuilt by the one command above and lives
**outside the tree**, at `<checkout>/../island-build/`.

## The island in numbers

| | |
|---|---|
| map | 7 168 × 7 168 m = **51.38 km²** |
| land | **40.65 km²** (79.1 %) |
| peak | **948.7 m** — real North Shore ground |
| sea floor | −60.0 m, on a 500 m shelf |
| coastline | **25.14 km** |
| terrain | 784 level-0 tiles of 257², 1 064 in the catalog, 5 LOD levels |
| source | 156 terrarium tiles at z15 = **3.11 m/px**, upsampled 3.11× onto a 1 m grid |
| water | **51 reaches / 25.88 km**, 2 lakes / 0.0847 km², **33 waterfall sites** (biggest a 29.5 m drop) |
| biomes | forest 38.2 %, plain 20.1 %, meadow 13.5 %, alpine 8.6 %, beach 6.5 %, farmland 5.9 %, urban **7.2 %** |
| roads | **33.74 km** over 11 links and 7 junctions; worst grade 0.108 against a 0.080 ceiling, 5 of 2 442 stretches over |
| settlements | **2 cities of 1.131 and 1.020 km² and 5 towns; 172 blocks, 60.88 km of street, about 1 800 buildings** (wave I8a) |

## The elevation is real and the shape is designed

The source is the AWS terrain-tiles terrarium pyramid — a keyless, worldwide DEM
— over the ground behind Ambleside, with Grouse and Hollyburn to the north.
World `(0, 0, 0)` is 49.343 N, 123.102 W, in UTM zone 10N. Remember the frame:
**+X east, +Y up, −Z north**.

**What the survey gives is the relief.** What the design gives is everything
else: the coastline (there is no island there), the sea shelf and the beaches,
the seven settlement sites and their terraces, the road network, and the biome
masks. **And what stands on the terraces is a RULE** (wave I8a): the level
carries one `PcgVolume` per settlement block — 172 of them, **206 bytes each**
(the fixture's level grew 1 648 B for 8 blocks and nothing else; this island's
34 597 B is that figure less the 835 B the re-derived water splines shed in the
same commit, so 34 597 / 172 is an average and not the price of a block) —
naming one of seven committed zone documents in `samples/settlement/`, and the
streets, blocks, lots and buildings are derived from the sites and the road
layer by `inf_editor_core::settlement`. Nothing about a settlement is committed
geometry. The build says so every run — `[source.upsampled]` is a standing
advisory, because a 1 m grid over a 3.11 m survey is 3.11× of interpolation and
pretending otherwise would be the most flattering lie this folder could tell.

## Three standing advisories, and why none of them blocks

* **`source.upsampled`** — above.
* **`source.sea_level_tiles`** — 8 of 156 source tiles are uniformly sea level,
  and a missing tile decodes exactly the same way. Nothing here can tell them
  apart; the extent can.
* **`source.implausible`** — 56 source samples carry −32 768 m, which is the
  terrarium codec's floor and means "the provider filled this pixel". It is
  *finite*, so every finiteness guard in this engine waves it through; it is
  nodata here, and nodata becomes ocean.

None of the three is something an author can fix, so none of them stops the
build. What does: a mask that names no biome, and a road network more than 1 %
of which is over its own ceiling.

## The two-pass route

`inf island route` plans the network and then **re-builds against it**, because
the corridor levelling is part of the carve and a road audited before its
corridor is cut is a road nobody has built. Measured: **8.11 % of stretches over
the ceiling before, 0.29 % after.** The seven that remain are places two routes
cross at different elevations, which this generator does not grade-separate.

## The surfaces, and the order the bridge runs in (wave ASSET0)

This folder commits **synthesised** materials: five ground sets and, since
ASSET0, `Road_Asphalt` — the one the level's `Roads` entity binds, and the
reason the street stopped being the engine's 0.8 debug grey. Every one of them
is generated by `inf_material::ground` from an integer hash with no
transcendental in the path, byte-locked on every CI leg, and carries no licence.
A clone of this repository builds the island and gets all of them.

**A machine with the Unreal reference project can do better, locally.**
`tools/ue-export/export.py` exports Megascans surfaces to a staging directory
outside this repository and `inf-import` writes them into the built project *at
the committed GUIDs* — so the level names one id and gets whichever texels the
machine has. Nothing imported is ever committed; see
`editor/crates/inf-editor-core/src/assets/ue_import.rs` for why that
arrangement, and `docs/memos/island-progress.md`'s ASSET0 ledger for the licence
table.

> **Before you run it, the licence position, in one paragraph.** There is no
> licence file anywhere under the reference project's `Content`, so the position
> is per pack rather than per project. The **Megascans surfaces** came through
> **Fab** (user-confirmed 2026-09-05), which means Fab's **Standard License** —
> any engine, perpetual, royalty-free — so they are **cleared to ship**. The two
> Marketplace/Fab packs (`Downtown_West`, `AdvancedRealisticGlass`) are still
> **unknown**: check their listings before they go anywhere. The **UE5
> mannequins** are Epic's own *UE-Only Content* and are a **local dev reference
> only** — never cooked, never shipped. The **ALS** clips are **MIT** and may
> ship with the notice preserved. **MetaHumans** are licensed for any engine
> under the mid-2025 terms and ship inside the cooked pack.
>
> **Cleared to ship is not cleared to commit.** Every one of these licences
> permits *use*; none permits redistributing the source assets in a public
> repository. So nothing the bridge writes is ever committed: both ends refuse a
> destination inside this checkout outright, `ue-staging/` is in `.gitignore`,
> and `char1a_gate::nothing_from_unreal_is_inside_the_checkout` walks the tree
> looking for one that slipped past both doors.

**The order matters and there is only one that works:**

```
inf island build --recipe samples/island/island.toml     # 1. writes the project
inf-import --manifest <staging>/manifest.json \
           --into <project> --bind Road_Asphalt=<key>    # 2. overwrites in place
inf cook --project <project>                             # 3. packs what is there
```

`write_content` copies this folder's `[content]` list into the project on **every**
build, so an import that ran first is overwritten by the next build. Import
between the build and the cook, or the cook ships the synthesised surface — which
is not wrong, only less photographic.

## What CI runs instead

`samples/island-fixture` — 2.36 km² of the same ground with its two source tiles
committed beside it, exercising every step of the recipe and never touching a
network. See `crates/inf-island/tests/island_fixture.rs`.
