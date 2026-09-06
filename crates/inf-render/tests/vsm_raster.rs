//! **The P27.2 caster pass**, on a real device — the per-page GPU cull, the one
//! render pass that owns the atlas, and the viewport/scissor pair that pins each
//! page to its slot.
//!
//! Every arm here reads the atlas **back off the device** and compares texels.
//! That is deliberate and it is the standing law: the claim P27.2 makes is *depth
//! in a rectangle*, and a pass report, a draw counter or a residency snapshot can
//! all be perfect while the rectangle is empty. So the counters are used for
//! anti-vacuity and the assertions are made on depth.
//!
//! What these arms are built to falsify:
//!
//! * a raster that draws **nothing** (the atlas is all clear — which is exactly
//!   what P27.1 left, so every other assertion in the phase still passes);
//! * a raster whose page **matrix** is wrong (the depth is there but it is not the
//!   depth the page's own projection produces — checked against a CPU
//!   re-derivation, not against itself);
//! * a raster that writes **outside its page** (a missing or wrong
//!   `set_viewport`: the caster covers its whole page, so a viewport spanning the
//!   atlas paints every other slot);
//! * a cull that keeps everything (a caster far outside a page must not reach it);
//! * a masked material that shadows as a **solid** rather than as a cutout.
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.

use inf_render::{GpuContext, RenderScene, RenderView, VsmLightHandle, VsmPage, VsmSettings};

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

const FW: u32 = 256;
const FH: u32 = 144;
/// The caster cube's side, in metres.
const SIDE: f32 = 2.0;
/// Off-centre in y for `vsm_marking`'s reason — a marked set symmetric about the
/// clipmap centre cannot see the page grid's vertical flip.
const CUBE_XY: (f32, f32) = (0.0, 1.3);

/// A **small** atlas: 8 slots, so a page's rect is a large fraction of it and an
/// escaped draw is impossible to miss. 128-texel pages, 8 pages ⇒ 512 KiB.
fn settings() -> VsmSettings {
    settings_with(8)
}

/// The same clipmap with an atlas of `slots` pages. A 128-texel `Depth32Float`
/// page is exactly 64 KiB (the P27.1 page-geometry ruling), so a slot count is a
/// budget.
fn settings_with(slots: u64) -> VsmSettings {
    VsmSettings {
        enabled: true,
        budget_bytes: slots * 64 * 1024,
        clipmap_pages_per_side: 8,
        clipmap_levels: 6,
        first_level_extent_m: 6.0,
        ..Default::default()
    }
}

/// Where the opaque backdrop's near face sits, in metres along the light.
const BACKDROP_Z: f32 = -3.0;
/// Half its thickness.
const BACKDROP_T: f32 = 0.1;

/// A wide, thin, **opaque** slab well behind the caster — the surface whose depth
/// marks the pages, so an arm about a caster that discards is not an arm about a
/// frame with no depth in it at all.
fn backdrop() -> inf_render::MeshInstance {
    inf_render::MeshInstance::lit(
        glam::DVec3::new(0.0, 0.0, (BACKDROP_Z - BACKDROP_T) as f64),
        glam::Quat::IDENTITY,
        glam::Vec3::new(40.0, 40.0, 2.0 * BACKDROP_T),
        [1.0, 1.0, 1.0, 1.0],
        9,
    )
}

fn view(eye_z: f64) -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 0.0, eye_z),
        forward: glam::Vec3::NEG_Z,
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// One cube under one directional shadow caster, on an empty sky.
fn scene(blend: u8, cutoff: f32, alpha: f32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    let mut inst = inf_render::MeshInstance::lit(
        glam::DVec3::new(CUBE_XY.0 as f64, CUBE_XY.1 as f64, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(SIDE),
        [1.0, 1.0, 1.0, alpha],
        1,
    );
    inst.blend = blend;
    inst.cutoff = cutoff;
    scene.instances.push(inst);
    scene.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        // Along +Z, so the light basis is unambiguous and a page's contents are
        // hand-checkable (`vsm.rs`'s clipmap twin arm uses the same direction for
        // the same reason).
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    scene.mark_dirty();
    scene
}

fn run(
    gpu: &GpuContext,
    scene: &RenderScene,
    v: &RenderView,
    set: &VsmSettings,
    frames: u64,
) -> inf_render::EngineRenderer {
    run_sequence(gpu, &[(scene, frames)], v, set)
}

/// The same renderer across a **sequence** of scenes — the door an arm about what
/// one frame leaves behind for the next has to come through (P27.2 audit). `run`
/// is this with one entry.
fn run_sequence(
    gpu: &GpuContext,
    steps: &[(&RenderScene, u64)],
    v: &RenderView,
    set: &VsmSettings,
) -> inf_render::EngineRenderer {
    run_tuned(gpu, steps, v, set, |_| {})
}

/// [`run_sequence`] with the rest of the render settings tunable — the door an arm
/// about a setting the caster pass *reads* (rather than one it owns) comes through.
fn run_tuned(
    gpu: &GpuContext,
    steps: &[(&RenderScene, u64)],
    v: &RenderView,
    set: &VsmSettings,
    tune: impl FnOnce(&mut inf_render::RenderSettings),
) -> inf_render::EngineRenderer {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    s.vsm = *set;
    tune(&mut s);
    renderer.set_settings(s);
    for (scene, frames) in steps {
        for _ in 0..*frames {
            renderer.render(gpu, scene, v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    renderer
}

/// The whole page atlas, off the device, as `f32` depth — **the WORLD**.
fn read_atlas(gpu: &GpuContext, renderer: &inf_render::EngineRenderer) -> (u32, u32, Vec<f32>) {
    let sys = renderer.vsm().expect("a live vsm system");
    let tex = sys.pools().atlas();
    let (w, h) = (tex.width(), tex.height());
    let unpadded = w as usize * 4;
    let padded = unpadded.next_multiple_of(256);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vsm-atlas-readback"),
        size: (padded * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vsm-atlas-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map");
    let data = slice.get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((w * h) as usize);
    for row in data.chunks(padded).take(h as usize) {
        for texel in row[..unpadded].chunks_exact(4) {
            out.push(f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]));
        }
    }
    drop(data);
    buffer.unmap();
    (w, h, out)
}

/// Every resident page, with the atlas rectangle it owns.
fn resident_pages(renderer: &inf_render::EngineRenderer) -> Vec<(u32, VsmPage, (u32, u32, u32))> {
    let sys = renderer.vsm().expect("a live vsm system");
    let res = sys.residency();
    let geom = res.geometry();
    let mut out = Vec::new();
    for slot in 0..geom.slot_count() {
        if let Some((light, page)) = res.slot_occupant(slot) {
            let (x, y) = geom.slot_origin(slot).expect("a seated slot has an origin");
            out.push((light.0, page, (x, y, geom.stored_page_size)));
        }
    }
    out
}

/// **What each resident slot depicts** (island wave VSM2), as the shipped page
/// cache keys it: `(light, face, level, world cell x, world cell y, the light's
/// content key)`.
///
/// The renderer's own `PageIdent` is private, so this is the same statement built
/// from the two public doors it is built from — the residency's clip origins and
/// the layout's content key. A test that compared page *matrices* instead would
/// be comparing something the cache no longer looks at.
fn page_identities(r: &inf_render::EngineRenderer) -> Vec<(u32, u32, u32, i64, i64, u64)> {
    let sys = r.vsm().expect("a live vsm system");
    resident_pages(r)
        .into_iter()
        .map(|(light, p, _)| {
            let o = sys.residency().clip_origins(VsmLightHandle(light));
            let g = o.get(p.level as usize).copied().unwrap_or((0, 0));
            let key = sys
                .layouts()
                .get(light as usize)
                .map_or(0, |l| l.content_key);
            (
                light,
                p.face,
                p.level,
                g.0 + i64::from(p.x),
                g.1 + i64::from(p.y),
                key,
            )
        })
        .collect()
}

/// Texels of one rectangle that are not the reverse-Z clear value.
fn written(atlas: &(u32, u32, Vec<f32>), rect: (u32, u32, u32)) -> Vec<f32> {
    let mut out = Vec::new();
    for_each_written(atlas, rect, |_, _, d| out.push(d));
    out
}

/// Every written texel of one rectangle, with its **atlas coordinates** — the door
/// an arm about *where* in a page the depth landed comes through (P27.2 audit).
fn for_each_written(
    atlas: &(u32, u32, Vec<f32>),
    rect: (u32, u32, u32),
    mut f: impl FnMut(u32, u32, f32),
) {
    let (w, _, ref data) = *atlas;
    for y in rect.1..rect.1 + rect.2 {
        for x in rect.0..rect.0 + rect.2 {
            let d = data[(y * w + x) as usize];
            if d != inf_render::VSM_DEPTH_CLEAR {
                f(x, y, d);
            }
        }
    }
}

/// The render-local world point a written texel stands for: undo the viewport
/// transform into the page's own NDC, then the page matrix.
///
/// **The reconstruction is what makes a depth arm independent of the geometry that
/// wrote it** (P27.2 audit): it turns "there is depth in this page" into "the
/// surface is *here*, in metres", which is a claim the fixture's own heights can be
/// checked against rather than a claim about a texel count.
fn texel_world(
    vp_inv: glam::Mat4,
    rect: (u32, u32, u32),
    x: u32,
    y: u32,
    depth: f32,
) -> glam::Vec3 {
    let s = rect.2 as f32;
    // wgpu's viewport puts NDC +1 at the TOP of the rect, and a texel's centre is
    // half a texel in from its corner.
    let ndc_x = ((x - rect.0) as f32 + 0.5) / s * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y - rect.1) as f32 + 0.5) / s * 2.0;
    let h = vp_inv * glam::Vec4::new(ndc_x, ndc_y, depth, 1.0);
    h.truncate() / h.w
}

/// The page matrix of one resident page, through the shipped door.
fn page_vp(renderer: &inf_render::EngineRenderer, light: u32, page: VsmPage) -> glam::Mat4 {
    renderer
        .vsm()
        .expect("a live vsm system")
        .page_matrix(VsmLightHandle(light), page)
        .expect("a resident page has a matrix")
}

// ── (a) the pass runs, and the depth it writes is the depth the page's own
//        projection produces ────────────────────────────────────────────────────

/// **THE RASTER ARM**: a resident page holds real caster depth, at the value the
/// page's matrix says, and the pages nothing casts into hold the clear value.
///
/// The depth is checked against a **CPU re-derivation through
/// `vsm_page_matrix`** — the cube's near face, projected into the page — rather
/// than against "something non-zero", so a raster that drew the wrong geometry,
/// at the wrong scale, or through the wrong level fails here.
#[test]
fn a_resident_page_holds_the_depth_its_own_projection_produces() {
    let Some(gpu) = gpu_or_skip("the VSM page raster") else {
        return;
    };
    let set = settings();
    let v = view(5.0);
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &v, &set, 6);

    let stats = renderer
        .vsm_raster_stats()
        .expect("the system exists once a light casts");
    // ANTI-VACUITY, three ways: the pass ran, it saw pages, and it issued draws.
    // `vsm_raster_frames` counts frames that rasterized **at least one page**, and
    // since P27.3 that is a small number on a static scene by design — the pages
    // settle and the cache serves them. One is the floor the claim needs.
    assert!(renderer.vsm_raster_frames() >= 1, "{stats:?}");
    assert!(
        stats.pages > 0 && stats.draws > 0 && stats.casters > 0,
        "{stats:?}"
    );
    assert_eq!(stats.deferred_pages, 0, "the fixture outgrew the page cap");

    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty(), "nothing was resident to rasterize");
    let atlas = read_atlas(&gpu, &renderer);

    // The cube's near face in light space: the light looks along −Z (its
    // `direction` is the direction TO the light, +Z), so the surface a page sees
    // is the face at `z = +SIDE/2`.
    let mut checked = 0;
    let mut total_written = 0usize;
    for (light, page, rect) in &pages {
        let vp = page_vp(&renderer, *light, *page);
        let hits = written(&atlas, *rect);
        total_written += hits.len();
        if hits.is_empty() {
            continue;
        }
        // The cube's near face, at the page's own centre of coverage. Every texel
        // the raster wrote came off that face (the cube is the only caster and the
        // face is flat and axis-aligned), so its depth is a CONSTANT across the
        // page — which is what makes a single hand-computed value the right
        // assertion rather than a range.
        let p = glam::Vec3::new(CUBE_XY.0, CUBE_XY.1, SIDE * 0.5);
        let c = vp * p.extend(1.0);
        let want = c.z / c.w;
        for d in &hits {
            assert!(
                (d - want).abs() < 2e-3,
                "page {page:?} holds depth {d} where its own projection puts the \
                 caster's near face at {want}"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0 && total_written > 64,
        "the atlas held {total_written} written texels across {checked} pages — \
         the raster drew nothing"
    );
    // …and reverse-Z means a written texel is GREATER than the clear, always.
    // A forward-Z page would fail this and pass everything above.
    assert!(
        written(&atlas, (0, 0, atlas.0.min(atlas.1)))
            .iter()
            .all(|d| *d > inf_render::VSM_DEPTH_CLEAR),
        "a page holds depth below the reverse-Z clear"
    );
}

// ── (b) the scissor/viewport proof ──────────────────────────────────────────

/// **THE RECTANGLE PROOF**: a caster that fills its page writes **only** inside
/// that page's 128 × 128 rect, and every texel of every other slot is exactly the
/// clear value.
///
/// This is the arm the `set_viewport` / `set_scissor_rect` pair exists for, and it
/// is built to falsify them: the caster's projected footprint covers its whole
/// page, so with the viewport left at the attachment's default the same geometry
/// would be splattered across the entire atlas and every slot would be written.
/// Measured — see the ledger — deleting `set_viewport` fails this arm; deleting
/// `set_scissor_rect` does **not**, because clipping happens against the clip
/// volume before the viewport transform, so the scissor here is defence in depth
/// rather than the thing that pins the rect.
#[test]
fn a_caster_writes_inside_its_page_and_nowhere_else() {
    let Some(gpu) = gpu_or_skip("the VSM page rectangle") else {
        return;
    };
    let set = settings();
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &set, 6);
    let atlas = read_atlas(&gpu, &renderer);
    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty());

    let side = inf_render::VSM_PAGE_SIZE;
    let mut inside = 0usize;
    let mut occupied: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
    for (_, _, rect) in &pages {
        inside += written(&atlas, *rect).len();
        occupied.insert((rect.0, rect.1));
    }
    assert!(
        inside > 64,
        "the resident pages hold {inside} written texels — nothing to bound"
    );

    // Every texel of the atlas that is NOT inside a resident page's rect.
    let (w, h, ref data) = atlas;
    let mut escaped = 0usize;
    for y in 0..h {
        for x in 0..w {
            let corner = ((x / side) * side, (y / side) * side);
            if occupied.contains(&corner) {
                continue;
            }
            if data[(y * w + x) as usize] != inf_render::VSM_DEPTH_CLEAR {
                escaped += 1;
            }
        }
    }
    assert_eq!(
        escaped, 0,
        "{escaped} texels outside every page's rect were written — the page \
         viewport does not pin the 128×128 rectangle"
    );
    // ANTI-VACUITY: there really WERE slots outside the resident set, so "no
    // escapes" is a statement about a region that exists.
    let slots = (w / side) * (h / side);
    assert!(
        occupied.len() < slots as usize,
        "every slot of the atlas was resident ({} of {slots}), so the arm bounded \
         an empty region",
        occupied.len()
    );
}

/// **THE REGISTRATION PROOF** (P27.2 audit): a caster's silhouette lands on the
/// texels its page's own NDC says, not merely *somewhere* inside the slot.
///
/// `a_caster_writes_inside_its_page_and_nowhere_else` bounds the content to the
/// rect and nothing pinned it *to the rect's corner*: a viewport inset by two
/// texels — the shape a future border would introduce — is inside every rect,
/// writes every page it should, holds exactly the depth the CPU predicts, and
/// survived the whole file. It is a real defect, because P27.4 samples a page by
/// mapping a receiver's light-space position onto the slot and a two-texel shear
/// puts every shadow two texels off its caster.
///
/// So this arm compares the **bounding box of the written texels** against the
/// forward projection of the cube's own corners through `vsm_page_matrix` and the
/// viewport transform. One texel of tolerance, which is what a pixel-centre
/// coverage rule costs.
#[test]
fn a_pages_content_is_registered_to_its_slots_corner() {
    let Some(gpu) = gpu_or_skip("the VSM page registration") else {
        return;
    };
    let set = settings_with(64);
    let renderer = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &set, 6);
    let atlas = read_atlas(&gpu, &renderer);
    let pages = resident_pages(&renderer);
    assert!(!pages.is_empty());

    let half = SIDE * 0.5;
    let mut checked = 0;
    // A page whose predicted box stops strictly inside the rect on some edge — the
    // anti-vacuity that says an EDGE was compared and not just "the whole slot".
    let mut partial = 0;
    for (light, page, rect) in &pages {
        let vp = page_vp(&renderer, *light, *page);
        let (mut ox0, mut oy0, mut ox1, mut oy1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut hits = 0;
        for_each_written(&atlas, *rect, |x, y, _| {
            ox0 = ox0.min(x);
            oy0 = oy0.min(y);
            ox1 = ox1.max(x);
            oy1 = oy1.max(y);
            hits += 1;
        });
        if hits == 0 {
            continue;
        }
        // The cube's near face, whose four corners are the silhouette: the light
        // looks along −Z and the face at `z = +SIDE/2` is the one a page sees.
        let (mut nx0, mut ny0, mut nx1, mut ny1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for (dx, dy) in [(-half, -half), (half, -half), (half, half), (-half, half)] {
            let c = vp * glam::Vec3::new(CUBE_XY.0 + dx, CUBE_XY.1 + dy, half).extend(1.0);
            let (nx, ny) = (c.x / c.w, c.y / c.w);
            nx0 = nx0.min(nx);
            nx1 = nx1.max(nx);
            ny0 = ny0.min(ny);
            ny1 = ny1.max(ny);
        }
        let s = rect.2 as f32;
        let to_x = |n: f32| rect.0 as f32 + (n * 0.5 + 0.5) * s;
        // NDC y is up and a viewport's y is down, so the NDC MAXIMUM is the top row.
        let to_y = |n: f32| rect.1 as f32 + (0.5 - n * 0.5) * s;
        let lo_x = to_x(nx0).max(rect.0 as f32);
        let hi_x = to_x(nx1).min((rect.0 + rect.2) as f32) - 1.0;
        let lo_y = to_y(ny1).max(rect.1 as f32);
        let hi_y = to_y(ny0).min((rect.1 + rect.2) as f32) - 1.0;
        for (label, want, got) in [
            ("left", lo_x, ox0),
            ("right", hi_x, ox1),
            ("top", lo_y, oy0),
            ("bottom", hi_y, oy1),
        ] {
            assert!(
                (got as f32 - want).abs() <= 1.0,
                "page {page:?} in rect {rect:?}: its {label} written texel is {got} \
                 where its own projection puts the caster's silhouette at {want:.2} \
                 — the page's content is not registered to its slot"
            );
        }
        if to_x(nx0) > rect.0 as f32 + 0.5 || to_y(ny1) > rect.1 as f32 + 0.5 {
            partial += 1;
        }
        checked += 1;
    }
    assert!(
        checked > 0 && partial > 0,
        "{checked} pages compared, {partial} of them with an edge strictly inside \
         the slot — an arm that only ever saw full pages cannot see a shear"
    );
}

// ── (c) the cull is subtractive ─────────────────────────

/// **THE CULL ARM**: the GPU's own per-page verdict, read back off the device and
/// compared against an independent CPU walk of the same spheres.
///
/// It has to be the args buffer rather than the atlas, and that is the finding
/// this arm is built around: **the atlas cannot tell a culled caster from a
/// clipped one.** A caster the page's frustum rejects writes no depth whether the
/// cull dropped it or the rasterizer did, so an image-side assertion is satisfied
/// by a cull that keeps everything — the failure that costs `pages × casters`
/// vertex invocations and is invisible everywhere else. The `instance_count`
/// words the cull wrote are the only record of what it decided, and reading them
/// off the device is *mirrored ≠ measured* (P26.5) applied to a decision.
#[test]
fn the_per_page_cull_drops_the_casters_the_page_cannot_see() {
    let Some(gpu) = gpu_or_skip("the VSM per-page cull") else {
        return;
    };
    let set = settings_with(64);
    // Two cubes, far apart in x, and a wide backdrop so pages exist between and
    // around them — without it every marked page sits under a cube and there is
    // nothing for the cull to reject.
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(-4.5, CUBE_XY.1 as f64, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(SIDE),
        [1.0, 1.0, 1.0, 1.0],
        2,
    ));
    s.instances.push(backdrop());
    s.mark_dirty();
    let renderer = run(&gpu, &s, &view(6.0), &set, 6);
    let sys = renderer.vsm().expect("live");
    let raster = sys.raster_state();
    let counts = raster.read_draw_counts(&gpu);
    let pages = raster.last_pages().to_vec();
    let groups = raster.last_groups();
    assert!(
        !pages.is_empty() && groups > 0,
        "the raster ran on no pages"
    );
    assert_eq!(counts.len(), pages.len() * groups);

    // The CPU twin: every caster's sphere, exactly as `pack_casters` derives it.
    let spheres: Vec<(glam::Vec3, f32)> = s
        .instances
        .iter()
        .map(|i| {
            let c = i.translation.as_vec3();
            let r = i.mesh.bounding_radius() * i.scale.x.max(i.scale.y).max(i.scale.z);
            (c, r)
        })
        .collect();

    let mut rejected = 0usize;
    let mut kept = 0usize;
    for (i, (light, page, _)) in pages.iter().enumerate() {
        let vp = page_vp(&renderer, light.0, *page);
        let want = spheres
            .iter()
            .filter(|(c, r)| inf_render::vsm_page_sees_sphere(&vp, *c, *r))
            .count() as u32;
        // Every caster in this fixture is a cube, so group 0 holds all of them and
        // the other four groups must be empty.
        let got = counts[i * groups];
        assert_eq!(
            got, want,
            "page {page:?}: the GPU cull kept {got} casters where the same test on \
             the CPU keeps {want}"
        );
        for g in 1..groups {
            assert_eq!(counts[i * groups + g], 0, "a non-cube group drew");
        }
        rejected += spheres.len() - want as usize;
        kept += want as usize;
    }
    // ANTI-VACUITY, both directions: the cull really rejected pairs and really
    // kept some. An agreement between two functions that both answer "all" is not
    // an agreement about culling.
    assert!(
        rejected > 0 && kept > 0,
        "the cull kept {kept} and rejected {rejected} (page, caster) pairs — one \
         of the two answers was never exercised"
    );
}

// ── (d) masked materials keep their alpha test in the page raster ───

/// **A cutout shadows as a cutout.** The same caster, masked with an alpha under
/// its cutoff, writes **no** depth of its own; opaque, it writes the page.
///
/// The scene carries an opaque backdrop, and that is not decoration: a masked
/// caster whose fragments all discard writes no *camera* depth either, so the
/// marking pass would mark nothing and the arm would pass with the whole feature
/// deleted. The backdrop is what marks the pages; the claim is then made about
/// depth **nearer the light than the backdrop**, which only the cube can produce.
#[test]
fn a_masked_caster_discards_in_the_page_raster() {
    let Some(gpu) = gpu_or_skip("the VSM masked caster") else {
        return;
    };
    let set = settings_with(64);
    let v = view(6.0);

    // Texels holding depth NEARER the light than the backdrop — i.e. the cube's.
    let cube_texels = |blend: u8, alpha: f32| -> (usize, inf_render::VsmRasterStats) {
        let mut s = scene(blend, 0.5, alpha);
        s.instances.push(backdrop());
        s.mark_dirty();
        let r = run(&gpu, &s, &v, &set, 6);
        let atlas = read_atlas(&gpu, &r);
        let mut n = 0usize;
        for (light, page, rect) in resident_pages(&r) {
            let vp = page_vp(&r, light, page);
            // The light looks along -Z, so a larger z is nearer it and — under
            // reverse-Z — holds the larger depth. Halfway between the two
            // surfaces separates them with room to spare.
            let at = |z: f32| {
                let c = vp * glam::Vec3::new(CUBE_XY.0, CUBE_XY.1, z).extend(1.0);
                c.z / c.w
            };
            let mid = 0.5 * (at(SIDE * 0.5) + at(BACKDROP_Z + BACKDROP_T));
            n += written(&atlas, rect).iter().filter(|d| **d > mid).count();
        }
        (n, r.vsm_raster_stats().expect("stats"))
    };

    // Opaque: the cube writes its own depth.
    let (solid, solid_stats) = cube_texels(0, 1.0);
    assert!(solid > 16, "the opaque control wrote {solid} cube texels");
    assert_eq!(
        solid_stats.masked_frames, 0,
        "an opaque scene bound the alpha-testing pipeline"
    );

    // Masked, alpha 0.2 under a 0.5 cutoff: every one of its fragments discards.
    let (cut, cut_stats) = cube_texels(1, 0.2);
    assert!(
        cut_stats.masked_frames > 0,
        "a masked caster did not reach the alpha-testing pipeline: {cut_stats:?}"
    );
    // The pass still RAN and still drew — this is a discard, not an absence.
    assert!(
        cut_stats.draws > 0 && cut_stats.casters > 0,
        "{cut_stats:?}"
    );
    assert_eq!(
        cut, 0,
        "a fully-cut-out caster wrote {cut} depth texels — its shadow is solid \
         where its material is a hole"
    );

    // …and the same masked material with alpha ABOVE its cutoff writes again, so
    // the arm above is the alpha test rather than "masked draws nothing".
    let (keep, _) = cube_texels(1, 0.9);
    assert!(
        keep > 16,
        "a masked caster whose alpha CLEARS its cutoff wrote {keep} texels"
    );
}

// ── (e) virtualized geometry casts ────────────────────────────

/// **THE HOLE THIS PHASE NAMES.** Phase 27's goal says "every caster path casts
/// (vgeom's 'casts no shadows' hole closes here)", and before this batch it was
/// total: `passes/shadow.rs` contains no occurrence of `vgeom` or `meshlet`, and
/// `passes/vgeom.rs` contains no shadow pipeline, no light-space matrix and no
/// caster registration.
///
/// A vmesh instance now writes depth into the pages its bounds touch. The arm is
/// built to falsify a path that merely *compiles*: the counters prove vgeom
/// casters were packed, and the atlas proves depth arrived where the CPU says the
/// asset's own bounding sphere puts it.
#[test]
fn a_virtualized_geometry_instance_casts_into_the_pages_it_touches() {
    let Some(gpu) = gpu_or_skip("the VSM meshlet-asset caster") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![
            inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh")
        ],
        ..Default::default()
    };
    // Standing up, so its silhouette faces the light along +Z.
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::new(0.0, 0.0, 0.0),
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(3.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(64);
    let renderer = run(&gpu, &s, &view(9.0), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(
        stats.vgeom_casters > 0,
        "no virtualized-geometry caster was packed: {stats:?}"
    );

    // The vmesh's own depth: nearer the light than the backdrop, which is the
    // only other caster in the scene. Anything the backdrop wrote is at the
    // backdrop's depth, so a texel past the midpoint can only be the vmesh's.
    let atlas = read_atlas(&gpu, &renderer);
    let mut vgeom_texels = 0usize;
    for (light, page, rect) in resident_pages(&renderer) {
        let vp = page_vp(&renderer, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        vgeom_texels += written(&atlas, rect).iter().filter(|d| **d > mid).count();
    }
    assert!(
        vgeom_texels > 16,
        "the vmesh wrote {vgeom_texels} texels nearer the light than the \
         backdrop — it is not casting"
    );

    // ANTI-VACUITY / control: the same scene with the vmesh instance removed
    // writes NOTHING past the backdrop, so the count above is the asset's own
    // depth and not the slab's.
    let mut bare = s.clone();
    bare.vgeom_instances.clear();
    bare.mark_dirty();
    let control = run(&gpu, &bare, &view(9.0), &set, 6);
    let control_atlas = read_atlas(&gpu, &control);
    let mut control_texels = 0usize;
    for (light, page, rect) in resident_pages(&control) {
        let vp = page_vp(&control, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        control_texels += written(&control_atlas, rect)
            .iter()
            .filter(|d| **d > mid)
            .count();
    }
    assert_eq!(
        control_texels, 0,
        "the backdrop alone wrote {control_texels} texels past its own depth"
    );
    let bare_stats = control.vsm_raster_stats().expect("stats");
    assert_eq!(
        bare_stats.vgeom_casters, 0,
        "a scene with no vmesh instance packed a vmesh caster"
    );

    // **WHAT THE ASSET HANDS OVER** (island wave I8c). The counter that answered
    // this wave's VSM clause: an asset's caster group is submitted WHOLE into
    // every page it reaches, so its index count multiplies by the page count.
    // Measured on the island: 149.5 M of a 156.0 M-index frame, from ONE
    // instance. Here it is asserted as a shape rather than as a magnitude —
    // meshlet indices are a real fraction of the load, they scale with the draws
    // that carry them, and a scene with no asset submits none.
    assert!(
        stats.draws_vgeom > 0 && stats.indices_vgeom > 0,
        "the vmesh cast and no meshlet indices were submitted: {stats:?}"
    );
    // **NOT `indices_vgeom % draws_vgeom == 0`** (the I8c audit). The first
    // version of this arm asserted that, on the reasoning that "every meshlet
    // draw submits the same level's whole index range" — which stopped being
    // true in the same wave that wrote it: an asset is packed once per distinct
    // classic level its page buckets ask for, and **this fixture packs several**
    // (`the_page_lod_floor_moves_a_shadow_by_the_texel_it_is_written_into`
    // measures 3 caster records over 8 meshlet draws at the tolerance it centres
    // on, and 2 at a flat 1 px — which is the point: the count is a property of
    // the tolerance and the chain, not a constant). The divisibility held by
    // arithmetic luck over two levels' index counts. What is true of every
    // meshlet draw is that it submits whole triangles of a real level.
    assert_eq!(
        stats.indices_vgeom % 3,
        0,
        "the meshlet index load is not a whole number of triangles: {stats:?}"
    );
    assert!(
        stats.indices_vgeom >= stats.draws_vgeom * 3,
        "{} meshlet draws submitted {} indices — a draw that submits less than a \
         triangle is an empty index range: {stats:?}",
        stats.draws_vgeom,
        stats.indices_vgeom
    );
    assert!(
        stats.indices_vgeom < stats.indices_drawn,
        "the backdrop casts too, so the asset cannot be the whole index load: \
         {stats:?}"
    );
    assert_eq!(
        (bare_stats.draws_vgeom, bare_stats.indices_vgeom),
        (0, 0),
        "a scene with no vmesh instance submitted meshlet indices: {bare_stats:?}"
    );
}

// ── (g) skinned and terrain cast ─────────────────────────────

/// Depth **nearer the light than the backdrop**, summed over every resident page
/// — the measurement every "does this path cast?" arm in this file makes.
fn depth_past_backdrop(gpu: &GpuContext, renderer: &inf_render::EngineRenderer) -> usize {
    let atlas = read_atlas(gpu, renderer);
    let mut n = 0usize;
    for (light, page, rect) in resident_pages(renderer) {
        let vp = page_vp(renderer, light, page);
        let at = |z: f32| {
            let c = vp * glam::Vec3::new(0.0, 0.0, z).extend(1.0);
            c.z / c.w
        };
        let mid = 0.5 * (at(0.0) + at(BACKDROP_Z + BACKDROP_T));
        n += written(&atlas, rect).iter().filter(|d| **d > mid).count();
    }
    n
}

/// **A skinned caster casts, and it casts its POSE.** The skeleton's palette
/// reaches the page raster, so a character's shadow is the character's silhouette
/// rather than its bind pose sitting wherever the asset was authored.
///
/// The control is the same instance with a palette that translates it out of the
/// light's reach: same mesh, same instance transform, same everything but the
/// matrices — so what the arm measures is the skinning, not the presence of a
/// draw call.
#[test]
fn a_skinned_caster_casts_through_its_own_palette() {
    let Some(gpu) = gpu_or_skip("the VSM skinned caster") else {
        return;
    };
    // A unit quad in the XY plane, every vertex bound to joint 0.
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });

    let build = |palette: glam::Mat4| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        s.skinned.push(inf_render::SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::new(0.0, CUBE_XY.1 as f64, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 3,
            mesh: 0,
            palette: std::sync::Arc::new(vec![palette]),
            shadow: inf_render::SkinnedShadow::BindSphere,
            vt: inf_render::VtTextureSet::NONE,
            sections: Vec::new(),
        });
        s.instances.push(backdrop());
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        s.mark_dirty();
        s
    };

    let set = settings_with(64);
    let posed = run(&gpu, &build(glam::Mat4::IDENTITY), &view(6.0), &set, 6);
    let stats = posed.vsm_raster_stats().expect("stats");
    assert!(
        stats.skinned_casters > 0,
        "no skinned caster was packed: {stats:?}"
    );
    let lit = depth_past_backdrop(&gpu, &posed);
    assert!(lit > 16, "the skinned quad wrote {lit} texels");

    // The SAME instance, posed 50 m behind the backdrop by its palette alone.
    // If the palette were ignored, the quad would sit where the bind pose puts it
    // and this count would match the one above.
    let far = run(
        &gpu,
        &build(glam::Mat4::from_translation(glam::Vec3::new(
            0.0, 0.0, -50.0,
        ))),
        &view(6.0),
        &set,
        6,
    );
    assert_eq!(
        depth_past_backdrop(&gpu, &far),
        0,
        "the joint palette did not move the caster — the page raster is drawing \
         the bind pose"
    );
    assert!(
        far.vsm_raster_stats().expect("stats").skinned_casters > 0,
        "the control packed no skinned caster, so it proves nothing"
    );
}

/// **A terrain tile casts**, out of its own heights rather than out of the
/// camera-fitted clipmap patch.
#[test]
fn a_terrain_tile_casts_from_its_own_heights() {
    let Some(gpu) = gpu_or_skip("the VSM terrain caster") else {
        return;
    };
    // One 33-sample tile, 0.5 m a sample, with a ridge along x that stands well
    // in front of the backdrop.
    const RES: u32 = 33;
    const MPS: f64 = 0.5;
    let span = (RES as f64 - 1.0) * MPS;
    let mut heights = vec![0f32; (RES * RES) as usize];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..RES {
        for i in 0..RES {
            // A ramp in z: the tile's own surface leans toward the light.
            let h = 2.0 - 3.0 * (j as f32 / (RES - 1) as f32);
            heights[(j * RES + i) as usize] = h;
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    let terrain = inf_render::RenderTerrain {
        id: 7,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: vec![inf_render::RenderTerrainTile {
            key: inf_render::TerrainTileKey::lod0((0, 0)),
            origin: glam::DVec3::new(-0.5 * span, 0.0, -0.5 * span),
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (lo, hi),
            holes: Vec::new(),
            version: 1,
        }],
        layers: Default::default(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    };

    let mut s = scene(0, 0.5, 1.0);
    s.terrains.push(terrain);
    s.mark_dirty();
    let set = settings_with(64);
    let renderer = run(&gpu, &s, &view(8.0), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(
        stats.terrain_casters > 0,
        "no terrain tile was packed as a caster: {stats:?}"
    );
    // The heightfield is the ONLY thing in this scene at those depths besides the
    // cube, and the cube is a metre across; the tile spans 16 m. So a page that
    // holds terrain depth holds far more written texels than the cube alone can
    // explain — measured against the same scene with the terrain removed.
    let with_terrain = {
        let atlas = read_atlas(&gpu, &renderer);
        resident_pages(&renderer)
            .iter()
            .map(|(_, _, r)| written(&atlas, *r).len())
            .sum::<usize>()
    };
    let mut bare = s.clone();
    bare.terrains.clear();
    bare.mark_dirty();
    let control = run(&gpu, &bare, &view(8.0), &set, 6);
    let without = {
        let atlas = read_atlas(&gpu, &control);
        resident_pages(&control)
            .iter()
            .map(|(_, _, r)| written(&atlas, *r).len())
            .sum::<usize>()
    };
    assert_eq!(
        control.vsm_raster_stats().expect("stats").terrain_casters,
        0,
        "a scene with no terrain packed a terrain caster"
    );
    assert!(
        with_terrain > without + 64,
        "the terrain added {} written texels over a control of {without} — it is \
         not casting",
        with_terrain.saturating_sub(without)
    );

    // **WHAT THE PASS HANDS OVER** (island wave I8c). The counters above count
    // *asks*; a draw's price is its index count, and a terrain group's is the
    // whole decimated tile — `min(VSM_TERRAIN_CASTER_CELLS, res − 1)` cells
    // squared, six indices a quad — submitted into every page the tile reaches,
    // however little of the world that page covers. That product is the number
    // island wave I8c's VSM clause is aimed at, so it is measured rather than
    // reasoned about.
    let cells = inf_render::VSM_TERRAIN_CASTER_CELLS.min(RES - 1);
    let per_tile = u64::from(cells * cells * 6);
    assert!(
        stats.draws_terrain > 0,
        "the terrain cast and no terrain draw was issued: {stats:?}"
    );
    assert_eq!(
        stats.indices_terrain,
        stats.draws_terrain * per_tile,
        "each terrain draw submits the whole {per_tile}-index tile, so the two \
         counters must be that product: {stats:?}"
    );
    assert!(
        stats.indices_terrain < stats.indices_drawn,
        "the cube casts too, so the terrain cannot be the whole index load: \
         {stats:?}"
    );
    // …and the histogram tiles the page set, exactly as the dirty split tiles
    // the dirty one.
    assert_eq!(
        stats.pages_by_level.iter().sum::<u64>(),
        stats.pages,
        "the level histogram does not tile the rastered pages: {stats:?}"
    );
    let bare_stats = control.vsm_raster_stats().expect("stats");
    assert_eq!(
        (bare_stats.draws_terrain, bare_stats.indices_terrain),
        (0, 0),
        "a scene with no terrain submitted terrain indices: {bare_stats:?}"
    );
    assert!(
        bare_stats.indices_drawn > 0,
        "the control drew nothing at all, so the comparison above is vacuous: \
         {bare_stats:?}"
    );
}

/// One planar terrain tile: `height(u, v) = a + b·u + c·v` over the tile's own
/// samples, so the caster mesh's triangulation reproduces it **exactly** whatever
/// the decimation does and a depth arm can assert metres rather than texels.
#[derive(Clone, Copy)]
struct PlanarTile {
    key: inf_render::TerrainTileKey,
    origin: glam::DVec3,
    plane: (f32, f32, f32),
}

const TILE_RES: u32 = 33;
const TILE_MPS: f64 = 0.5;
const TILE_SPAN: f64 = (TILE_RES as f64 - 1.0) * TILE_MPS;

impl PlanarTile {
    fn height(&self, u: f32, v: f32) -> f32 {
        self.plane.0 + self.plane.1 * u + self.plane.2 * v
    }
    /// The world height of this tile's surface at render-local `(x, z)`, or `None`
    /// outside its own footprint.
    fn world_y(&self, x: f32, z: f32) -> Option<f32> {
        let u = (x as f64 - self.origin.x) / TILE_SPAN;
        let v = (z as f64 - self.origin.z) / TILE_SPAN;
        // A texel exactly on the edge is inside; the slack is one sample.
        let e = TILE_MPS / TILE_SPAN;
        ((-e..=1.0 + e).contains(&u) && (-e..=1.0 + e).contains(&v))
            .then(|| self.origin.y as f32 + self.height(u as f32, v as f32))
    }
    fn build(&self) -> inf_render::RenderTerrainTile {
        let mut heights = vec![0f32; (TILE_RES * TILE_RES) as usize];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for j in 0..TILE_RES {
            for i in 0..TILE_RES {
                let h = self.height(
                    i as f32 / (TILE_RES - 1) as f32,
                    j as f32 / (TILE_RES - 1) as f32,
                );
                heights[(j * TILE_RES + i) as usize] = h;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        inf_render::RenderTerrainTile {
            key: self.key,
            origin: self.origin,
            heights,
            weights: Vec::new(),
            biomes: Vec::new(),
            height_bounds: (lo, hi),
            holes: Vec::new(),
            version: 1,
        }
    }
}

fn planar_terrain(tiles: &[PlanarTile]) -> inf_render::RenderTerrain {
    inf_render::RenderTerrain {
        id: 7,
        tile_resolution: TILE_RES,
        meters_per_sample: TILE_MPS,
        tiles: tiles.iter().map(PlanarTile::build).collect(),
        layers: Default::default(),
        macro_variation: 0.0,
        biome_palette: Vec::new(),
    }
}

/// A camera above the ground looking down at it, so the tiles mark pages.
fn terrain_view() -> RenderView {
    RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 14.0, 14.0),
        forward: glam::Vec3::new(0.0, -1.0, -1.0).normalize(),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// A scene of nothing but planar terrain under an overhead sun — so **every**
/// written texel of the atlas is ground and there is nothing else for a residual
/// to be blamed on.
fn terrain_scene(tiles: &[PlanarTile]) -> RenderScene {
    let mut s = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    s.terrains.push(planar_terrain(tiles));
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        // Straight up: the direction TO the light, so the sun looks down and a
        // page's depth is world height. Nothing here depends on the light basis —
        // the page matrix is INVERTED rather than assumed.
        direction: glam::Vec3::Y,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();
    s
}

/// Every written atlas texel, reconstructed into render-local metres and compared
/// against the nearest tile surface below it. Returns `(texels checked, worst
/// residual in metres, texels that landed on no tile at all)`.
fn terrain_residuals(
    gpu: &GpuContext,
    renderer: &inf_render::EngineRenderer,
    tiles: &[PlanarTile],
) -> (usize, f32, usize) {
    let atlas = read_atlas(gpu, renderer);
    let (mut checked, mut worst, mut orphan) = (0usize, 0f32, 0usize);
    for (light, page, rect) in resident_pages(renderer) {
        let inv = page_vp(renderer, light, page).inverse();
        for_each_written(&atlas, rect, |x, y, d| {
            let p = texel_world(inv, rect, x, y, d);
            let best = tiles
                .iter()
                .filter_map(|t| t.world_y(p.x, p.z))
                .map(|y| (y - p.y).abs())
                .fold(f32::INFINITY, f32::min);
            if best.is_finite() {
                checked += 1;
                worst = worst.max(best);
            } else {
                orphan += 1;
            }
        });
    }
    (checked, worst, orphan)
}

/// **THE TERRAIN SURFACE ARM** (P27.2 audit): the depth a page holds over ground
/// is the tile's **own** surface, in metres, at the world height the tile's origin
/// and heights put it.
///
/// The first terrain arm counted texels against a control — which says the tile
/// casts *something* and nothing about *what*. Measured: transposing the height
/// index (`heights[si·res + sj]`) and dropping the tile origin's `y` from the
/// vertex both survived the whole file, the second one because the fixture's tile
/// origin was `y = 0` and a fixture that cannot distinguish is not a control. This
/// arm reconstructs every written texel through the page matrix's inverse and
/// compares it to `origin.y + height(u, v)`.
///
/// The fixture's height field is **planar** and its origin is off the ground plane
/// in all three axes, so the assertion is exact under any triangulation and no term
/// of the composition is zero.
#[test]
fn a_terrain_page_holds_the_tiles_own_surface_in_metres() {
    let Some(gpu) = gpu_or_skip("the VSM terrain surface") else {
        return;
    };
    let tiles = [PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        // Off-origin on every axis. `y = 3.5` is the term the old fixture zeroed.
        origin: glam::DVec3::new(-0.5 * TILE_SPAN, 3.5, -0.5 * TILE_SPAN),
        // Sloped on BOTH axes, and by different amounts, so a transposed height
        // index is a different surface rather than the same one.
        plane: (1.0, 2.5, -4.0),
    }];
    let set = settings_with(64);
    let renderer = run(&gpu, &terrain_scene(&tiles), &terrain_view(), &set, 6);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.terrain_casters > 0, "{stats:?}");

    let (checked, worst, orphan) = terrain_residuals(&gpu, &renderer, &tiles);
    assert!(
        checked > 512,
        "only {checked} ground texels were reconstructed — the arm bounded almost \
         nothing"
    );
    assert_eq!(
        orphan, 0,
        "{orphan} written texels reconstruct to a point over no tile at all"
    );
    assert!(
        worst < 0.05,
        "a page's depth puts the ground {worst} m off the tile's own surface"
    );
}

/// **The terrain caster cache is keyed on the tile, not on its place in a
/// streaming list** (P27.2 audit) — `b921fd3`'s fix, which shipped without an arm.
///
/// The list is a residency: tiles arrive and leave, and the entry that was index 0
/// last frame belongs to a different tile this frame. Keyed on the index, a tile
/// that streams in over an evicted one's slot inherits the evicted one's *mesh*
/// whenever the two share a version stamp — and the two always do, because a tile's
/// version starts at 1.
///
/// So: render a tile, evict it, stream a different tile into its place at the same
/// version, and assert the ground is at the NEW tile's height. Under the old key
/// the residual is the whole difference between the two.
#[test]
fn a_streamed_in_terrain_tile_does_not_inherit_the_evicted_ones_mesh() {
    let Some(gpu) = gpu_or_skip("the VSM terrain cache key") else {
        return;
    };
    // `keep` is resident throughout; `first` is evicted and `second` takes its
    // place in the list, at a DIFFERENT height and a different tile key.
    let keep = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((1, 0)),
        origin: glam::DVec3::new(0.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (1.0, 0.0, 0.0),
    };
    let first = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        origin: glam::DVec3::new(-1.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (0.0, 0.0, 0.0),
    };
    let second = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 1)),
        origin: glam::DVec3::new(-1.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        // Six metres above where the evicted tile's mesh sits.
        plane: (6.0, 0.0, 0.0),
    };
    let set = settings_with(64);
    let before = terrain_scene(&[first, keep]);
    let after_tiles = [second, keep];
    let after = terrain_scene(&after_tiles);
    let v = terrain_view();
    let renderer = run_sequence(&gpu, &[(&before, 6), (&after, 6)], &v, &set);
    assert!(
        renderer.vsm_raster_stats().expect("stats").terrain_casters > 0,
        "no terrain caster survived the swap"
    );

    let (checked, worst, orphan) = terrain_residuals(&gpu, &renderer, &after_tiles);
    assert!(checked > 512, "only {checked} ground texels after the swap");
    assert_eq!(orphan, 0, "{orphan} texels over no resident tile");
    assert!(
        worst < 0.05,
        "after a tile was evicted and another streamed into its place, the ground \
         is {worst} m off the resident tiles' own surfaces — the caster mesh is the \
         EVICTED tile's, inherited through a cache keyed on a streaming index"
    );
    // ANTI-VACUITY: the evicted tile's mesh sat six metres below the one that
    // replaced it, so a stale mesh would have been far outside the tolerance above.
    let over_the_swap = -(TILE_SPAN as f32);
    let now = second
        .world_y(over_the_swap, 0.0)
        .expect("inside the new tile");
    let then = first
        .world_y(over_the_swap, 0.0)
        .expect("inside the old one");
    assert!(
        (now - then).abs() > 5.0,
        "the two tiles are {} m apart, which the {} m tolerance would not have seen",
        (now - then).abs(),
        0.05
    );
}

// ── (h) off path, and the settings door ─────────────────────────────────────

/// With virtual shadows off, the caster pass never opens — the byte-stability
/// guarantee every golden rests on, as a counter rather than a hope.
#[test]
fn a_renderer_with_virtual_shadows_off_rasterizes_no_page() {
    let Some(gpu) = gpu_or_skip("the VSM off path") else {
        return;
    };
    let off = VsmSettings {
        enabled: false,
        ..settings()
    };
    let r = run(&gpu, &scene(0, 0.5, 1.0), &view(5.0), &off, 4);
    assert_eq!(r.vsm_raster_frames(), 0);
    assert!(r.vsm_raster_stats().is_none());
    assert!(r.vsm().is_none());

    // …and a scene whose only light does not cast: the setting is ON and the pass
    // still never opens, which is the other half of the off path.
    let mut dark = scene(0, 0.5, 1.0);
    dark.lights[0].cast_shadows = false;
    dark.mark_dirty();
    let r = run(&gpu, &dark, &view(5.0), &settings(), 4);
    assert_eq!(r.vsm_raster_frames(), 0);
}

/// **The settings boundary, at the renderer's own door** (P27.2): an illegal
/// virtual-shadow configuration is refused and nothing is applied — not the legal
/// half either.
#[test]
fn the_renderer_refuses_an_illegal_shadow_configuration_whole() {
    let Some(gpu) = gpu_or_skip("the VSM settings boundary") else {
        return;
    };
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let before = *renderer.settings();
    let mut bad = before;
    // One legal change and one illegal one in the same block.
    bad.exposure = 3.25;
    bad.vsm = VsmSettings {
        clipmap_pages_per_side: 65_536,
        ..Default::default()
    };
    let err = renderer
        .try_set_settings(bad)
        .expect_err("a 2^32-page grid was accepted");
    assert!(
        matches!(err, inf_render::VsmSettingsError::PageSpace { .. }),
        "{err}"
    );
    assert_eq!(
        *renderer.settings(),
        before,
        "a refused settings block applied its legal half"
    );
    // The infallible door refuses too — it logs instead of returning.
    renderer.set_settings(bad);
    assert_eq!(*renderer.settings(), before);
    // ANTI-VACUITY: the same block minus the illegal field DOES apply.
    let mut good = bad;
    good.vsm = VsmSettings::default();
    assert!(renderer.try_set_settings(good).is_ok());
    assert_eq!(renderer.settings().exposure, 3.25);
}

// ── (i) the P27.2 audit's arms ──────────────────────────────────────────────

/// **THE POSE MARGIN, MEASURED** (P27.2 audit): the cull sphere a skinned caster
/// is tested with contains a pose that has **left** its bind-pose bound.
///
/// `SKINNED_POSE_MARGIN` carries the batch's own reasoning — "a skeleton moves
/// vertices, so a bind-pose bound is not conservative for an arbitrary pose, and
/// culling on a bound the pose escapes would delete a limb's shadow at exactly the
/// moment the limb moved" — and setting it to `0.0` survived every arm in this
/// file. It has to: a bound that is too tight only loses the pages the escaped limb
/// reached, and at the levels a 256 × 144 fixture marks, a page is metres wide.
///
/// So the assertion is on the **shipped caster record** — the sphere the GPU cull
/// actually ran — against the posed vertices the raster actually draws. Both halves
/// are needed: the pose is inside the margined sphere, and it is outside the
/// bind-pose one, which is what makes the margin the thing under test rather than
/// the sphere.
#[test]
fn a_skinned_casters_cull_sphere_contains_a_pose_that_left_the_bind_pose() {
    let Some(gpu) = gpu_or_skip("the VSM skinned pose margin") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let verts = [v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)];
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: verts.to_vec(),
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    // 0.6 m along x: the far corner ends up 1.89 m from the bind centre, which is
    // outside the bind sphere (1.41 m) and inside the margined one (2.12 m).
    let palette = glam::Mat4::from_translation(glam::Vec3::new(0.6, 0.0, 0.0));
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    s.skinned.push(inf_render::SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        translation: glam::DVec3::new(0.0, CUBE_XY.1 as f64, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0; 3],
        id: 3,
        mesh: 0,
        palette: std::sync::Arc::new(vec![palette]),
        shadow: inf_render::SkinnedShadow::BindSphere,
        vt: inf_render::VtTextureSet::NONE,
        sections: Vec::new(),
    });
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let renderer = run(&gpu, &s, &view(6.0), &settings_with(64), 6);
    let sys = renderer.vsm().expect("live");
    let casters = sys.raster_state().last_casters();
    // The rigid groups come first and are exactly one per primitive kind, so any
    // group past them is this scene's only other caster: the skinned quad.
    let c = casters
        .iter()
        .find(|c| c.ids[0] >= inf_render::VSM_RIGID_GROUPS)
        .expect("a skinned caster record was packed");
    let centre = glam::Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]);
    let radius = c.sphere[3];
    let model = glam::Mat4::from_cols_array(&c.model);
    let mut worst = 0f32;
    for vert in &verts {
        // What `vsm_skinned.wgsl` draws: the palette, then the caster's model.
        let skinned = (palette * glam::Vec3::from(vert.pos).extend(1.0)).truncate();
        worst = worst.max((model.transform_point3(skinned) - centre).length());
    }
    assert!(
        worst <= radius + 1e-4,
        "the posed quad reaches {worst} m from its cull sphere's centre and the \
         sphere is {radius} m — the page cull can delete this caster from a page \
         its own geometry covers"
    );
    // ANTI-VACUITY, and the whole point: the BIND pose's own bound does not contain
    // it. Without the margin the assertion above is the one that fails.
    let bind = radius / (1.0 + inf_render::SKINNED_POSE_MARGIN);
    assert!(
        worst > bind,
        "the fixture's pose stays inside the bind-pose bound ({worst} m against \
         {bind} m), so it proves nothing about the margin"
    );
    // …and the pose really is drawn, so this is a claim about a caster that casts.
    assert!(renderer.vsm_raster_stats().expect("stats").skinned_casters > 0);
}

/// **The caster ceiling counts what it refuses** (P27.2 audit) — `266acda`'s fix,
/// which shipped without an arm.
///
/// `VSM_MAX_CASTERS` was a bare `break` before that commit: a level past 16 384
/// casters had its tail stop casting with no counter and no log line. The fix added
/// the counter; nothing exercised it, and zeroing the increment survives every
/// other arm in this file because no fixture with one cube in it reaches the
/// ceiling. This one reaches it.
#[test]
fn the_caster_ceiling_counts_the_casters_it_refuses() {
    let Some(gpu) = gpu_or_skip("the VSM caster ceiling") else {
        return;
    };
    const OVER: u32 = 100;
    let mut s = scene(0, 0.5, 1.0);
    // A dense slab of cubes behind the fixture's own one. They do not have to be
    // visible — `pack_casters` walks the scene, not the frame — but they are packed
    // in scene order, so the ones past the ceiling are the ones refused.
    let total = inf_render::VSM_MAX_CASTERS + OVER;
    for i in 1..total {
        let (x, y) = ((i % 128) as f64 * 0.05 - 3.2, (i / 128) as f64 * 0.05 - 3.2);
        s.instances.push(inf_render::MeshInstance::lit(
            glam::DVec3::new(x, y, -1.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.04),
            [1.0, 1.0, 1.0, 1.0],
            100 + i,
        ));
    }
    s.mark_dirty();
    let renderer = run(&gpu, &s, &view(5.0), &settings(), 3);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.frames > 0, "the pass never opened: {stats:?}");
    assert_eq!(
        stats.casters,
        u64::from(inf_render::VSM_MAX_CASTERS) * stats.frames,
        "the ceiling did not bound the packed set: {stats:?}"
    );
    assert_eq!(
        stats.dropped_casters,
        u64::from(OVER) * stats.frames,
        "{} casters over the ceiling were refused and {} were counted — a silent \
         cap is how the far half of a level stops casting",
        u64::from(OVER) * stats.frames,
        stats.dropped_casters
    );
    // ANTI-VACUITY: the group ceiling is NOT what refused them, so the two counters
    // are telling different stories rather than one story twice.
    assert_eq!(stats.dropped_groups, 0, "{stats:?}");
    // …and `scatter_casters` counts what the SCATTER pack contributed, not what
    // the merged bucket holds (P27.3 audit). This fixture has 16 384 rigid
    // instances and no scatter batch at all, so the honest answer is zero — and a
    // counter that counted the merged bucket would report every one of them.
    assert_eq!(
        stats.scatter_casters, 0,
        "a scene with no scatter batch reported {} scattered casters — the \
         counter is counting the merged bucket rather than the pack's survivors",
        stats.scatter_casters
    );
    // …and the summary a host reads says the number.
    let line = renderer.vsm_summary().expect("a live system");
    assert!(
        line.contains(&format!("{} casters dropped", stats.dropped_casters)),
        "{line}"
    );
}

/// **A vgeom caster draws the level the CAMERA justifies** (P27.2 audit) — the
/// deviation memo's load-bearing sentence, measured.
///
/// `docs/memos/p27-2-vgeom-casters.md` rules that virtualized geometry casts
/// through "the same `pick_classic_level` against the same `lod_threshold`, at the
/// same `VgeomSettings::pixel_error`". Nothing could see it: a caster drawn from
/// the coarsest level of the chain lands in the same pages at almost the same
/// depths, so replacing the pick with `errors.len() - 1` survived every arm in this
/// file. `VsmRasterStats::vgeom_level_sum` is the counter that makes the sentence a
/// measurement, and `pixel_error` is the input the ruling names.
#[test]
fn a_vgeom_casters_level_is_the_one_its_pixel_error_justifies() {
    let Some(gpu) = gpu_or_skip("the VSM vgeom LOD") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![
            inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh")
        ],
        ..Default::default()
    };
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::ZERO,
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(3.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(64);
    let at = |pixel_error: f32| {
        let r = run_tuned(&gpu, &[(&s, 6)], &view(9.0), &set, move |rs| {
            rs.vgeom.pixel_error = pixel_error;
        });
        r.vsm_raster_stats().expect("stats")
    };
    // A tenth of a pixel of tolerated error: the finest level of the chain.
    let fine = at(0.1);
    // …and a tolerance no level can miss: the coarsest.
    let coarse = at(400.0);
    assert!(
        fine.vgeom_casters > 0 && coarse.vgeom_casters > 0,
        "one of the two runs packed no vgeom caster at all ({fine:?} / {coarse:?})"
    );
    // **A tight tolerance does not saturate and a loose one does exactly.** The
    // equality this used to be — one caster, level 0 — stopped being the shape of
    // the answer at island wave I8c: a meshlet caster is packed once per distinct
    // level its page BUCKETS ask for, so a tight tolerance splits the record
    // across the finer half of the chain and a loose one collapses it onto the
    // coarsest. Both halves are read against the chain's own length, which is
    // what makes them claims about `pick_classic_level` rather than about a
    // number this fixture happens to produce.
    let coarsest = (inf_vgeom::VgeomMesh::classic_lods(&mesh).len() - 1) as u64;
    assert!(
        coarsest > 0,
        "the fixture mesh has no LOD chain to pick from"
    );
    assert!(
        fine.vgeom_level_sum < fine.vgeom_casters * coarsest,
        "every caster at a tenth of a pixel of tolerated error drew the coarsest \
         level {coarsest} of the chain: {fine:?}"
    );
    assert_eq!(
        coarse.vgeom_level_sum,
        coarse.vgeom_casters * coarsest,
        "400 px of tolerated error did not put every caster on the coarsest level \
         {coarsest}: {coarse:?}"
    );
    // …and the tolerance is what moves it: the MEAN level rises, compared by
    // cross-multiplication because the two runs no longer pack the same number of
    // records.
    assert!(
        coarse.vgeom_level_sum * fine.vgeom_casters > fine.vgeom_level_sum * coarse.vgeom_casters,
        "the same caster at 400 px of tolerated error drew the same mean level — \
         the page raster is not picking through `pick_classic_level` at all \
         ({fine:?} / {coarse:?})"
    );
    // **AND THE POINT OF THE PICK, IN INDICES** (island wave I8c). The level is
    // not a label: a coarser one is less geometry submitted into the same pages
    // over the same number of draws, which is the whole of this wave's VSM clause.
    assert_eq!(
        (fine.draws_vgeom, fine.pages),
        (coarse.draws_vgeom, coarse.pages),
        "the two runs drew different page/draw sets, so the index comparison \
         below is not about the level ({fine:?} / {coarse:?})"
    );
    assert!(
        coarse.indices_vgeom * 4 < fine.indices_vgeom,
        "400 px of tolerated error submitted {} indices against {} at 0.1 px over \
         the same {} draws — the coarse level is not coarser geometry",
        coarse.indices_vgeom,
        fine.indices_vgeom,
        fine.draws_vgeom
    );
}

/// **THE TOLERANCE A PAGE-QUALITY ARM MEASURES AT, READ OFF THE DAG IT ACTUALLY
/// BUILT** (the island wave I8c CI-red fix, 2026-08-27).
///
/// # The macOS red this exists to close
///
/// `the_page_lod_floor_moves_a_shadow_by_the_texel_it_is_written_into` bounds a
/// shadow's displacement in **texels of the page it is written into**, which is
/// the claim; the number it compares is the deviation of whichever classic cut
/// the page drew, which is **`meshopt`'s**. The P18 law says that output is not
/// cross-platform, and here is what that cost: at a flat `pixel_error` of 1.0
/// this fixture's level-3 page had a tolerance of `0.031250` of object space
/// against a chain whose level-3 error is `0.031951` — **2.2 % apart**. On
/// x86_64 the page drew cut 2 and moved 0.79 of a texel; on aarch64 the same
/// 2.2 % went the other way, the page drew cut **3**, and it moved **2.05**.
/// The renderer was right in both runs. The *fixture* was standing on a pick
/// boundary, which is `test_support`'s own documented failure — *"a flythrough
/// tuned clear of an error boundary on Windows landed on the far side of it on
/// macOS"* — one suite over.
///
/// # The remedy, which is `budget_for_pages`'s
///
/// A test budget is *"counted in pages read off the live page directory, never
/// in bytes measured somewhere else"*, and it *"sits at the midpoint of the open
/// interval … so a platform whose page bytes differ moves both endpoints **and**
/// the midpoint together"*. This is that rule for a tolerance instead of a
/// budget. `pixel_error` scales every bucket's threshold linearly, so the pick
/// boundaries are the values `error / unit` — one per (chain error, page) pair —
/// and the tolerance to measure at is the **geometric midpoint of the widest
/// interval between two consecutive boundaries over which the pages still draw
/// more than one distinct cut**. Both endpoints are read off the chain this
/// platform built, so the midpoint moves with them.
///
/// `units` is one entry per distinct page, already reduced to what multiplies
/// `pixel_error`: `max(the page's world-per-texel / max_scale, the camera's own
/// threshold per pixel)` — the caster pack's `max(page term, camera floor)` with
/// the linear factor divided out. Returns `(pixel_error, margin)`, the margin
/// being the multiplicative distance to the nearest boundary on **either** side
/// (so `1.414` means every threshold can move 41 % in either direction before a
/// page changes cut). Measured on this fixture: **`0.723`, at a margin of
/// `1.414`**, against the `1.022` the flat 1.0 px stood at.
///
/// The window is a factor of four either side of the shipped 1 px: a tolerance
/// outside it is not the setting a shipped frame uses, and an arm that wandered
/// there would be measuring a different question.
fn centred_pixel_error(errors: &[f32], units: &[f32]) -> (f32, f32) {
    const LO: f64 = 0.25;
    const HI: f64 = 4.0;
    let mut crit = vec![LO, HI];
    for &u in units {
        for &b in errors.iter().skip(1) {
            let pe = f64::from(b) / f64::from(u);
            if pe > LO && pe < HI {
                crit.push(pe);
            }
        }
    }
    crit.sort_by(|a, b| a.partial_cmp(b).expect("the boundaries are finite"));
    // (margin, pixel_error) of the best interval seen.
    let mut best: Option<(f64, f64)> = None;
    for w in crit.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi <= lo {
            continue;
        }
        let mid = (lo * hi).sqrt();
        // **The interval has to be one the arm can measure anything in.** Every
        // page drawing the SAME cut is a legal renderer and a vacuous arm: the
        // atlas then shows no per-page differentiation at all, which is the one
        // thing the page-LOD floor exists to produce.
        let picks: std::collections::BTreeSet<usize> = units
            .iter()
            .map(|&u| inf_vgeom::pick_classic_level(errors, (mid * f64::from(u)) as f32))
            .collect();
        if picks.len() < 2 {
            continue;
        }
        let margin = (hi / lo).sqrt();
        let better = match best {
            None => true,
            // Ties break toward the shipped 1.0 px, so the arm measures as close
            // to the real setting as the chain allows.
            Some((m, p)) => {
                margin - m > 1e-9 || ((m - margin).abs() <= 1e-9 && mid.ln().abs() < p.ln().abs())
            }
        };
        if better {
            best = Some((margin, mid));
        }
    }
    let (margin, pe) = best.expect(
        "no tolerance within a factor of four of 1 px makes these pages draw two \
         different cuts of this chain — the fixture cannot exercise a page-LOD \
         floor at all, which is a broken premise and not a bound to widen",
    );
    (pe as f32, margin as f32)
}

/// **WHAT THE PAGE-LOD FLOOR COSTS A SHADOW, IN METRES OF THE SURFACE THAT CASTS
/// IT** (the island wave I8c audit) — the arm the wave's VSM clause did not have.
///
/// I8c stopped submitting a meshlet asset at the camera's classic level into
/// every page and started picking the level each page's own world-per-texel
/// justifies. That is 149.5 M indices a frame saved and it is also, by
/// construction, **a coarser silhouette written into a coarser page**. The wave
/// measured the milliseconds and asserted the *partition* (a page draws the asset
/// once); nothing measured the picture.
///
/// This does, against the only reference that exists: the same scene with the
/// tolerance collapsed, which is the finest cut of the chain in every page — the
/// geometry P27.2 submitted. The difference between the two atlases is the depth
/// the floor moved, and it is reported in metres **and in texels of the page it
/// happened in**, because "one texel of the page it is written into" is the whole
/// claim.
///
/// The bound asserted is that claim plus the camera's own term, which was always
/// there: a page's tolerance is `max(pixel_error × its texel, the camera's world
/// tolerance)`, so a shadow surface can move by that and no more — with **one**
/// texel of slack on top, for the rasterizer's own half-texel at each of the two
/// cuts.
///
/// **The tolerance is not a constant, and that was a macOS red** (the I8c CI-red
/// fix, 2026-08-27). A flat 1.0 px put this fixture's level-3 page 2.2 % away
/// from a pick boundary of the chain `meshopt` happened to build, so the same
/// correct renderer drew cut 2 there on x86_64 and cut 3 on aarch64 — 0.79 of a
/// texel against 2.05, against a bound of two. The bound and the property are
/// right; what was measured on somebody's machine was the *pick*. The tolerance
/// now comes from [`centred_pixel_error`], which reads the boundaries off the
/// chain this platform built and sits at the midpoint between them — the
/// `budget_for_pages` doctrine, applied to a tolerance. On x86_64 that is
/// **0.723 px at a margin of 1.414**, and it reproduces the picks (cut 2 into a
/// level-3 page, cut 3 into a level-4 one) the flat 1.0 px reached by luck.
///
/// **Measured on an RTX 4070 Ti** (the I8c audit, unmoved by the fix): 4 972 of
/// 111 332 shared texels move (4.47 %) and the worst is **0.1484 m — 0.79 of one
/// texel of the level-4 page it happened in**. Which is the verdict the wave's
/// ledger did not have: the floor moves a shadow by under a texel of the page
/// that holds it, and the receiver's own bias is a whole normal texel plus
/// `(R + ½)·√2` of slope, so the coarser cut cannot open acne the bias does not
/// already cover. Mutation-verified, and harder at the centred tolerance than at
/// the flat one it replaced: a floor that tolerates **eight** texels instead of
/// one moves a level-4 page by **1.0572 m — 5.64 texels — against the 0.3231 m
/// that page tolerates**, a 3.3× overshoot where the flat 1 px gave 0.7734
/// against 0.3750 (2.06×). This arm names it either way.
#[test]
fn the_page_lod_floor_moves_a_shadow_by_the_texel_it_is_written_into() {
    let Some(gpu) = gpu_or_skip("the VSM page-LOD floor's shadow quality") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let asset = inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh");
    let bounds = asset.bounds();
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![asset],
        ..Default::default()
    };
    let rot = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let scale = glam::Vec3::splat(3.0);
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::ZERO,
        rot,
        scale,
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(64);
    let v = view(9.0);
    let at = |pixel_error: f32| {
        run_tuned(&gpu, &[(&s, 6)], &v, &set, move |rs| {
            rs.vgeom.pixel_error = pixel_error;
        })
    };
    // The reference: a tolerance so tight that every bucket and the camera alike
    // resolve the finest cut, which is what P27.2 submitted into every page. It
    // runs first because its page directory is the ladder the shipped run's
    // tolerance is chosen against.
    let finest = at(1.0e-3);

    // The camera's own world tolerance, through the door the caster pack picks
    // with — it was the whole rule before I8c and it is still the floor. Taken
    // PER PIXEL of tolerated error, because `lod_threshold` is linear in
    // `pixel_error` and one call then carries every tolerance below.
    let max_scale = scale.abs().max_element();
    let model = v.origin.model_matrix(glam::DVec3::ZERO, rot, scale);
    let camera_unit = inf_render::passes::vgeom::lod_threshold(
        v.eye_local(),
        model.transform_point3(glam::Vec3::from(bounds.0)),
        bounds.1 * max_scale,
        max_scale,
        &v,
        1.0,
    );
    // **The tolerance, read off the DAG this platform built** (the I8c CI-red
    // fix) — see [`centred_pixel_error`]. The units are the caster pack's own
    // `max(pixel_error × wpt / max_scale, camera threshold)` with the linear
    // factor divided out, one per resident page.
    let errs = inf_vgeom::VgeomMesh::classic_lod_errors(&mesh);
    let mut units: Vec<f32> = Vec::new();
    for (light, page, rect) in resident_pages(&finest) {
        let vp = page_vp(&finest, light, page);
        let wpt = 2.0 / (vp.row(0).truncate().length() * rect.2 as f32);
        if wpt.is_finite() && wpt > 0.0 {
            units.push((wpt / max_scale).max(camera_unit));
        }
    }
    assert!(
        !units.is_empty(),
        "the reference run holds no resident page, so there is no ladder to \
         choose a tolerance against"
    );
    units.sort_by(|a, b| a.partial_cmp(b).expect("a page's texel is finite"));
    units.dedup();
    let (pixel_error, pick_margin) = centred_pixel_error(&errs, &units);
    // ANTI-FLAKE: the tolerance has to stand clear of every pick boundary, or the
    // arm is back on the knife edge the macOS red came off. 1.414 is what this
    // fixture affords; a chain that affords under 1.25 is one whose bands are
    // narrower than the page ladder's own factor of two, and the honest answer is
    // to say so rather than to measure at a boundary.
    assert!(
        pick_margin >= 1.25,
        "the nearest pick boundary of this chain is only {pick_margin:.3}× from \
         the best tolerance in the window ({pixel_error:.4} px) — a DAG whose \
         errors move by less than that changes which cut a page draws, which is \
         exactly the macOS red this arm was rewritten to close. Chain: {errs:?}"
    );
    // The shipped tolerance — the floor engages.
    let shipped = at(pixel_error);
    let (sh_stats, fi_stats) = (
        shipped.vsm_raster_stats().expect("stats"),
        finest.vsm_raster_stats().expect("stats"),
    );
    // ANTI-VACUITY (1): the two runs really drew different cuts, so what follows
    // is a measurement of the floor and not of the renderer's repeatability.
    assert!(
        sh_stats.vgeom_casters > 0 && fi_stats.vgeom_casters > 0,
        "one of the two runs packed no meshlet caster ({sh_stats:?} / {fi_stats:?})"
    );
    assert!(
        sh_stats.vgeom_level_sum * fi_stats.vgeom_casters
            > fi_stats.vgeom_level_sum * sh_stats.vgeom_casters,
        "the shipped run drew no coarser than the collapsed-tolerance one, so the \
         page-LOD floor never engaged and this arm measures nothing \
         ({sh_stats:?} / {fi_stats:?})"
    );

    let camera_world = camera_unit * pixel_error * max_scale;

    let (sh_atlas, fi_atlas) = (read_atlas(&gpu, &shipped), read_atlas(&gpu, &finest));
    let fi_rects: std::collections::BTreeMap<(u32, VsmPage), (u32, u32, u32)> =
        resident_pages(&finest)
            .into_iter()
            .map(|(l, p, r)| ((l, p), r))
            .collect();
    let mut compared = 0usize;
    let mut moved_texels = 0usize;
    let mut worst = (0.0f64, 0.0f64, 0u32); // metres, texels of its page, level
    for (light, page, rect) in resident_pages(&shipped) {
        let Some(&other) = fi_rects.get(&(light, page)) else {
            continue;
        };
        let vp = page_vp(&shipped, light, page);
        let ndc_per_m = vp.row(2).truncate().length();
        let wpt = 2.0 / (vp.row(0).truncate().length() * rect.2 as f32);
        if ndc_per_m <= 0.0 || !wpt.is_finite() || wpt <= 0.0 {
            continue;
        }
        for dy in 0..rect.2 {
            for dx in 0..rect.2 {
                let a = sh_atlas.2[((rect.1 + dy) * sh_atlas.0 + rect.0 + dx) as usize];
                let b = fi_atlas.2[((other.1 + dy) * fi_atlas.0 + other.0 + dx) as usize];
                if a == inf_render::VSM_DEPTH_CLEAR || b == inf_render::VSM_DEPTH_CLEAR {
                    continue;
                }
                compared += 1;
                if a == b {
                    continue;
                }
                moved_texels += 1;
                let m = f64::from((a - b).abs() / ndc_per_m);
                if m > worst.0 {
                    worst = (m, m / f64::from(wpt), page.level);
                }
                // **The bound, per page.** A page's tolerance is `pixel_error` of
                // its own texels or the camera's world tolerance, whichever is
                // coarser — the two terms the caster pack itself picks against.
                let tolerated = f64::from((pixel_error * wpt).max(camera_world)) + f64::from(wpt);
                assert!(
                    m <= tolerated,
                    "a level-{} page's shadow surface moved {m:.4} m when the \
                     page-LOD floor picked its cut, against the {tolerated:.4} m \
                     that page tolerates ({wpt:.4} m a texel at {pixel_error:.4} \
                     px, camera {camera_world:.4} m). The floor is drawing a cut \
                     coarser than the page it is written into, which is the one \
                     thing island wave I8c's VSM clause is not allowed to do",
                    page.level
                );
            }
        }
    }
    // ANTI-VACUITY (2): the atlases really were compared.
    assert!(
        compared > 1_000,
        "only {compared} texels were written by both runs — the comparison is \
         between two empty atlases"
    );
    println!(
        "THE PAGE-LOD FLOOR'S SHADOW PRICE: {moved_texels} of {compared} shared \
         texels moved ({:.2} %); the worst is {:.4} m — {:.2} texels of the \
         level-{} page it happened in — against a camera tolerance of \
         {camera_world:.4} m. Measured at {pixel_error:.4} px, the midpoint \
         between this chain's own pick boundaries ({pick_margin:.3}× clear of \
         the nearest). Shipped mean classic level {:.2} over {} caster records a \
         frame ({} meshlet draws), against {:.2} over {} at a collapsed \
         tolerance.",
        moved_texels as f64 / compared.max(1) as f64 * 100.0,
        worst.0,
        worst.1,
        worst.2,
        sh_stats.vgeom_level_sum as f64 / sh_stats.vgeom_casters.max(1) as f64,
        sh_stats.vgeom_casters as f64 / sh_stats.frames.max(1) as f64,
        sh_stats.draws_vgeom as f64 / sh_stats.frames.max(1) as f64,
        fi_stats.vgeom_level_sum as f64 / fi_stats.vgeom_casters.max(1) as f64,
        fi_stats.vgeom_casters as f64 / fi_stats.frames.max(1) as f64,
    );

    // **THE GROUP MULTIPLICATION, MEASURED** (the I8c audit). The wave's carried
    // item 4 says an asset "can now hold up to one group per clipmap level" and
    // that "on the island every bucket picks the same level, so it costs exactly
    // one; a scene with many assets close to the camera could multiply". This
    // fixture is one asset at 9 m through a six-level clipmap and it **does**
    // multiply — so the item is a measurement rather than a possibility, and the
    // ceiling it presses on (`VSM_MAX_GROUPS`, 1 024) is worth the counter beside
    // it. `dropped_groups` is the refusal and it is silent here, which is the
    // other half of the claim.
    assert!(
        sh_stats.vgeom_casters > sh_stats.frames,
        "one asset through a six-level clipmap packed one caster record a frame, \
         so the buckets all picked one level and carried item 4's multiplication \
         is not exercised here: {sh_stats:?}"
    );
    assert_eq!(
        (sh_stats.dropped_groups, sh_stats.dropped_casters),
        (0, 0),
        "the group ceiling refused something in a three-group fixture: {sh_stats:?}"
    );

    // **AND THE PARTITION, ON THE DRAWS THE FRAME ACTUALLY ISSUED** (the I8c
    // audit). The wave asserts the partition over `vgeom_caster_levels` — a pure
    // function — and says it is *"enforced on the device … and mirrored on the
    // CPU"*. Neither of those two is armed: each is masked by the other (drop the
    // CPU skip and the device cull leaves the extra draw empty; drop the device
    // test and the CPU's group mask never issues the draw), so a mutation to
    // either is invisible in the atlas.
    //
    // The counter this wave minted can see it. One instance in **three** groups
    // across eight pages issues **eight** meshlet draws — at most one a page,
    // which IS "every page draws the asset once, at the level that page can
    // show". Without the CPU mirror it is one a page PER GROUP, and the ceiling
    // below is the shape that survives a page the asset's sphere misses.
    assert!(
        sh_stats.draws_vgeom > 0 && sh_stats.draws_vgeom <= sh_stats.pages,
        "{} pages issued {} meshlet draws over {} caster records — a page is \
         drawing this instance once per group instead of once, so the bucket \
         masks are not partitioning the pages: {sh_stats:?}",
        sh_stats.pages,
        sh_stats.draws_vgeom,
        sh_stats.vgeom_casters,
    );
}

/// **A frame with pages and no casters CLEARS them** (P27.2 audit).
///
/// The pass used to return early when nothing packed, on the reasoning that "the
/// pass below clears the whole atlas, so this early return is only taken when there
/// is no pass to open at all". It is taken when there are pages and no casters —
/// and the editor's infinite grid is exactly that configuration: it writes camera
/// depth, so it marks pages, and it is not a caster. Delete every object in a level
/// and the atlas goes on holding their shadows.
#[test]
fn a_frame_with_no_caster_clears_the_pages_the_last_one_filled() {
    let Some(gpu) = gpu_or_skip("the VSM caster-less clear") else {
        return;
    };
    let mut filled = scene(0, 0.5, 1.0);
    filled.grid_enabled = true;
    filled.mark_dirty();
    // The same scene with its one object deleted. The grid still writes depth, so
    // pages are still marked and still resident.
    let mut emptied = filled.clone();
    emptied.instances.clear();
    emptied.mark_dirty();

    let set = settings_with(64);
    let v = view(5.0);
    let renderer = run_sequence(&gpu, &[(&filled, 6), (&emptied, 6)], &v, &set);
    let stats = renderer.vsm_raster_stats().expect("stats");
    let pages = resident_pages(&renderer);
    // ANTI-VACUITY, three ways: pages are resident, the first half really drew
    // casters, and the pass opened **again** once they were gone.
    //
    // "Again" rather than "on every frame" since P27.3: the pass opens for the
    // frames whose content stamps moved, and deleting every object moves them all
    // exactly once. `frames >= 2` is therefore the claim — the filled half filled
    // them, the emptied half cleared them — and `cached_pages > 0` proves the
    // steady state in between really was the cache and not a pass drawing nothing.
    assert!(!pages.is_empty(), "nothing was resident to clear");
    assert!(
        stats.casters > 0,
        "the filled half packed nothing: {stats:?}"
    );
    assert!(
        stats.frames >= 2,
        "the pass stopped opening once the casters were gone: {stats:?}"
    );
    assert!(
        stats.cached_pages > 0,
        "no frame was served by the cache, so this is P27.2's every-frame raster \
         rather than P27.3's: {stats:?}"
    );

    let atlas = read_atlas(&gpu, &renderer);
    let left: usize = pages
        .iter()
        .map(|(_, _, r)| written(&atlas, *r).len())
        .sum();
    assert_eq!(
        left, 0,
        "{left} texels still hold a deleted object's depth — a caster-less frame \
         left the atlas as it found it"
    );
}

/// **The group ceiling counts what it refuses too** (P27.2 audit).
///
/// `VSM_MAX_GROUPS` is the ceiling the first write-up did not have: the
/// per-(page, group) draw uniform is `pages x groups x 256 B`, and a skinned
/// instance is a group because its palette is a bind group. A thousand characters
/// is a thousand groups, and without a ceiling that buffer passes what a default
/// device will allocate.
///
/// So the fixture is a thousand and thirty characters, sharing one two-triangle
/// mesh: the ceiling has to refuse eleven of them, and it has to say so in both
/// counters.
#[test]
fn the_group_ceiling_counts_the_groups_it_refuses() {
    let Some(gpu) = gpu_or_skip("the VSM group ceiling") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-0.1, -0.1), v(0.1, -0.1), v(0.1, 0.1), v(-0.1, 0.1)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    // Enough instances that the rigid groups plus the skinned ones overrun the
    // ceiling by a countable margin.
    const OVER: u32 = 11;
    let want = inf_render::VSM_MAX_GROUPS + OVER - inf_render::VSM_RIGID_GROUPS;
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    for i in 0..want {
        s.skinned.push(inf_render::SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::new((i % 32) as f64 * 0.2 - 3.2, (i / 32) as f64 * 0.2, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 1_000 + i,
            mesh: 0,
            palette: inf_render::identity_palette(),
            shadow: inf_render::SkinnedShadow::BindSphere,
            vt: inf_render::VtTextureSet::NONE,
            sections: Vec::new(),
        });
    }
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let renderer = run(&gpu, &s, &view(6.0), &settings(), 3);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.frames > 0, "the pass never opened: {stats:?}");
    assert_eq!(
        stats.dropped_groups,
        u64::from(OVER) * stats.frames,
        "{} groups over the ceiling and {} counted: {stats:?}",
        u64::from(OVER) * stats.frames,
        stats.dropped_groups
    );
    // Every refused group took its caster with it, and the two counters agree.
    assert_eq!(stats.dropped_casters, stats.dropped_groups, "{stats:?}");
    assert_eq!(
        stats.skinned_casters,
        u64::from(want - OVER) * stats.frames,
        "{stats:?}"
    );
    // ANTI-VACUITY: the CASTER ceiling is not what refused them — a thousand
    // characters is nowhere near 16 384 — so this is the group ceiling's own arm.
    assert!(stats.casters < u64::from(inf_render::VSM_MAX_CASTERS) * stats.frames);
    // …and the pages still rasterized, so the ceiling refused a tail rather than
    // taking the frame down.
    assert!(
        stats.draws > 0 && renderer.vsm_raster_frames() > 0,
        "{stats:?}"
    );
}

/// **ISLAND WAVE NPC1e: the crowd's SHADOW LOD, which the proxy did not give.**
///
/// The arm above retires `VSM_MAX_GROUPS` and NPC1b's own carried item 4 says
/// what it left standing: 968 proxy boxes walking through Harbour City scattered
/// page invalidation over **168.6 pages a frame against the island's own 56.3**,
/// at 1 236 page draws against 328, and the NPC1b audit measured the *deferred*
/// pages doubling with them. One group is the right answer to a group ceiling and
/// no answer at all to how many pages a moving crowd dirties.
///
/// So `SkinnedShadow::None` — the tier's own answer past `CrowdTier::Near` — and
/// what this arm holds is that it is a **whole** drop: no caster, no group, no
/// palette slot, and the pages it would have dirtied are not dirtied.
///
/// **Three claims, and each falsifies a different wrong fix**: the counter
/// counts (a silent skip would read zero and look like an empty scene); the
/// proxy casters really fall away (a `None` treated as a `Proxy` keeps them);
/// and the pages a rastering frame touches really come down against the same
/// population asking for proxies, which is the number the ledger is about.
#[test]
fn a_crowd_past_the_shadow_lod_casts_nothing_and_dirties_fewer_pages() {
    let Some(gpu) = gpu_or_skip("the VSM crowd shadow LOD") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-0.1, -0.1), v(0.1, -0.1), v(0.1, 0.1), v(-0.1, 0.1)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    const CROWD: u32 = 240;
    let build = |shadow: inf_render::SkinnedShadow| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        for i in 0..CROWD {
            s.skinned.push(inf_render::SkinnedInstance {
                blend: 0,
                cutoff: 0.5,
                translation: glam::DVec3::new(
                    (i % 16) as f64 * 0.4 - 3.2,
                    (i / 16) as f64 * 0.4,
                    0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
                color: [1.0, 1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 1.0,
                emissive: [0.0; 3],
                id: 2_000 + i,
                mesh: 0,
                palette: inf_render::identity_palette(),
                shadow,
                vt: inf_render::VtTextureSet::NONE,
                sections: Vec::new(),
            });
        }
        s.instances.push(backdrop());
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        s.mark_dirty();
        s
    };

    let proxied = run(
        &gpu,
        &build(inf_render::SkinnedShadow::Proxy),
        &view(6.0),
        &settings(),
        3,
    );
    let proxied = proxied.vsm_raster_stats().expect("stats");
    let lodded = run(
        &gpu,
        &build(inf_render::SkinnedShadow::None),
        &view(6.0),
        &settings(),
        3,
    );
    let summary = lodded.vsm_summary().expect("a summary");
    let lodded = lodded.vsm_raster_stats().expect("stats");

    assert!(
        proxied.frames > 0 && lodded.frames > 0,
        "the pass never opened"
    );
    // 1. The counter counts, and it is the whole crowd.
    assert_eq!(
        lodded.shadow_lod_skipped,
        u64::from(CROWD) * lodded.frames,
        "the shadow LOD did not skip the crowd: {lodded:?}"
    );
    assert_eq!(
        proxied.shadow_lod_skipped, 0,
        "a crowd asking for proxies was counted as skipped: {proxied:?}"
    );
    assert!(
        summary.contains("shadow-LOD skipped"),
        "the counter is not in the summary: {summary:?}"
    );
    // 2. It is a WHOLE drop, not a cheaper caster.
    assert_eq!(lodded.proxy_casters, 0, "{lodded:?}");
    assert_eq!(lodded.skinned_casters, 0, "{lodded:?}");
    assert!(
        proxied.proxy_casters > 0,
        "the control cast no proxies, so there was nothing to save: {proxied:?}"
    );
    // 3. …and the **invalidation** work those casters did is not done. This is
    //    the quantity the churn is made of: `invalidation_touches` is the number
    //    of (caster, page) folds the stamp scatter performs, so a caster that is
    //    not packed cannot touch a page.
    //
    //    **The PAGE count is measured on the island and not here**, deliberately.
    //    On this fixture the whole crowd and the backdrop share eight pages, so
    //    the page count is the backdrop's and reads 8.0 either way — a bound on
    //    it would be a gate that passes for the wrong reason. What a toy fixture
    //    can hold is the work per caster; what a city holds is the page spread,
    //    and the ledger's 168.6-against-56.3 comes from the island instrument.
    let per = |n: u64, s: &inf_render::VsmRasterStats| n as f64 / s.frames.max(1) as f64;
    println!(
        "NPC1e / crowd shadow LOD, per rastering frame: {:.1} invalidation touches \
         and {:.1} pages with {CROWD} proxies; {:.1} and {:.1} with none",
        per(proxied.invalidation_touches, &proxied),
        per(proxied.pages, &proxied),
        per(lodded.invalidation_touches, &lodded),
        per(lodded.pages, &lodded),
    );
    assert!(
        lodded.invalidation_touches < proxied.invalidation_touches,
        "the shadow LOD folded as many caster-page touches as {CROWD} proxy \
         casters did ({} against {}), so nothing was skipped where it counts",
        lodded.invalidation_touches,
        proxied.invalidation_touches
    );
    // …and the frame still rastered a shadow map, so "fewer touches" is not
    // "nothing drawn".
    assert!(lodded.draws > 0 && lodded.pages > 0, "{lodded:?}");
}

/// **THE CROWD PROXY RETIRES THE GROUP CEILING** (wave NPC1b).
///
/// The arm above is this file's statement of wall 4: a skinned instance is one
/// geometry group, `VSM_MAX_GROUPS` is 1 024, and past it a crowd's shadows are
/// refused. This is the same fixture with the same population, asking for the
/// proxy instead — and the number that has to move is `dropped_groups`, from
/// eleven a frame to **zero**.
///
/// The two together are what makes the claim falsifiable. A proxy that quietly
/// dropped the casters instead of grouping them would pass the zero-drops half
/// and fail `proxy_casters`; a proxy that kept a group each would pass
/// `proxy_casters` and fail the drops.
#[test]
fn a_crowd_of_proxies_is_one_group_and_refuses_nothing() {
    let Some(gpu) = gpu_or_skip("the VSM crowd proxy") else {
        return;
    };
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-0.1, -0.1), v(0.1, -0.1), v(0.1, 0.1), v(-0.1, 0.1)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    const OVER: u32 = 11;
    let want = inf_render::VSM_MAX_GROUPS + OVER - inf_render::VSM_RIGID_GROUPS;
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    for i in 0..want {
        s.skinned.push(inf_render::SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::new((i % 32) as f64 * 0.2 - 3.2, (i / 32) as f64 * 0.2, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 1_000 + i,
            mesh: 0,
            palette: inf_render::identity_palette(),
            shadow: inf_render::SkinnedShadow::Proxy,
            vt: inf_render::VtTextureSet::NONE,
            sections: Vec::new(),
        });
    }
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let renderer = run(&gpu, &s, &view(6.0), &settings(), 3);
    let stats = renderer.vsm_raster_stats().expect("stats");
    assert!(stats.frames > 0, "the pass never opened: {stats:?}");
    assert_eq!(
        stats.dropped_groups, 0,
        "the crowd proxy still overran the group ceiling: {stats:?}"
    );
    assert_eq!(stats.dropped_casters, 0, "{stats:?}");
    // Every agent still casts — as a proxy, not as a silhouette of its own.
    assert_eq!(
        stats.proxy_casters,
        u64::from(want) * stats.frames,
        "{stats:?}"
    );
    assert_eq!(
        stats.skinned_casters, 0,
        "a proxy instance also packed a skinned caster: {stats:?}"
    );
    // …and the frame really rasterized, so "nothing refused" is not "nothing
    // drawn".
    assert!(
        stats.draws > 0 && renderer.vsm_raster_frames() > 0,
        "{stats:?}"
    );
    assert!(
        renderer.vsm_summary().unwrap().contains("crowd-proxy"),
        "the counter is not in the summary"
    );
}

/// **The exact posed bound is tighter than the margin, and still contains the
/// pose** (wave NPC1b) — the shipped half of
/// `the_palette_union_bound_is_tighter_than_the_shipped_pose_margin`, which
/// computed this bound in the test file and said it belonged to a later batch.
///
/// One fixture, two instances, differing only in `shadow`. The `Posed` one has to
/// hold a strictly smaller cull sphere than the `BindSphere` one — and both have
/// to contain the posed geometry, because a bound that culls a limb's shadow at
/// the moment the limb moves is the worst shape a defect can have.
#[test]
fn the_posed_bound_is_tighter_than_the_margin_and_still_contains_the_pose() {
    let Some(gpu) = gpu_or_skip("the exact posed skinned bound") else {
        return;
    };
    // A quad at ±1, skinned entirely to joint 0, displaced 0.6 m along x — the
    // same fixture the margin arm drives.
    let v = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    let palette = std::sync::Arc::new(vec![glam::Mat4::from_translation(glam::Vec3::new(
        0.6, 0.0, 0.0,
    ))]);
    let scene_with = |shadow: inf_render::SkinnedShadow| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        s.skinned.push(inf_render::SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::new(0.0, CUBE_XY.1 as f64, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 3,
            mesh: 0,
            palette: palette.clone(),
            shadow,
            vt: inf_render::VtTextureSet::NONE,
            sections: Vec::new(),
        });
        s.instances.push(backdrop());
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        s.mark_dirty();
        s
    };

    let sphere_of = |shadow| {
        let scene = scene_with(shadow);
        let renderer = run(&gpu, &scene, &view(6.0), &settings_with(64), 6);
        let sys = renderer.vsm().expect("the raster ran");
        let c = sys
            .raster_state()
            .last_casters()
            .iter()
            .find(|c| c.ids[0] >= inf_render::VSM_RIGID_GROUPS)
            .copied()
            .expect("no skinned caster was packed");
        (
            glam::Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]),
            c.sphere[3],
        )
    };

    let (bind_c, bind_r) = sphere_of(inf_render::SkinnedShadow::BindSphere);
    let (pose_c, pose_r) = sphere_of(inf_render::SkinnedShadow::Posed);

    // Both contain every posed vertex.
    for corner in [
        glam::Vec3::new(-1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 0.0),
        glam::Vec3::new(-1.0, 1.0, 0.0),
    ] {
        let p = palette[0].transform_point3(corner) + glam::Vec3::new(0.0, CUBE_XY.1, 0.0);
        assert!(
            (p - bind_c).length() <= bind_r + 1e-3,
            "the margin bound lost {p:?}"
        );
        assert!(
            (p - pose_c).length() <= pose_r + 1e-3,
            "the EXACT bound lost {p:?}, which is the one shape this must never do"
        );
    }
    // …and the exact one is strictly smaller, which is the point of it.
    assert!(
        pose_r < bind_r,
        "the posed bound ({pose_r}) is no tighter than the margin ({bind_r}), so \
         opting in buys nothing"
    );
    println!(
        "NPC1b posed bound: margin r {bind_r:.4} m, exact r {pose_r:.4} m — \
         {:.0} % of the radius, {:.0} % of the volume",
        100.0 * pose_r / bind_r,
        100.0 * (pose_r / bind_r).powi(3)
    );
}

// ── (i) P27.3: the page cache, and what invalidates it ───────────────────────

/// Render `steps`, then hand back the renderer **and** the raster stats at the
/// end of each step — the door every caching arm needs, because the claim is
/// always about what one step did rather than about a session total.
fn run_stepped(
    gpu: &GpuContext,
    steps: &[(&RenderScene, u64)],
    v: &RenderView,
    set: &VsmSettings,
) -> (inf_render::EngineRenderer, Vec<inf_render::VsmRasterStats>) {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut s = *renderer.settings();
    s.vsm = *set;
    renderer.set_settings(s);
    let mut marks = Vec::with_capacity(steps.len());
    for (scene, frames) in steps {
        for _ in 0..*frames {
            renderer.render(gpu, scene, v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        marks.push(renderer.vsm_raster_stats().expect("stats"));
    }
    (renderer, marks)
}

/// The whole atlas as raw bits — for the arms whose claim is **byte-identical**
/// rather than "close enough in metres".
fn atlas_bits(gpu: &GpuContext, renderer: &inf_render::EngineRenderer) -> Vec<u32> {
    let (_, _, d) = read_atlas(gpu, renderer);
    d.iter().map(|f| f.to_bits()).collect()
}

/// **THE CACHING CLAUSE** (P27.3, clause 1): a static scene stops rasterizing
/// pages, and the counter that says so is alive before it says it.
///
/// Built to fail the way the phase gate demands ("the arm FAILS if the counter is
/// dead"): a raster that never ran also re-rasterizes zero pages, so the first
/// half of this arm proves pages **were** rasterized and the second proves they
/// stopped. A no-op cache — one that reports every page cached without ever
/// filling one — fails the first half; a cache that never hits fails the second.
#[test]
fn a_static_scene_stops_rasterizing_pages_after_warm_up() {
    let Some(gpu) = gpu_or_skip("the VSM page cache") else {
        return;
    };
    let s = scene(0, 0.5, 1.0);
    let set = settings_with(64);
    let v = view(5.0);
    // Two windows over one unchanging scene: the first warms the cache, the
    // second is the steady state.
    let (renderer, marks) = run_stepped(&gpu, &[(&s, 6), (&s, 6)], &v, &set);
    let (warm, steady) = (marks[0], marks[1]);
    let resident = resident_pages(&renderer).len();

    // ANTI-VACUITY: the warm-up really rasterized, drew and packed.
    assert!(
        warm.pages > 0 && warm.draws > 0 && warm.casters > 0,
        "the warm-up rasterized nothing, so 'zero afterwards' says nothing: {warm:?}"
    );
    assert!(resident > 0, "no page was resident to cache");
    // **ZERO.** Not "fewer": the second six frames touched no page at all.
    assert_eq!(
        steady.pages - warm.pages,
        0,
        "a static scene re-rasterized {} pages after warm-up ({warm:?} -> {steady:?})",
        steady.pages - warm.pages
    );
    assert_eq!(steady.draws - warm.draws, 0, "{warm:?} -> {steady:?}");
    assert_eq!(
        steady.frames - warm.frames,
        0,
        "the pass opened on a frame with nothing to do"
    );
    // …and the cache is what did it, rather than the pages having gone away.
    assert!(
        steady.cached_pages - warm.cached_pages >= 6 * resident as u64,
        "the steady window served {} cached pages over 6 frames of {resident} \
         resident: {steady:?}",
        steady.cached_pages - warm.cached_pages
    );
    assert_eq!(
        renderer.vsm().expect("live").raster_state().cached_slots(),
        resident,
        "the cache vouches for a different number of slots than are resident"
    );
}

/// **A CACHED PAGE'S TEXELS ARE WHAT A FRESH RASTER PRODUCES** — bit for bit, off
/// the device (P27.3).
///
/// The claim a cache makes is not "we skipped a pass", it is "the texels are
/// right", and the two are only the same if nothing else wrote them. So: warm the
/// cache, copy the whole atlas off the GPU, **throw the cache away**, render one
/// more frame — which re-rasterizes every resident page from scratch — and compare
/// the two images word for word.
///
/// The anti-vacuity is the half that matters: the flush has to have caused a real
/// re-raster (`pages` moves by the resident count), or this compares an atlas
/// against itself.
#[test]
fn a_cached_pages_texels_are_what_a_fresh_raster_produces() {
    let Some(gpu) = gpu_or_skip("the VSM cache's honesty") else {
        return;
    };
    let s = scene(0, 0.5, 1.0);
    let set = settings_with(64);
    let v = view(5.0);
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);
    for _ in 0..8 {
        renderer.render(&gpu, &s, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let resident = resident_pages(&renderer).len() as u64;
    assert!(resident > 0, "nothing was resident");
    let before = atlas_bits(&gpu, &renderer);
    let cached = renderer.vsm_raster_stats().expect("stats");

    renderer.vsm_mut().expect("live").flush_page_cache();
    renderer.render(&gpu, &s, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after = atlas_bits(&gpu, &renderer);
    let fresh = renderer.vsm_raster_stats().expect("stats");

    // ANTI-VACUITY: the flushed frame really re-rasterized every resident page.
    assert_eq!(
        fresh.pages - cached.pages,
        resident,
        "the flush re-rasterized {} of {resident} resident pages, so this arm is \
         comparing an atlas with itself",
        fresh.pages - cached.pages
    );
    assert!(fresh.draws > cached.draws, "{cached:?} -> {fresh:?}");
    // …and there was something in the atlas to compare.
    let written = before
        .iter()
        .filter(|&&b| b != inf_render::VSM_DEPTH_CLEAR.to_bits())
        .count();
    assert!(
        written > 256,
        "only {written} texels of the atlas held depth — the comparison below is \
         about an empty image"
    );
    assert_eq!(before.len(), after.len(), "the atlas changed size");
    let differing = before.iter().zip(&after).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} atlas texels differ between the cached image and the \
         one a fresh raster produced — the cache is serving depth the raster \
         would not have written",
        before.len()
    );
}

/// **PAGE-EXACT INVALIDATION** (P27.3, clause 2): moving one object re-rasterizes
/// exactly the pages its light-space bounds touch, and **nothing else's texels
/// move**.
///
/// Two assertions, and the second is the one the scissor is load-bearing for.
///
/// 1. The count: the pages the frame rasterized are exactly the pages whose
///    frustum the mover's sphere entered **or** left — computed independently on
///    the CPU from the two positions, through the cull's own
///    `vsm_page_sees_sphere`. Strictly fewer than the resident set, or the clause
///    is met by "invalidate everything".
/// 2. The texels: every page the mover did *not* touch holds bit-identically what
///    it held before. With `set_scissor_rect` deleted from the per-page clear, the
///    clear covers the whole atlas and this half fails on every untouched page.
#[test]
fn a_mover_invalidates_exactly_the_pages_its_bounds_touch() {
    let Some(gpu) = gpu_or_skip("the VSM page-exact invalidation") else {
        return;
    };
    // A wide static backdrop so many pages are resident and full, plus one small
    // mover that covers a couple of them.
    const MOVER_XY: (f64, f64) = (-1.5, 1.0);
    const MOVER_SCALE: f32 = 0.4;
    const STEP: f64 = 0.6;
    let mut base = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    base.instances.push(backdrop());
    base.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(MOVER_XY.0, MOVER_XY.1, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(MOVER_SCALE),
        [1.0, 1.0, 1.0, 1.0],
        3,
    ));
    base.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    base.mark_dirty();
    // The same scene with the mover a short step to the right — far enough to
    // leave one page and enter another, nowhere near far enough to reach most.
    let mut moved = base.clone();
    moved.instances[1].translation.x += STEP;
    moved.mark_dirty();

    let set = settings_with(64);
    let v = view(5.0);
    let (mut renderer, marks) = run_stepped(&gpu, &[(&base, 8)], &v, &set);
    let warm = marks[0];
    let pages = resident_pages(&renderer);
    assert!(
        pages.len() >= 4,
        "only {} pages were resident — 'exactly the ones it touches' needs pages \
         it does NOT touch",
        pages.len()
    );
    let before = atlas_bits(&gpu, &renderer);

    // The independent answer: which resident pages either position's sphere is in.
    // The radius is the primitive's own bounding radius times the largest scale —
    // `pack_casters`'s derivation, restated here rather than read off the pack, so
    // the two are two statements.
    let r = inf_render::PrimMesh::Cube.bounding_radius() * MOVER_SCALE;
    let a = glam::Vec3::new(MOVER_XY.0 as f32, MOVER_XY.1 as f32, 0.0);
    let b = glam::Vec3::new((MOVER_XY.0 + STEP) as f32, MOVER_XY.1 as f32, 0.0);
    let want: std::collections::BTreeSet<usize> = pages
        .iter()
        .enumerate()
        .filter(|(_, (light, page, _))| {
            let vp = page_vp(&renderer, *light, *page);
            inf_render::vsm_page_sees_sphere(&vp, a, r)
                || inf_render::vsm_page_sees_sphere(&vp, b, r)
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        !want.is_empty() && want.len() < pages.len(),
        "the mover touches {} of {} resident pages — the fixture cannot tell \
         'exactly' from 'everything'",
        want.len(),
        pages.len()
    );

    // One frame of the moved scene.
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    renderer.render(&gpu, &moved, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after_stats = renderer.vsm_raster_stats().expect("stats");
    let touched = after_stats.pages - warm.pages;
    assert!(
        touched > 0,
        "moving an object rasterized nothing at all: {warm:?} -> {after_stats:?}"
    );
    assert_eq!(
        touched,
        want.len() as u64,
        "the mover's bounds touch {} pages and the frame rasterized {touched} \
         ({warm:?} -> {after_stats:?})",
        want.len()
    );

    // **The texels.** Every page outside `want` is bit-identical.
    let after = atlas_bits(&gpu, &renderer);
    let (w, _, _) = read_atlas(&gpu, &renderer);
    let mut untouched_texels = 0usize;
    let mut moved_outside = 0usize;
    for (i, (_, _, rect)) in pages.iter().enumerate() {
        if want.contains(&i) {
            continue;
        }
        for y in rect.1..rect.1 + rect.2 {
            for x in rect.0..rect.0 + rect.2 {
                let k = (y * w + x) as usize;
                untouched_texels += 1;
                moved_outside += usize::from(before[k] != after[k]);
            }
        }
    }
    assert!(
        untouched_texels > 4096,
        "only {untouched_texels} texels were outside the mover's pages"
    );
    assert_eq!(
        moved_outside, 0,
        "{moved_outside} of {untouched_texels} texels changed in pages the mover \
         never touched — a page-wide clear, or an invalidation that is not \
         page-exact"
    );
}

/// The straddling caster's cube side, in metres. Its bounding sphere's radius is
/// `√3/2 · side` = **6.93 m**, and that is the number that has to exceed
/// [`STRADDLE_OVERHANG`] for the sphere to still reach into the box after the
/// centre has left it.
const STRADDLE_SIDE: f32 = 8.0;
/// How far **past a face of the depth box** the straddler's centre sits. Strictly
/// between zero and the sphere's radius is the whole fixture: the centre is
/// outside the box and the sphere is not.
const STRADDLE_OVERHANG: f64 = 1.0;
/// How far it then moves, **along the light and toward it**. Small enough that it
/// is still straddling afterwards, large enough that the depth it writes moves by
/// three orders of magnitude more than an `f32` step.
const STRADDLE_STEP: f64 = 0.5;

/// **A caster whose CENTRE is outside the depth box still invalidates the pages
/// it writes** — the P27.3 audit's one carried *gap*, discharged at both faces.
///
/// # What was unarmed
///
/// `vsm_raster`'s `scatter_caster_stamps` folds a clipmap caster into a page's
/// content stamp only if the caster's sphere is inside the light's depth box:
/// `ndc.z ∈ [−rz, 1 + rz]`, where `rz` is the sphere's radius through the
/// matrix's own row 2. Those are the **cull's** two planes with the same radius
/// slack (`page_clip_planes` tests `z ≥ 0`, the far one, and `z ≤ w`, the near
/// one, each at `−radius`), which is what makes the stamp conservative in the
/// cull's direction: it may fold a caster into a page the cull then rejects, and
/// can never miss a page the cull keeps.
///
/// The P27.3 audit's mutation round tightened that envelope to the sphere's
/// **centre** — `ndc.z ∈ [0, 1]` — and **it survived the whole tree**, because
/// every caster in every fixture up to this one sat well inside a box hundreds of
/// metres deep. Its fail direction is the wrong one: a caster the cull draws but
/// the scatter does not stamp leaves the page's key unmoved, so the page is
/// served **from cache with the old caster in it**. A stale shadow, cached.
///
/// # The fixture, one per face
///
/// One cube 1 m past a face of the clipmap's own 384 m box, with a 6.93 m
/// bounding sphere — so its centre is outside and metres of its geometry are
/// inside, where they rasterize and **win** the reverse-Z compare against a
/// backdrop placed just inside the same face. Then it moves half a metre **along
/// the light**, and along the light only: its NDC *rectangle* does not move at
/// all, so the page set it folds into is identical at both positions and the only
/// thing that can invalidate the page is the depth envelope having admitted it in
/// the first place.
///
/// The ruling is asserted on the **atlas**, not on a counter: every texel that
/// moved, moved by exactly the metres the caster moved, converted by the box's
/// own range. A tightened envelope leaves them byte-identical — and a straddler
/// that never won its depth compare changes *nothing*, which fails the same
/// assertion, so "the caster is really in the page" needs no separate threshold.
#[test]
fn a_caster_straddling_the_depth_boxs_faces_still_invalidates_its_pages() {
    let Some(gpu) = gpu_or_skip("the VSM invalidation depth envelope") else {
        return;
    };
    let set = settings_with(64);
    let v = view(5.0);
    // The along-light box `build_projections` gives a clipmap: the coarsest
    // level's diameter, with the eye pulled back half of it — so with the light
    // along `+Z` (`fwd` is `−Z`) the NEAR face sits at `+range/2` of the layout
    // centre and the FAR face at `−range/2`. That centre is the origin for this
    // view: `clipmap_layout` snaps the along-light coordinate at the coarsest
    // stride, 48 m, and the eye is 5 m.
    let range = 2.0 * set.first_level_extent_m * (1u32 << (set.clipmap_levels - 1)) as f32;
    let half_box = f64::from(range) * 0.5;
    let radius = inf_render::PrimMesh::Cube.bounding_radius() * STRADDLE_SIDE;

    // (face name, the straddler's centre, the backdrop's own lit face). The
    // backdrop is just INSIDE the same face in both cases, so the straddler's own
    // surface is the one nearer the light and the page holds its depth.
    for (face, centre_z, backdrop_z) in [
        ("near", half_box + STRADDLE_OVERHANG, f64::from(BACKDROP_Z)),
        ("far", -half_box - STRADDLE_OVERHANG, -half_box + 2.0),
    ] {
        let slab = |z: f64| {
            inf_render::MeshInstance::lit(
                glam::DVec3::new(0.0, 0.0, z - f64::from(BACKDROP_T)),
                glam::Quat::IDENTITY,
                glam::Vec3::new(40.0, 40.0, 2.0 * BACKDROP_T),
                [1.0, 1.0, 1.0, 1.0],
                9,
            )
        };
        let straddler = |z: f64| {
            inf_render::MeshInstance::lit(
                glam::DVec3::new(0.0, 1.3, z),
                glam::Quat::IDENTITY,
                glam::Vec3::splat(STRADDLE_SIDE),
                [1.0, 1.0, 1.0, 1.0],
                3,
            )
        };
        let mut base = RenderScene {
            grid_enabled: false,
            ..Default::default()
        };
        base.instances.push(slab(backdrop_z));
        base.instances.push(straddler(centre_z));
        base.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        base.mark_dirty();
        let mut moved = base.clone();
        moved.instances[1] = straddler(centre_z + STRADDLE_STEP);
        moved.mark_dirty();

        let (mut renderer, marks) = run_stepped(&gpu, &[(&base, 8)], &v, &set);

        // ── the premises, before the ruling ──────────────────────────────────
        //
        // Read off the SHIPPED matrix, so the fixture cannot be straddling a box
        // the renderer does not have.
        let vp0 = {
            let sys = renderer.vsm().expect("a live vsm system");
            glam::Mat4::from_cols_array(&sys.projections()[0].view_proj)
        };
        let rz = radius * vp0.row(2).truncate().length();
        for z in [centre_z, centre_z + STRADDLE_STEP] {
            let c = vp0 * glam::Vec3::new(0.0, 1.3, z as f32).extend(1.0);
            let n = c.z / c.w;
            assert!(
                !(0.0..=1.0).contains(&n),
                "the {face}-face straddler's centre is INSIDE the depth box at \
                 z = {z} (ndc.z = {n}), so an envelope tightened to the centre \
                 admits it too and this arm proves nothing"
            );
            assert!(
                n >= -rz && n <= 1.0 + rz,
                "the {face}-face straddler's sphere left the box entirely at \
                 z = {z} (ndc.z = {n} against ±{rz}), so the SHIPPED envelope \
                 drops it as well"
            );
        }

        // …and the CULL keeps it, which is what makes a scatter that drops it a
        // MISS rather than the two agreeing.
        let pages = resident_pages(&renderer);
        assert!(
            pages.len() >= 2,
            "only {} pages were resident on the {face} face",
            pages.len()
        );
        let centre = glam::Vec3::new(0.0, 1.3, centre_z as f32);
        let seen = pages
            .iter()
            .filter(|(light, page, _)| {
                inf_render::vsm_page_sees_sphere(&page_vp(&renderer, *light, *page), centre, radius)
            })
            .count();
        assert!(
            seen > 0,
            "the per-page cull rejects the {face}-face straddler on every \
             resident page, so the scatter dropping it would be agreement rather \
             than a stale page"
        );

        // …and the cache is WARM: one more frame of the same scene rasterizes
        // nothing at all.
        let warm = marks[0];
        let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
        renderer.render(&gpu, &base, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let still = renderer.vsm_raster_stats().expect("stats");
        assert_eq!(
            still.pages, warm.pages,
            "a static scene is still rasterizing pages on the {face} face, so \
             'the move invalidated it' below is not a measurement ({warm:?} -> \
             {still:?})"
        );
        let before = atlas_bits(&gpu, &renderer);

        // ── THE RULING ───────────────────────────────────────────────────────
        renderer.render(&gpu, &moved, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let after_stats = renderer.vsm_raster_stats().expect("stats");
        assert!(
            after_stats.pages > still.pages,
            "the {face}-face straddler moved {STRADDLE_STEP} m along the light \
             and NOT ONE page re-rasterized — its stamp never reached them, so \
             every page it writes is served from cache with the old caster in it \
             ({still:?} -> {after_stats:?})"
        );
        let after = atlas_bits(&gpu, &renderer);

        // …and the WORLD moved with it. The straddler's NDC rectangle is
        // identical at both positions, so a texel that changed at all is a texel
        // the straddler owns — and it moved by `STEP / range` in NDC and by
        // nothing else. A straddler that lost its depth compare everywhere
        // changes NO texel and fails the count below.
        let want = (STRADDLE_STEP / f64::from(range)) as f32;
        let mut agreed = 0usize;
        for (b, a) in before.iter().zip(&after) {
            if b == a {
                continue;
            }
            let (b, a) = (f32::from_bits(*b), f32::from_bits(*a));
            assert!(
                (a - b - want).abs() < 1e-4,
                "a {face}-face straddler texel moved {} in NDC and the caster \
                 moved {STRADDLE_STEP} m of a {range} m box, which is {want}",
                a - b
            );
            agreed += 1;
        }
        assert!(
            agreed > 256,
            "only {agreed} atlas texels carried the {face}-face straddler's own \
             depth — it never won its reverse-Z compare against the backdrop, so \
             the page above re-rasterized to the same picture"
        );
    }
}

/// **The drain order is finest level first** (P27.3, clause 3's other half).
///
/// When something invalidates every level at once — the sun crossing a quantum, a
/// rebase, a cut — the frame's cap has to choose, and the choice is a function of
/// the page set rather than of arrival order: finest level first, then slot. The
/// finest pages are the ones a receiver reads at the most screen pixels, so a
/// budget that ran out would leave the coarse half stale rather than the near half.
#[test]
fn the_dirty_drain_takes_the_finest_levels_first() {
    let Some(gpu) = gpu_or_skip("the VSM drain order") else {
        return;
    };
    // A long floor receding from the camera, so the marked set spans several
    // clipmap levels: the level rule is one shadow texel per screen pixel, and a
    // pixel's world footprint grows with distance. The single cube every other arm
    // uses sits at one distance and marks **one** level, which is a fixture that
    // cannot see an ordering at all (measured — that is what the first draft of
    // this arm ran against).
    let mut s = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    s.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(0.0, -1.0, -30.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(60.0, 0.2, 80.0),
        [1.0, 1.0, 1.0, 1.0],
        4,
    ));
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Y,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();
    let set = settings_with(64);
    let v = view(5.0);
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);
    for _ in 0..8 {
        renderer.render(&gpu, &s, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    // A full invalidation, so every level is dirty in one frame.
    renderer.vsm_mut().expect("live").flush_page_cache();
    renderer.render(&gpu, &s, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let sys = renderer.vsm().expect("live");
    let drawn = sys.raster_state().last_pages();
    assert!(!drawn.is_empty(), "the flushed frame rasterized nothing");
    let levels: Vec<u32> = drawn.iter().map(|(_, p, _)| p.level).collect();
    assert!(
        levels.windows(2).all(|w| w[0] <= w[1]),
        "the drain order is not finest-first: {levels:?}"
    );
    // ANTI-VACUITY: more than one level was dirty, or "sorted" is vacuous.
    let distinct: std::collections::BTreeSet<u32> = levels.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "every dirty page was at level {distinct:?}, so the order says nothing"
    );
}

/// **Two routes to one residency make the same raster decisions** (the P26.2
/// "function of state, not history" law).
///
/// Two renderers reach the same scene by different paths — one straight through,
/// one with the cache thrown away half way — and the atlases they hold are
/// bit-identical. The second half of the law is the interesting one: their
/// **stamps** differ, because the residency's generation counter is process-global
/// and monotone, and that difference reaches no raster.
#[test]
fn two_routes_to_one_residency_make_the_same_raster_decisions() {
    let Some(gpu) = gpu_or_skip("the VSM route independence") else {
        return;
    };
    let s = scene(0, 0.5, 1.0);
    let set = settings_with(64);
    let v = view(5.0);
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let build = |flush_at: Option<u64>| {
        let mut r = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
        let mut rs = *r.settings();
        rs.vsm = set;
        r.set_settings(rs);
        for i in 0..10u64 {
            if Some(i) == flush_at {
                r.vsm_mut().expect("live").flush_page_cache();
            }
            r.render(&gpu, &s, &v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        r
    };
    let straight = build(None);
    let detoured = build(Some(6));

    let a = atlas_bits(&gpu, &straight);
    let b = atlas_bits(&gpu, &detoured);
    assert_eq!(a.len(), b.len());
    let written = a
        .iter()
        .filter(|&&x| x != inf_render::VSM_DEPTH_CLEAR.to_bits())
        .count();
    assert!(written > 256, "only {written} texels held depth");
    assert_eq!(
        a.iter().zip(&b).filter(|(x, y)| x != y).count(),
        0,
        "two routes to the same residency produced different atlases"
    );
    // The resident SET agrees too, page for page and slot for slot.
    assert_eq!(
        resident_pages(&straight),
        resident_pages(&detoured),
        "the two routes hold different pages in different slots"
    );
    // …and the stamps do NOT agree, which is the half that makes this a law about
    // state rather than a coincidence about two identical runs.
    let generation = |r: &inf_render::EngineRenderer| {
        r.vsm()
            .expect("live")
            .residency()
            .generation(VsmLightHandle(0))
            .expect("registered")
    };
    assert_ne!(
        generation(&straight),
        generation(&detoured),
        "the two residencies carry the same generation stamp, so 'stamps differ \
         and rasters do not' was not exercised"
    );
    // The route that flushed really did pay for it, or it was not a second route.
    assert!(
        detoured.vsm_raster_stats().expect("stats").pages
            > straight.vsm_raster_stats().expect("stats").pages,
        "the detour rasterized no more pages than the straight run"
    );
}

/// **A camera cut flushes the cache — and the stamps would have caught it anyway**
/// (P27.3's `is_camera_cut` clause, measured rather than asserted).
///
/// The bullet asks for the cut to invalidate through the `is_camera_cut`
/// precedent, and it does. What this arm records is the honest standing of that
/// trigger, on the `set_scissor_rect` precedent from P27.2: a cut moves the
/// clipmap's snapped centre by far more than a page, so **every page's own matrix
/// changes**, and the content stamps invalidate exactly the same set without the
/// trigger. It is defence in depth, its counter is alive, and this is where that
/// is written down rather than implied.
#[test]
fn a_camera_cut_flushes_the_cache_and_the_stamps_would_have_too() {
    let Some(gpu) = gpu_or_skip("the VSM camera cut") else {
        return;
    };
    let s = scene(0, 0.5, 1.0);
    let set = settings_with(64);
    let near = view(5.0);
    // 194 m away: past `is_camera_cut`'s own 50 m threshold, and far enough that
    // the clipmap re-centres.
    let far = view(199.0);
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);
    for _ in 0..6 {
        renderer.render(&gpu, &s, &near, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let before = renderer.vsm_raster_stats().expect("stats");
    assert_eq!(before.cut_flushes, 0, "a cut fired on a still camera");
    // The page **identities** as they stand, so the redundancy claim is measured
    // on the shipped door rather than argued.
    let near_identities = page_identities(&renderer);

    renderer.render(&gpu, &s, &far, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after = renderer.vsm_raster_stats().expect("stats");
    assert_eq!(
        after.cut_flushes, 1,
        "a 194 m jump did not read as a cut: {before:?} -> {after:?}"
    );
    assert!(
        after.pages > before.pages,
        "the cut flushed the cache and nothing was re-rasterized"
    );
    // **The redundancy, measured — and island wave VSM2 moved the thing that has
    // to be measured.** The cache key is no longer the page matrix: it is the
    // page's **world cell** plus the light's `content_key`, so a matrix
    // comparison here would be a gate that cannot see its own claim (the I7b
    // audit's shape). What makes the flush redundant now is that this cut also
    // slides the clipmap's *along-light* snap, which is in `content_key` — so
    // every surviving cell's stamp moves with it, and the lateral survivors a
    // scroll would keep are invalidated anyway.
    let far_identities = page_identities(&renderer);
    assert!(!near_identities.is_empty() && !far_identities.is_empty());
    let same = near_identities
        .iter()
        .filter(|m| far_identities.contains(m))
        .count();
    assert_eq!(
        same, 0,
        "{same} page identities survived a 194 m camera cut unchanged — the cut \
         trigger is NOT redundant with the content stamps, and this arm's ruling \
         has to be rewritten rather than the trigger removed"
    );
}

// ── (j) P27.3: a carved terrain hole casts nothing, and a carve invalidates ──

/// Row-packed per-sample hole bits, exactly as a projector writes them —
/// `ceil(res/32) * res` words, LSB first (`RenderTerrainTile::is_hole`'s layout).
fn pack_holes(res: u32, holed: impl Fn(u32, u32) -> bool) -> Vec<u32> {
    let words = res.div_ceil(32) as usize;
    let mut out = vec![0u32; words * res as usize];
    for j in 0..res {
        for i in 0..res {
            if holed(i, j) {
                out[j as usize * words + (i >> 5) as usize] |= 1u32 << (i & 31);
            }
        }
    }
    out
}

/// A planar tile with a square hole through the middle of its sample grid.
fn holed_tile(t: PlanarTile, lo: u32, hi: u32) -> inf_render::RenderTerrainTile {
    let mut built = t.build();
    built.holes = pack_holes(TILE_RES, |i, j| {
        (lo..hi).contains(&i) && (lo..hi).contains(&j)
    });
    built.version = 2;
    built
}

/// **A CARVED HOLE CASTS NOTHING** (P27.3) — the P27.2 remainder, closed.
///
/// P27.2's caster mesh read `heights` and ignored `holes`, so a tunnel mouth
/// shadowed as solid ground: the one defect in that pass a player standing under
/// it could see. The claim is made about **depth in metres**, not about a texel
/// count: every written texel of the atlas is reconstructed through its page's own
/// matrix, and none of them may land over the hole's world footprint — while the
/// same fixture without the mask covers it densely.
#[test]
fn a_carved_terrain_hole_casts_no_shadow() {
    let Some(gpu) = gpu_or_skip("the VSM terrain holes") else {
        return;
    };
    let tile = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        origin: glam::DVec3::new(-0.5 * TILE_SPAN, 3.5, -0.5 * TILE_SPAN),
        plane: (1.0, 2.5, -4.0),
    };
    // The middle third of the tile's samples, so the hole is comfortably wider
    // than the caster mesh's decimation stride and its world footprint is easy to
    // state.
    const LO: u32 = 11;
    const HI: u32 = 22;
    let hole_lo = LO as f64 / (TILE_RES - 1) as f64;
    let hole_hi = (HI - 1) as f64 / (TILE_RES - 1) as f64;
    let inside = |p: glam::Vec3| {
        let u = (p.x as f64 - tile.origin.x) / TILE_SPAN;
        let v = (p.z as f64 - tile.origin.z) / TILE_SPAN;
        // Half a decimation cell of slack on each side: the caster quad that
        // *contains* a holed sample is dropped whole, so the removed region is at
        // least the hole and at most one cell wider.
        let e = 1.0 / inf_render::VSM_TERRAIN_CASTER_CELLS as f64;
        (hole_lo + e..hole_hi - e).contains(&u) && (hole_lo + e..hole_hi - e).contains(&v)
    };

    let set = settings_with(64);
    let v = terrain_view();
    let solid = terrain_scene(&[tile]);
    let mut carved = solid.clone();
    carved.terrains[0].tiles[0] = holed_tile(tile, LO, HI);
    carved.mark_dirty();

    let count_over_hole = |renderer: &inf_render::EngineRenderer| {
        let atlas = read_atlas(&gpu, renderer);
        let (mut over, mut total) = (0usize, 0usize);
        for (light, page, rect) in resident_pages(renderer) {
            let inv = page_vp(renderer, light, page).inverse();
            for_each_written(&atlas, rect, |x, y, d| {
                let p = texel_world(inv, rect, x, y, d);
                total += 1;
                over += usize::from(inside(p));
            });
        }
        (over, total)
    };

    // The control first: the same tile with no mask covers the hole's footprint.
    let solid_r = run(&gpu, &solid, &v, &set, 6);
    let (solid_over, solid_total) = count_over_hole(&solid_r);
    assert!(
        solid_total > 512,
        "only {solid_total} ground texels were written at all"
    );
    assert!(
        solid_over > 64,
        "the uncarved control wrote only {solid_over} texels over the hole's \
         footprint, so 'zero after the carve' would say nothing"
    );

    let carved_r = run(&gpu, &carved, &v, &set, 6);
    assert!(
        carved_r.vsm_raster_stats().expect("stats").terrain_casters > 0,
        "the carved tile stopped being a caster altogether"
    );
    let (carved_over, carved_total) = count_over_hole(&carved_r);
    assert_eq!(
        carved_over, 0,
        "{carved_over} atlas texels put ground inside a carved hole — a tunnel \
         mouth is shadowing as solid ground"
    );
    // …and the rest of the tile still casts: the fix removes the hole, not the
    // tile.
    assert!(
        carved_total > solid_total / 2,
        "the carve removed {carved_total} of {solid_total} written texels — that \
         is the whole tile, not its hole"
    );
}

/// **A carve invalidates exactly the pages it touches** (P27.3, clause 2 meeting
/// the P21/P22 machinery).
///
/// Two tiles side by side; one is carved. The pages the frame re-rasterizes are
/// exactly the pages the carved tile's own bounds reach — computed independently
/// through the cull's `vsm_page_sees_sphere` against the caster sphere the pack
/// derives — and the other tile's pages hold bit-identically what they held.
///
/// The version bump is not decoration: a projector bumps a tile's version when it
/// writes holes into it, and `hole_words` in the caster cache key is what catches
/// the *first* carve, which changes the mask's length rather than its content.
#[test]
fn a_carve_invalidates_only_the_pages_the_carved_tile_touches() {
    let Some(gpu) = gpu_or_skip("the VSM carve invalidation") else {
        return;
    };
    let left = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        origin: glam::DVec3::new(-1.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (0.0, 0.0, 0.0),
    };
    let right = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((1, 0)),
        origin: glam::DVec3::new(0.5 * TILE_SPAN, 0.0, -0.5 * TILE_SPAN),
        plane: (0.0, 0.0, 0.0),
    };
    let set = settings_with(64);
    let v = terrain_view();
    let before_scene = terrain_scene(&[left, right]);
    let mut after_scene = before_scene.clone();
    after_scene.terrains[0].tiles[0] = holed_tile(left, 11, 22);
    after_scene.mark_dirty();

    let (mut renderer, marks) = run_stepped(&gpu, &[(&before_scene, 8)], &v, &set);
    let warm = marks[0];
    let pages = resident_pages(&renderer);
    assert!(
        pages.len() >= 4,
        "only {} pages resident — the fixture cannot tell 'exactly' from \
         'everything'",
        pages.len()
    );
    let before = atlas_bits(&gpu, &renderer);

    // The carved tile's caster sphere, as the pack derives it: the render-local
    // bound of its own decimated surface. Rebuilt here from the tile rather than
    // read off the pack, so the two are two statements.
    let lo = glam::Vec3::new(left.origin.x as f32, 0.0, left.origin.z as f32);
    let hi = lo + glam::Vec3::new(TILE_SPAN as f32, 0.0, TILE_SPAN as f32);
    let centre = 0.5 * (lo + hi);
    let radius = (hi - centre).length();
    let want: std::collections::BTreeSet<usize> = pages
        .iter()
        .enumerate()
        .filter(|(_, (light, page, _))| {
            inf_render::vsm_page_sees_sphere(&page_vp(&renderer, *light, *page), centre, radius)
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        !want.is_empty() && want.len() < pages.len(),
        "the carved tile reaches {} of {} pages",
        want.len(),
        pages.len()
    );

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    renderer.render(&gpu, &after_scene, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after_stats = renderer.vsm_raster_stats().expect("stats");
    let touched = after_stats.pages - warm.pages;
    assert!(
        touched > 0,
        "carving a tile re-rasterized nothing — the caster cache did not notice \
         the mask ({warm:?} -> {after_stats:?})"
    );
    assert_eq!(
        touched,
        want.len() as u64,
        "the carved tile reaches {} pages and the frame rasterized {touched}",
        want.len()
    );

    let after = atlas_bits(&gpu, &renderer);
    let (w, _, _) = read_atlas(&gpu, &renderer);
    let (mut checked, mut moved) = (0usize, 0usize);
    for (i, (_, _, rect)) in pages.iter().enumerate() {
        if want.contains(&i) {
            continue;
        }
        for y in rect.1..rect.1 + rect.2 {
            for x in rect.0..rect.0 + rect.2 {
                let k = (y * w + x) as usize;
                checked += 1;
                moved += usize::from(before[k] != after[k]);
            }
        }
    }
    assert!(
        checked > 4096,
        "only {checked} texels outside the carved tile"
    );
    assert_eq!(
        moved, 0,
        "{moved} of {checked} texels moved in pages the carved tile never reaches"
    );
}

/// **A character that animates invalidates its pages; one that stands still does
/// not** (P27.3).
///
/// The stamp of a skinned caster folds its whole joint palette, because the
/// palette **is** the geometry the vertex shader draws — a bind pose does not move
/// and a pose does. Both halves have to be asserted or the claim collapses into
/// one of two useless ones: a stamp that ignored the palette would leave a walking
/// character's shadow frozen at the pose the page was first filled at (measured —
/// dropping the palette from the fold survives every other arm in this file), and
/// a stamp that folded the frame index would re-rasterize a statue.
///
/// The pages are counted rather than the frames: a palette change invalidates only
/// the pages the character's cull sphere reaches, which is the same page-exact
/// claim the rigid mover makes, at the granularity a character has.
#[test]
fn an_animating_character_invalidates_its_pages_and_a_still_one_does_not() {
    let Some(gpu) = gpu_or_skip("the VSM skinned stamp") else {
        return;
    };
    let vert = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![
            vert(-0.5, -0.5),
            vert(0.5, -0.5),
            vert(0.5, 0.5),
            vert(-0.5, 0.5),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    let posed = |palette: glam::Mat4| {
        let mut s = RenderScene {
            grid_enabled: false,
            skinned_meshes: vec![mesh.clone()],
            ..Default::default()
        };
        s.skinned.push(inf_render::SkinnedInstance {
            blend: 0,
            cutoff: 0.5,
            translation: glam::DVec3::new(-1.4, 1.1, 0.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::splat(0.3),
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            id: 77,
            mesh: 0,
            palette: std::sync::Arc::new(vec![palette]),
            shadow: inf_render::SkinnedShadow::BindSphere,
            vt: inf_render::VtTextureSet::NONE,
            sections: Vec::new(),
        });
        s.instances.push(backdrop());
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::Z,
            cast_shadows: true,
            ..Default::default()
        });
        s.mark_dirty();
        s
    };
    // The same pose twice, then a different one. `mark_dirty` moves the scene
    // version in every case, so what tells the three steps apart is the stamp and
    // nothing else.
    let still = posed(glam::Mat4::IDENTITY);
    let still_again = posed(glam::Mat4::IDENTITY);
    let moved = posed(glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.4, 0.0)));

    let set = settings_with(64);
    let v = view(5.0);
    let (renderer, marks) = run_stepped(
        &gpu,
        &[(&still, 8), (&still_again, 4), (&moved, 2)],
        &v,
        &set,
    );
    let (warm, held, animated) = (marks[0], marks[1], marks[2]);
    assert!(
        warm.skinned_casters > 0,
        "no skinned caster was packed at all: {warm:?}"
    );
    assert!(warm.pages > 0, "the warm-up rasterized nothing: {warm:?}");
    // **Still**: four more frames of the identical pose touch nothing, even though
    // the scene version moved under every one of them.
    assert_eq!(
        held.pages - warm.pages,
        0,
        "a character holding one pose re-rasterized {} pages ({warm:?} -> {held:?})",
        held.pages - warm.pages
    );
    // **Animating**: the new pose invalidates, and only the pages the character
    // reaches.
    let touched = animated.pages - held.pages;
    assert!(
        touched > 0,
        "a character that changed pose re-rasterized nothing — its shadow is \
         frozen at the pose its pages were first filled at ({held:?} -> \
         {animated:?})"
    );
    let resident = resident_pages(&renderer).len() as u64;
    assert!(
        touched < resident,
        "a pose change re-rasterized {touched} of {resident} resident pages — that \
         is every page, not the character's"
    );
}

// ── (k) P27.3 AUDIT: the stamp inputs nothing was checking ──────────────────

/// **The skinned caster cache holds the `Arc` its pointer key names** (P27.3
/// audit) — the clause that makes an address an identity.
///
/// P27.3 moved the bind-pose cache from `scene.version` onto `Arc::as_ptr`, which
/// is `passes::skinned`'s rule — but that pass states the condition its key rests
/// on outright: *"the cache holds the `Arc` itself, which is what makes the
/// pointer a sound key: the allocation cannot be freed and reused under a live
/// entry"*. `VsmRaster` copied the key and not the clause, and the consequence is
/// the worst shape a cache defect has: a scene that drops mesh A and pushes a
/// different mesh B is free to land B on A's address, at which point `retain`
/// keeps A's buffers, the upload is skipped, the raster draws A's silhouette for
/// B — **and the page is not re-rasterized**, because the content stamp folds that
/// same address, so the wrong shadow is *cached*.
///
/// Asserted as ownership rather than as a reproduction, because reproducing an
/// ABA means winning a race with the allocator and an arm that only sometimes
/// reproduces is an arm that only sometimes fails. The strong count is isolated to
/// this pass by turning VSM on **under one renderer**: what the count gains across
/// that boundary is the page raster's own reference and nothing else's.
#[test]
fn the_skinned_caster_cache_holds_the_arc_its_pointer_key_names() {
    let Some(gpu) = gpu_or_skip("the VSM skinned cache's key") else {
        return;
    };
    let vert = |x: f32, y: f32| inf_render::SkinnedVertex {
        pos: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        joints: [0, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mesh = std::sync::Arc::new(inf_render::SkinnedMeshData {
        vertices: vec![
            vert(-0.5, -0.5),
            vert(0.5, -0.5),
            vert(0.5, 0.5),
            vert(-0.5, 0.5),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    });
    let mut s = RenderScene {
        grid_enabled: false,
        skinned_meshes: vec![mesh.clone()],
        ..Default::default()
    };
    s.skinned.push(inf_render::SkinnedInstance {
        blend: 0,
        cutoff: 0.5,
        translation: glam::DVec3::new(-1.4, 1.1, 0.0),
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::splat(0.3),
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0; 3],
        id: 77,
        mesh: 0,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        vt: inf_render::VtTextureSet::NONE,
        sections: Vec::new(),
    });
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    // Virtual shadows OFF first — the default. Everything else in the frame runs,
    // including `passes::skinned`'s own (correctly held) cache.
    let scene_only = std::sync::Arc::strong_count(&mesh);
    for _ in 0..8 {
        renderer.render(&gpu, &s, &view(5.0), &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let without_vsm = std::sync::Arc::strong_count(&mesh);
    // ANTI-VACUITY on the instrument: `passes::skinned` really does vouch for the
    // allocation, so a count that did not move below is a statement about the page
    // raster and not about `Arc::strong_count` being inert here.
    assert!(
        without_vsm > scene_only,
        "the lit skinned pass did not take a reference ({scene_only} -> \
         {without_vsm}) — this arm's instrument is dead"
    );

    let mut rs = *renderer.settings();
    rs.vsm = settings_with(64);
    renderer.set_settings(rs);
    for _ in 0..8 {
        renderer.render(&gpu, &s, &view(5.0), &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    assert!(
        renderer.vsm_raster_stats().expect("stats").skinned_casters > 0,
        "no skinned caster was packed, so the cache under test was never filled"
    );
    let with_vsm = std::sync::Arc::strong_count(&mesh);
    assert!(
        with_vsm > without_vsm,
        "the page raster cached this mesh's bind pose under its ADDRESS \
         ({without_vsm} -> {with_vsm} references) without holding the `Arc` — the \
         allocation can be freed and handed to a different mesh under the live \
         entry, and the stamp folds the same address, so the wrong silhouette \
         would be *cached* rather than merely drawn once"
    );
}

/// **A vgeom caster that crosses a LOD threshold invalidates its pages** (P27.3
/// audit).
///
/// The level a `.inf_vmesh` casts at is a **camera** decision — `pick_classic_level`
/// against the camera's own `lod_threshold`, the P27.2 deviation memo's ruling — so
/// it is the one caster input that moves while the caster does not. If it were not
/// in the content stamp, "a cached page equals a fresh raster" would be false the
/// moment the viewer walked far enough to coarsen the cut, and the byte-compare arm
/// could not see it: that arm never moves the camera between caching and comparing.
/// Deleting the level from the fold survived the whole file before this.
///
/// The tolerance is moved rather than the camera, because a camera step also moves
/// every page matrix and would prove nothing about the caster. The control is the
/// same settings change over a scene with **no** virtualized geometry: if changing
/// `pixel_error` re-rasterized pages by some other route, it would do it there too.
#[test]
fn a_vgeom_caster_that_crosses_a_lod_threshold_invalidates_its_pages() {
    let Some(gpu) = gpu_or_skip("the VSM vgeom LOD stamp") else {
        return;
    };
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::dense_grid_mesh(24));
    let mut s = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![
            inf_render::VgeomAsset::from_mesh(0x5150, &mesh).expect("index the vmesh")
        ],
        ..Default::default()
    };
    s.vgeom_instances.push(inf_render::VgeomInstance::lit(
        0x5150,
        glam::DVec3::ZERO,
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(3.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();
    // The control: the same frame with the vmesh taken out.
    let mut control = s.clone();
    control.vgeom_instances.clear();
    control.vgeom_assets.clear();
    control.mark_dirty();

    let set = settings_with(64);
    let v = view(9.0);
    // Warm at a tenth of a pixel of tolerated error (the finest level), then move
    // the tolerance past every level in the chain and take ONE more frame.
    let coarsen = |scene: &RenderScene| {
        let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
        let mut r = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
        let mut rs = *r.settings();
        rs.vsm = set;
        rs.vgeom.pixel_error = 0.1;
        r.set_settings(rs);
        for _ in 0..8 {
            r.render(&gpu, scene, &v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let warm = r.vsm_raster_stats().expect("stats");
        rs.vgeom.pixel_error = 400.0;
        r.set_settings(rs);
        r.render(&gpu, scene, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let after = r.vsm_raster_stats().expect("stats");
        (warm, after, resident_pages(&r).len() as u64)
    };

    let (warm, after, resident) = coarsen(&s);
    // ANTI-VACUITY: the level really moved, so this is an arm about a LOD change
    // and not about a settings write.
    // **The tight tolerance does not saturate.** The equality this used to be —
    // one caster at level 0 — stopped being the shape of the answer at island
    // wave I8c: a meshlet caster is packed once per distinct level its page
    // BUCKETS ask for, so what is left of the claim is that the finest tolerance
    // is nowhere near the coarsest level, which is what makes the coarsening
    // below a threshold crossing rather than a no-op.
    let coarsest = (inf_vgeom::VgeomMesh::classic_lods(&mesh).len() - 1) as u64;
    assert!(
        warm.vgeom_casters > 0 && warm.vgeom_level_sum < warm.vgeom_casters * coarsest,
        "every caster at a tenth of a pixel of error already drew the coarsest \
         level {coarsest}: {warm:?}"
    );
    assert!(
        after.vgeom_level_sum > warm.vgeom_level_sum,
        "400 px of tolerated error drew the same level as 0.1 px — the fixture \
         never crossed a threshold ({warm:?} -> {after:?})"
    );
    let touched = after.pages - warm.pages;
    assert!(
        touched > 0,
        "a caster that changed geometry re-rasterized nothing — its pages hold the \
         previous cut's silhouette and the cache is serving it ({warm:?} -> \
         {after:?})"
    );
    assert!(
        touched <= resident,
        "{touched} pages for {resident} resident: the invalidation is not \
         page-exact"
    );

    // **The control.** Nothing in the frame but the setting changed, and with no
    // virtualized geometry in the scene that setting reaches no caster.
    let (c_warm, c_after, _) = coarsen(&control);
    assert_eq!(
        c_after.pages - c_warm.pages,
        0,
        "changing `pixel_error` re-rasterized {} pages in a scene with no vgeom at \
         all, so the assertion above is not about the LOD ({c_warm:?} -> \
         {c_after:?})",
        c_after.pages - c_warm.pages
    );
}

/// **A cut-out caster's alpha-test window is in its page's stamp** (P27.3 audit).
///
/// A masked caster's page depth is decided by `fs_masked`'s `alpha < cutoff`, so
/// the cutoff and the base-colour alpha are raster inputs exactly as the model
/// matrix is — they decide whether the caster writes depth at all. They ride in
/// `VsmCasterRaw::mat`, and deleting `mat` from the caster stamp survived every
/// other arm in this file: a material edit that opened or closed a cutout would
/// leave the shadow it used to cast frozen in the atlas.
///
/// Asserted on the **texels**, not on the counter: the cutout closes, so the
/// caster's own depth has to leave the atlas.
#[test]
fn a_cutout_casters_alpha_test_window_is_in_its_pages_stamp() {
    let Some(gpu) = gpu_or_skip("the VSM alpha-test stamp") else {
        return;
    };
    // Masked, and passing its own test: alpha 1.0 against a cutoff of 0.5, so the
    // cube casts. The backdrop keeps the pages marked either way.
    let mut casting = scene(1, 0.5, 1.0);
    casting.instances.push(backdrop());
    casting.mark_dirty();
    // The same caster with the cutoff raised past its alpha — every fragment
    // discards, so it stops casting. Nothing else about it moves: same transform,
    // same bounds, same geometry, same scene shape.
    let mut discarding = casting.clone();
    discarding.instances[0].cutoff = 1.5;
    discarding.mark_dirty();

    let set = settings_with(64);
    let v = view(5.0);
    let (mut renderer, marks) = run_stepped(&gpu, &[(&casting, 8)], &v, &set);
    let warm = marks[0];
    assert!(
        warm.masked_frames > 0,
        "the fixture packed no masked caster"
    );
    let pages = resident_pages(&renderer);
    let before = atlas_bits(&gpu, &renderer);

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    renderer.render(&gpu, &discarding, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after_stats = renderer.vsm_raster_stats().expect("stats");
    assert!(
        after_stats.pages > warm.pages,
        "raising a cutout's cutoff past its own alpha re-rasterized nothing — the \
         alpha-test window is not in the content stamp ({warm:?} -> \
         {after_stats:?})"
    );

    // …and the texels moved with it: the cube's depth left the atlas.
    let after = atlas_bits(&gpu, &renderer);
    let (w, _, _) = read_atlas(&gpu, &renderer);
    let mut changed = 0usize;
    for (_, _, rect) in &pages {
        for y in rect.1..rect.1 + rect.2 {
            for x in rect.0..rect.0 + rect.2 {
                let k = (y * w + x) as usize;
                changed += usize::from(before[k] != after[k]);
            }
        }
    }
    assert!(
        changed > 0,
        "the pages re-rasterized and not one texel moved — the discard did not \
         reach the depth this arm is about"
    );
}

/// **A terrain tile whose version moves invalidates its pages** (P27.3 audit).
///
/// `version` is the one thing in the terrain caster's fold that a sculpt is
/// guaranteed to move, and deleting it survived the whole file — because the two
/// carve arms move the hole mask and the triangle count as well, and *those* are
/// what they were measuring. So the fixture is built to move nothing else: the
/// tile's plane is **transposed** rather than tilted, which changes every interior
/// height while leaving the height *range* — and therefore the caster's bounding
/// sphere, which is the other geometric term in the fold — bit-identical.
#[test]
fn a_terrain_tile_whose_version_moves_invalidates_its_pages() {
    let Some(gpu) = gpu_or_skip("the VSM terrain version stamp") else {
        return;
    };
    let before_tile = PlanarTile {
        key: inf_render::TerrainTileKey::lod0((0, 0)),
        origin: glam::DVec3::new(-0.5 * TILE_SPAN, 3.5, -0.5 * TILE_SPAN),
        plane: (1.0, 2.5, -4.0),
    };
    // The same plane with its two gradients swapped: the surface is a different
    // surface at every interior sample and its min/max are the same two numbers.
    let after_tile = PlanarTile {
        plane: (1.0, -4.0, 2.5),
        ..before_tile
    };
    let set = settings_with(64);
    let v = terrain_view();
    let before_scene = terrain_scene(&[before_tile]);
    let mut after_scene = terrain_scene(&[after_tile]);
    // The sculpt's stamp. Without it nothing downstream would rebuild the mesh at
    // all, which is a different (and already-armed) claim.
    after_scene.terrains[0].tiles[0].version = 2;
    after_scene.mark_dirty();

    // ANTI-VACUITY, computed from the fixture: the bound the fold carries beside
    // the version did NOT move, so what is left to notice the edit is the version.
    let (a, b) = (
        before_scene.terrains[0].tiles[0].height_bounds,
        after_scene.terrains[0].tiles[0].height_bounds,
    );
    assert_eq!(a, b, "the transposed plane moved the tile's height bounds");
    assert_ne!(
        before_scene.terrains[0].tiles[0].heights, after_scene.terrains[0].tiles[0].heights,
        "the two tiles have identical heights, so there is no edit to notice"
    );

    let (mut renderer, marks) = run_stepped(&gpu, &[(&before_scene, 8)], &v, &set);
    let warm = marks[0];
    assert!(warm.terrain_casters > 0, "no terrain caster: {warm:?}");
    let pages = resident_pages(&renderer);
    let resident = pages.len() as u64;
    assert!(resident > 0, "nothing was resident");
    let before = atlas_bits(&gpu, &renderer);

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    renderer.render(&gpu, &after_scene, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after_stats = renderer.vsm_raster_stats().expect("stats");
    let touched = after_stats.pages - warm.pages;
    assert!(
        touched > 0,
        "a sculpt that moved the tile's version re-rasterized nothing — the pages \
         hold the previous surface ({warm:?} -> {after_stats:?})"
    );
    assert!(
        touched <= resident,
        "{touched} of {resident} is not page-exact"
    );

    // …and the depth in the atlas really is the new surface.
    let after = atlas_bits(&gpu, &renderer);
    let (w, _, _) = read_atlas(&gpu, &renderer);
    let mut moved = 0usize;
    for (_, _, rect) in &pages {
        for y in rect.1..rect.1 + rect.2 {
            for x in rect.0..rect.0 + rect.2 {
                let k = (y * w + x) as usize;
                moved += usize::from(before[k] != after[k]);
            }
        }
    }
    assert!(
        moved > 0,
        "the pages re-rasterized and the atlas is unchanged — the caster mesh was \
         not rebuilt from the new heights"
    );
}

/// **A perspective light's pages are invalidated by the mover they see** (P27.3
/// audit).
///
/// A clipmap's pages are a lattice, so the invalidation scatter derives their
/// rectangles arithmetically. A spot's are not, so its pages take the per-page
/// sphere test directly — a completely separate branch of `scatter_caster_stamps`,
/// and one that **no arm reached**: deleting it (folding nothing into any
/// perspective page, ever) survived the whole file, which would leave every spot
/// and point light's shadow frozen at whatever the frame that first filled its
/// pages saw.
#[test]
fn a_spot_lights_pages_are_invalidated_by_the_mover_they_see() {
    let Some(gpu) = gpu_or_skip("the VSM perspective invalidation") else {
        return;
    };
    let mut base = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    base.instances.push(backdrop());
    base.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(-0.6, 0.4, 0.0),
        glam::Quat::IDENTITY,
        glam::Vec3::splat(0.5),
        [1.0, 1.0, 1.0, 1.0],
        3,
    ));
    // A spot behind the camera shining down -Z at the backdrop. `direction` points
    // TOWARD the light, so a beam travelling along -Z is `+Z`.
    base.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Spot,
        position: glam::DVec3::new(0.0, 0.0, 7.0),
        direction: glam::Vec3::Z,
        cast_shadows: true,
        ..Default::default()
    });
    base.mark_dirty();
    let mut moved = base.clone();
    moved.instances[1].translation.x += 0.5;
    moved.mark_dirty();

    let set = settings_with(64);
    let v = view(5.0);
    let (mut renderer, marks) = run_stepped(&gpu, &[(&base, 8)], &v, &set);
    let warm = marks[0];
    // The fixture really is a perspective tree, or the arm is a second clipmap arm.
    let sys = renderer.vsm().expect("live");
    assert_eq!(
        sys.trees().len(),
        1,
        "the fixture registered {} trees",
        sys.trees().len()
    );
    assert_eq!(
        sys.trees()[0].kind,
        inf_render::VsmTreeKind::Quadtree,
        "the spot did not take a quadtree"
    );
    let resident = resident_pages(&renderer).len() as u64;
    assert!(resident > 0, "the spot marked no page at all");
    assert!(warm.pages > 0, "the warm-up rasterized nothing: {warm:?}");

    // The steady state first: the identical scene touches nothing, so what the
    // move does below is the move and not a light that never caches.
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    renderer.render(&gpu, &base, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let held = renderer.vsm_raster_stats().expect("stats");
    assert_eq!(
        held.pages - warm.pages,
        0,
        "a spot light's pages do not cache at all ({warm:?} -> {held:?})"
    );

    renderer.render(&gpu, &moved, &v, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after = renderer.vsm_raster_stats().expect("stats");
    let touched = after.pages - held.pages;
    assert!(
        touched > 0,
        "a mover under a SPOT light re-rasterized nothing — the perspective half \
         of the invalidation scatter folds no caster into any page, and every \
         spot and point shadow is frozen at the frame that filled it ({held:?} -> \
         {after:?})"
    );
    assert!(
        touched <= resident,
        "{touched} of {resident} pages — the perspective branch invalidates \
         everything rather than what the mover reaches"
    );
}

/// **A clipmap grid shift re-labels a page** (P27.3 audit) — the aliasing ruling,
/// measured, and it does not say what the P27.3 ledger said it said.
///
/// # What island wave VSM2 changed about this arm's conclusion
///
/// The paragraphs below are P27.3's, kept because the *measurement* is unchanged
/// and it is the evidence the wave acted on: a clipmap level's window slides, a
/// world cell moves from page `(x, y)` to `(x − 1, y)`, and the matrix it wants is
/// the one the atlas already holds. What has changed is the sentence at the end.
/// **The key does not pay for the label any more**: residency carries the slot
/// with the world cell (`inf_vsm::VsmResidency::set_clip_origins`) and the cache
/// is keyed on that cell, so the `free_hits` this arm counts are now taken rather
/// than refused. The counter-assertion is
/// `a_camera_translation_re_labels_no_page_and_re_rasters_only_what_enters`; what
/// stays here is the phenomenon, and the soundness condition below — every
/// collision inside one light and one level — which the world-cell key now states
/// explicitly instead of relying on.
///
/// The P27.3 ledger justifies all three members of `(light, page, stamp)` with:
/// *"a slot that is evicted, refilled by another page and re-admitted to the first
/// would otherwise read as a hit while holding the second page's depth"*. Measured,
/// that is backwards. Dropping `page` from the key, dropping `light`, and dropping
/// `level`+`light` from the geometric fold each survive the whole file — and this
/// arm says why: the stamp's geometric half is the page's **own matrix**, which is
/// its world footprint, so a page that shares a stamp shares the depth it wants.
/// A hit under a stamp-only key is a *correct* hit.
///
/// What the walk below finds is the case that makes those two members non-trivial
/// in the other direction. When a clipmap level's grid shifts by one page, the
/// world cell that was page `(x, y)` becomes page `(x−1, y)` — the same footprint,
/// a different address, a **bit-identical matrix**. So the extra members can only
/// ever turn a correct hit into a miss, and what they cost is a re-raster of depth
/// the atlas already holds. That is the *"there is no clipmap scroll"* remainder,
/// wearing the cache key instead of the residency, and this arm puts a number on
/// the fraction of it a stamp-only key would recover for free.
///
/// Both halves are asserted, and the first is what keeps the members honest: every
/// collision is inside **one light and one level**. A stamp shared across two
/// lights or two levels would mean the fold's `light` and `level` terms were doing
/// no work and a stamp-only key was unsound — the day that happens this arm fails
/// and the ruling is rewritten rather than the members quietly removed.
#[test]
fn a_clipmap_grid_shift_re_labels_a_page_and_the_cache_key_pays_for_it() {
    let Some(gpu) = gpu_or_skip("the VSM cache key's aliasing") else {
        return;
    };
    // Two lights, so "the same page of a different light" is in the sample, and a
    // camera that walks far enough for coarse levels to shift their grids.
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(backdrop());
    s.lights.push(inf_render::RenderLight {
        kind: inf_render::LightKind::Directional,
        direction: glam::Vec3::new(0.4, 0.8, 0.45).normalize(),
        cast_shadows: true,
        ..Default::default()
    });
    s.mark_dirty();

    let set = settings_with(16);
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);

    // matrix bits -> every (light, page) that ever presented it.
    let mut seen: std::collections::BTreeMap<
        [u32; 16],
        std::collections::BTreeSet<(u32, VsmPage)>,
    > = std::collections::BTreeMap::new();
    // slot -> (occupant, its matrix) last frame, so a refill can be classified.
    let mut occupant: std::collections::BTreeMap<u32, ((u32, VsmPage), [u32; 16])> =
        std::collections::BTreeMap::new();
    let (mut refills, mut free_hits) = (0usize, 0usize);
    for step in 0..24 {
        let v = view(5.0 + step as f64 * 1.5);
        renderer.render(&gpu, &s, &v, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let mut now = std::collections::BTreeMap::new();
        for (light, page, rect) in resident_pages(&renderer) {
            let m = page_vp(&renderer, light, page)
                .to_cols_array()
                .map(f32::to_bits);
            seen.entry(m).or_default().insert((light, page));
            let slot = rect.0 * 1_000_000 + rect.1;
            if let Some(&(prev, prev_m)) = occupant.get(&slot) {
                if prev != (light, page) {
                    refills += 1;
                    // The slot changed occupant and the new occupant wants exactly
                    // the depth the old one left: a hit a stamp-only key would take
                    // and this key refuses.
                    free_hits += usize::from(prev_m == m);
                }
            }
            now.insert(slot, ((light, page), m));
        }
        occupant = now;
    }

    // **The collisions are re-labellings**, never a stamp shared across lights or
    // levels — which is what makes the geometric fold's `light` and `level` terms
    // load-bearing and a stamp-only key sound.
    let mut relabelled = 0usize;
    for (m, addrs) in &seen {
        if addrs.len() < 2 {
            continue;
        }
        relabelled += addrs.len() - 1;
        let first = addrs.iter().next().expect("non-empty");
        for a in addrs {
            assert_eq!(
                (a.0, a.1.face, a.1.level),
                (first.0, first.1.face, first.1.level),
                "pages {addrs:?} share the page matrix {m:?} across two lights or \
                 two levels — the content stamp cannot tell them apart, so the \
                 cache key's extra members are load-bearing for CORRECTNESS and \
                 this ruling has to be rewritten"
            );
        }
    }
    // ANTI-VACUITY on the walk: it saw many pages, refilled slots, and really did
    // re-label some of them.
    assert!(
        seen.len() > 24,
        "only {} distinct page matrices over the walk",
        seen.len()
    );
    assert!(
        refills > 0,
        "no slot ever changed occupant, so the aliasing case this arm rules on was \
         never reached"
    );
    assert!(
        relabelled > 0,
        "no clipmap level ever re-labelled a page over a 34 m camera walk — the \
         cost this arm measures is not reachable and the ledger's aliasing \
         sentence has no case at all"
    );
    // …and the number: refills whose depth the atlas already held, which P27.3's
    // label-keyed cache re-rasterized and island wave VSM2's world-cell key
    // serves.
    eprintln!(
        "VSM CACHE KEY: {refills} slot refills over the walk, {free_hits} of them \
         wanting depth the slot already held ({relabelled} re-labelled addresses \
         over {} distinct matrices)",
        seen.len()
    );
}

// ── ISLAND WAVE VSM2: THE CLIPMAP SCROLL ────────────────────────────────────

/// The eye `x` metres along the light's **`right`** and `eye_z` back along it.
///
/// `right` is `+X` for this file's `+Z` sun (`light_basis` composes it as
/// `fwd × up` with `fwd = −Z`), so walking `x` scrolls the clipmap windows and
/// leaves the along-light snap — the one term of the centre that IS in the
/// content key — exactly where it was.
fn view_beside(x: f64, eye_z: f64) -> RenderView {
    RenderView {
        eye_world: glam::DVec3::new(x, 0.0, eye_z),
        ..view(eye_z)
    }
}

/// **A fixture whose atlas actually fills**, and the reason it takes this shape.
///
/// A page is 128² texels and the level rule puts about one of them under each
/// screen pixel, so one frame of one sun needs about `pixels / 128²` pages — **36**
/// at 1 024 × 576, and **9** at this file's usual 256 × 144. That is a property of
/// the clipmap and not of the fixture: a single-sun scene at a test resolution
/// cannot make a whole atlas dirty at once, so the raster's own
/// `VSM_MAX_RASTER_PAGES` ceiling, and therefore everything about deferral, would
/// be **unreachable** and an arm about the drain would be asserting on a raster
/// that never defers. (Measured, over seven configurations: 6 to 123 resident
/// pages, whatever the grid, the extent or the walk.)
///
/// Four suns over one depth buffer mark four ladders — legal, cheap, and the only
/// thing in reach that fills a thousand slots: **343 resident by frame 40, 488 by
/// frame 80.**
const LADDER_W: u32 = 1024;
const LADDER_H: u32 = 576;
/// One level-0 page a step at the settings below (`2 × 6 / 64`), so every frame
/// scrolls the finest window.
const LADDER_STEP_M: f64 = 0.1875;

fn big_ladder() -> VsmSettings {
    VsmSettings {
        clipmap_pages_per_side: 64,
        clipmap_levels: 8,
        first_level_extent_m: 6.0,
        ..settings_with(2048)
    }
}

/// This file's cube and backdrop plus a long receding floor — a marked set that
/// spans levels rather than one distance — under `suns` directional lights.
fn ladder_scene(suns: usize) -> RenderScene {
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(backdrop());
    s.instances.push(inf_render::MeshInstance::lit(
        glam::DVec3::new(0.0, -1.0, -30.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(60.0, 0.2, 80.0),
        [1.0, 1.0, 1.0, 1.0],
        4,
    ));
    for k in 1..suns {
        s.lights.push(inf_render::RenderLight {
            kind: inf_render::LightKind::Directional,
            direction: glam::Vec3::new(0.1 * k as f32, 0.3, 1.0).normalize(),
            cast_shadows: true,
            ..Default::default()
        });
    }
    s.mark_dirty();
    s
}

/// The ladder's camera at `step`: a pure translation along the light's `right`.
fn ladder_view(step: u32) -> RenderView {
    RenderView {
        width: LADDER_W,
        height: LADDER_H,
        ..view_beside(f64::from(step) * LADDER_STEP_M, 5.0)
    }
}

/// **A PURE CAMERA TRANSLATION RE-LABELS NOTHING** (island wave VSM2, clause 1)
/// — the counter-assertion, on a real device, through the shipped classifier.
///
/// Wave I7b measured the lit island's dirty split at **400.8 re-slotted / 532.0
/// moved / 0.0 re-cast per rastering frame**, and named the mechanism: a camera
/// travelling 0.9 m against a 1.0 m level-0 page shifts the clipmap window, and a
/// window that shifts *re-labels* every resident page of that level. Under
/// P27.3's `(light, page, stamp)` key each of those re-labelled pages read as a
/// `Geometry` miss — a re-raster of depth the atlas was already holding.
///
/// So the assertion is that the **`moved` bucket is zero**, over a walk that
/// really does shift the windows. It is not "small": the identity a slot is keyed
/// on has no label in it, so a scroll cannot move it at all, and any non-zero
/// reading here would be a page whose *box* moved for another reason.
///
/// What is left is `re-slotted` — the row and column the window newly exposes,
/// which have never been drawn and must be. That is the work the mechanism is
/// supposed to cost, so it is bounded rather than banned, against the resident
/// set the same frame reports.
#[test]
fn a_camera_translation_re_labels_no_page_and_re_rasters_only_what_enters() {
    let Some(gpu) = gpu_or_skip("the VSM clipmap scroll") else {
        return;
    };
    let s = ladder_scene(4);
    let target = inf_render::HeadlessTarget::new(&gpu, LADDER_W, LADDER_H);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = big_ladder();
    renderer.set_settings(rs);

    // Warm: the atlas fills and the cache goes quiet.
    let mut step = 0u32;
    while step < 40 {
        renderer.render(
            &gpu,
            &s,
            &ladder_view(step),
            &target.view,
            (LADDER_W, LADDER_H),
        );
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        step += 1;
    }
    let warm = renderer.vsm_raster_stats().expect("stats");
    let warm_stream = renderer.vsm().expect("live").stats();
    assert!(warm.pages > 0, "the warm-up rasterized nothing at all");

    // …then a pure translation along the light's `right`, a level-0 page at a
    // time so every step shifts the finest window.
    //
    // **Which world cells the atlas holds is tracked independently**, off the
    // residency's own origins, because the load-bearing count below is not "a
    // small number of pages entered" — it is "the pages that were re-drawn are
    // EXACTLY the pages that arrived", and only a set difference can say that.
    let cells = |r: &inf_render::EngineRenderer| {
        page_identities(r)
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
    };
    let mut held = cells(&renderer);
    let mut resident = held.len();
    let mut arrived = 0u64;
    for _ in 0..24u32 {
        renderer.render(
            &gpu,
            &s,
            &ladder_view(step),
            &target.view,
            (LADDER_W, LADDER_H),
        );
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        step += 1;
        let now = cells(&renderer);
        arrived += now.difference(&held).count() as u64;
        resident = resident.max(now.len());
        held = now;
    }
    let end = renderer.vsm_raster_stats().expect("stats");
    let end_stream = renderer.vsm().expect("live").stats();

    // ANTI-VACUITY: the walk really scrolled, and the re-seat really carried.
    let shifts = end_stream.level_shifts - warm_stream.level_shifts;
    let carried = end_stream.scroll_carried - warm_stream.scroll_carried;
    assert!(
        shifts >= 24,
        "only {shifts} level shifts over 24 page-steps"
    );
    assert!(
        carried > 0,
        "the scroll carried no page at all — the windows moved and residency did \
         not follow them, so `moved == 0` below is a statement about an empty atlas"
    );
    assert!(resident > 64, "only {resident} resident pages");

    // **THE CLAIM.**
    let moved = end.dirty_geometry - warm.dirty_geometry;
    assert_eq!(
        moved, 0,
        "{moved} pages were re-rastered because their own box moved, over a walk \
         in which nothing but the camera did. That is the `moved` bucket wave I7b \
         measured at 532.0 a frame, and the whole of what world-cell residency is \
         for"
    );
    assert_eq!(
        end.dirty_casters - warm.dirty_casters,
        0,
        "nothing under a page changed and the caster fold says otherwise"
    );

    // …and what it cost instead, **exactly**: every page charged to `re-slotted`
    // is a world cell that was not in the atlas last frame, and there are no
    // others. A ratio ("small against the resident set") would not do: a cache
    // keyed on the LABEL rather than on the cell re-draws the pages a scroll
    // re-labels and still leaves most of the atlas served, so it passes any
    // threshold loose enough to survive the real entering band. Measured on this
    // fixture — the label key re-draws **1 323** pages where the cell key
    // re-draws 88, and only this equality separates them.
    let entered = end.dirty_slot - warm.dirty_slot;
    let cached = end.cached_pages - warm.cached_pages;
    eprintln!(
        "VSM2 SCROLL: 24 page-steps over {resident} resident pages — {entered} \
         entered ({arrived} arrived), {cached} served from the atlas, {moved} \
         re-labelled, {carried} carried across {shifts} window shifts"
    );
    assert!(arrived > 0, "no cell entered the window over 24 page-steps");
    assert_eq!(
        entered, arrived,
        "{entered} pages were re-drawn for want of a slot and only {arrived} world \
         cells arrived. The difference is pages the atlas was already holding, \
         re-drawn because their grid LABEL moved — which is the whole cost wave \
         I7b measured and this wave removed"
    );
    assert!(
        entered < cached / 20,
        "{entered} pages entered against {cached} served: the atlas is being \
         re-drawn rather than scrolled"
    );
}

/// **THE DEFERRAL BACKLOG DRAINS, AND THE BOUND IS THE ATLAS DIVIDED BY THE
/// BUDGET** (island wave VSM2, clause 2).
///
/// Wave I7b measured **256 pages rastered (the ceiling) and 677 deferred, every
/// frame, for ever** — a queue that could not drain because the thing filling it
/// was the camera itself. Two claims here, and the second is the one that was
/// false before this wave:
///
/// * a burst — a cache flush, which is what a camera cut, an origin rebase and a
///   sun quantum all produce — drains in exactly
///   `ceil(resident / VSM_MAX_RASTER_PAGES)` frames, derived from the state the
///   same run reports rather than from a constant;
/// * and once drained it **stays** drained through a camera that keeps moving.
///
/// `VSM_MAX_RASTER_PAGES` stays where P27.3 put it and stays loud: the deferral
/// counter is read here, and the shipped path still logs every one of them.
#[test]
fn the_deferral_backlog_drains_within_the_atlas_over_the_budget() {
    let Some(gpu) = gpu_or_skip("the VSM deferral drain") else {
        return;
    };
    let s = ladder_scene(4);
    let target = inf_render::HeadlessTarget::new(&gpu, LADDER_W, LADDER_H);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = big_ladder();
    renderer.set_settings(rs);
    let frame = |r: &mut inf_render::EngineRenderer, step: u32| {
        r.render(
            &gpu,
            &s,
            &ladder_view(step),
            &target.view,
            (LADDER_W, LADDER_H),
        );
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        r.vsm_raster_stats().expect("stats")
    };
    let mut step = 0u32;
    while step < 48 {
        frame(&mut renderer, step);
        step += 1;
    }

    // THE BURST. Everything the atlas holds, dirty in one frame — **and the
    // camera is held still through the drain**, because a burst and a scroll
    // running together would be measuring two things and the bound belongs to
    // one of them. The moving half is the second act below.
    let resident = resident_pages(&renderer).len();
    assert!(
        resident > inf_render::VSM_MAX_RASTER_PAGES as usize,
        "only {resident} resident pages against a {} page budget — a burst cannot \
         exceed the ceiling here, so nothing would ever be deferred and this arm \
         would pass on a raster that never defers",
        inf_render::VSM_MAX_RASTER_PAGES
    );
    renderer.vsm_mut().expect("live").flush_page_cache();
    let bound = resident.div_ceil(inf_render::VSM_MAX_RASTER_PAGES as usize);

    let mut before = renderer.vsm_raster_stats().expect("stats");
    let mut drained_at = None;
    let mut deferred_total = 0u64;
    for f in 1..=bound + 4 {
        let now = frame(&mut renderer, step);
        deferred_total += now.deferred_pages - before.deferred_pages;
        let dirty = now.dirty_pages - before.dirty_pages;
        if dirty == 0 && drained_at.is_none() {
            drained_at = Some(f - 1);
        }
        before = now;
    }
    eprintln!(
        "VSM2 DRAIN: {resident} resident pages at a {}-page budget drained in {:?} \
         frames (bound {bound}), {deferred_total} deferrals on the way",
        inf_render::VSM_MAX_RASTER_PAGES,
        drained_at
    );
    assert!(
        deferred_total > 0,
        "the burst deferred nothing, so the drain below is not a drain"
    );
    assert_eq!(
        drained_at,
        Some(bound),
        "a whole-atlas burst of {resident} pages at {} a frame did not drain in \
         the {bound} frames the arithmetic allows",
        inf_render::VSM_MAX_RASTER_PAGES
    );

    // …AND IT STAYS DRAINED through a camera that keeps moving — the half that
    // was false for the whole of wave I7b.
    let settled = renderer.vsm_raster_stats().expect("stats");
    for _ in 0..24 {
        step += 1;
        frame(&mut renderer, step);
    }
    let end = renderer.vsm_raster_stats().expect("stats");
    assert_eq!(
        end.deferred_pages - settled.deferred_pages,
        0,
        "the backlog re-formed under a moving camera — which is exactly the state \
         wave I7b measured at 677 deferred pages a frame, for ever"
    );
}

/// **A CASTER THAT MOVES DURING A SCROLL STILL RE-RASTERS ITS PAGE** (island wave
/// VSM2, clause 3) — the staleness the world-cell key could hide and does not.
///
/// The wave's whole gain is that a page whose *label* changed keeps its texels.
/// The failure mode that buys is a page that keeps texels it should not: a caster
/// that moves in the same frame the window scrolls, where the slot's identity is
/// unchanged and only the caster fold can tell. The existing
/// `a_mover_invalidates_exactly_the_pages_its_bounds_touch` holds the camera
/// still; this one does not.
///
/// Read on the **atlas**, not on a counter: the depth under the caster has to
/// change. The counter is the anti-vacuity beside it.
#[test]
fn a_caster_that_moves_during_a_scroll_re_rasters_its_world_cell() {
    let Some(gpu) = gpu_or_skip("the VSM scroll staleness") else {
        return;
    };
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(backdrop());
    s.mark_dirty();
    let set = big_ladder();
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);

    // Twelve steps of a scrolling camera with a still caster: the control, and the
    // state the mutation below has to be distinguished from.
    let mut step = 0u32;
    let run = |r: &mut inf_render::EngineRenderer, sc: &RenderScene, step: &mut u32, n: u32| {
        for _ in 0..n {
            r.render(
                &gpu,
                sc,
                &view_beside(f64::from(*step) * 0.375, 5.0),
                &target.view,
                (FW, FH),
            );
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            *step += 1;
        }
        r.vsm_raster_stats().expect("stats")
    };
    run(&mut renderer, &s, &mut step, 16);
    let still = run(&mut renderer, &s, &mut step, 12);
    let before = atlas_bits(&gpu, &renderer);
    let quiet = renderer.vsm_raster_stats().expect("stats");
    assert_eq!(
        quiet.dirty_casters - still.dirty_casters,
        0,
        "a still caster under a scrolling camera invalidated pages by content — \
         the control this arm needs is not quiet"
    );

    // …now move the caster IN THE SAME FRAME the window scrolls. Half a cube
    // along the light's `right`, which stays inside the same level-0 page for
    // part of its footprint and crosses into the next for the rest — so the
    // question "did the slot notice" is asked of a cell whose label is moving too.
    let mut moved = s.clone();
    moved.instances[0].translation.x += 0.9;
    moved.mark_dirty();
    let after_stats = run(&mut renderer, &moved, &mut step, 1);
    let after = atlas_bits(&gpu, &renderer);

    let recast = after_stats.dirty_casters - quiet.dirty_casters;
    let changed = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "VSM2 SCROLL-STALENESS: {recast} pages re-cast, {changed} atlas texels \
         changed"
    );
    assert!(
        recast > 0,
        "a caster moved while the clipmap scrolled and not one page was charged \
         to the casters — the world-cell key is holding depth of a world that has \
         moved"
    );
    assert!(
        changed > 0,
        "the atlas is byte-identical after a caster moved under a scrolling \
         camera: the shadow is stale on the device, whatever the counters say"
    );
}

/// **THE SUN'S QUANTUM REACHES THE ATLAS** (island wave VSM2) — P27.3's clause-3
/// policy, on a device, for the first time.
///
/// P27.3 quantizes the sun's direction so a page's content is a function of the
/// *quantized* angle: below the quantum nothing is re-drawn, above it everything
/// is. Until this wave that held **by construction** — the geometric stamp was the
/// page matrix folded bit for bit, and the direction is one of its inputs — so
/// there was nothing to arm and nothing was armed (the tree drives the quantizer's
/// arithmetic, `the_sun_quantum_is_one_shadow_texel_at_the_reference_height`, and
/// never a sun that turns).
///
/// The wave's world-cell key replaces those matrix bits with
/// `ClipmapLayout::content_key`, which names the quantized direction **explicitly**
/// — so the property stops being structural and starts being a line of code that
/// can be deleted. Measured: dropping the direction from that fold leaves every
/// GPU arm in this file green. It leaves this one red.
///
/// Read on the atlas, because "the sun moved and the shadows did not" is a claim
/// about texels.
#[test]
fn a_sun_that_crosses_its_quantum_re_rasters_the_atlas_and_one_that_does_not_does_not() {
    let Some(gpu) = gpu_or_skip("the VSM sun quantum") else {
        return;
    };
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(backdrop());
    s.mark_dirty();
    let set = settings_with(64);
    let q = inf_render::vsm_sun_quantum(&set);
    let base = s.lights[0].direction;
    // Under and over, through the shipped quantizer rather than through the
    // formula beside it — the fixture's own precondition.
    let under = (base + glam::Vec3::X * (q * 0.1)).normalize();
    let over = (base + glam::Vec3::X * (q * 20.0)).normalize();
    assert_eq!(
        inf_render::quantize_light_dir(base, q),
        inf_render::quantize_light_dir(under, q),
        "the fixture's sub-quantum nudge crosses the quantum"
    );
    assert_ne!(
        inf_render::quantize_light_dir(base, q),
        inf_render::quantize_light_dir(over, q),
        "the fixture's nudge does not cross the quantum"
    );

    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);
    let v = view(5.0);
    let turn = |r: &mut inf_render::EngineRenderer, d: glam::Vec3, n: u32| {
        let mut sc = s.clone();
        sc.lights[0].direction = d;
        sc.mark_dirty();
        for _ in 0..n {
            r.render(&gpu, &sc, &v, &target.view, (FW, FH));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
        (r.vsm_raster_stats().expect("stats"), atlas_bits(&gpu, r))
    };
    let (warm, before) = turn(&mut renderer, base, 12);
    assert!(warm.pages > 0, "the warm-up rasterized nothing");

    // 1. UNDER the quantum: the quantizer swallows it whole.
    let (quiet, still) = turn(&mut renderer, under, 1);
    assert_eq!(
        quiet.dirty_pages - warm.dirty_pages,
        0,
        "a sub-quantum sun nudge re-rasterized pages — the quantum is not the \
         policy P27.3 measured"
    );
    assert_eq!(still, before, "…and it moved the atlas anyway");

    // 2. OVER it: every page of the box is looking somewhere new.
    let (moved, after) = turn(&mut renderer, over, 1);
    let geo = moved.dirty_geometry - quiet.dirty_geometry;
    let changed = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("VSM2 SUN QUANTUM: {geo} pages moved, {changed} atlas texels changed");
    assert!(
        geo > 0,
        "the sun crossed its quantum and not one page was charged to its own \
         geometry — the quantized direction has left the clipmap's content key \
         and a turning sun now drags stale shadows behind it"
    );
    assert!(
        changed > 0,
        "the sun crossed its quantum and the atlas is byte-identical"
    );
}

/// **A FLOATING-ORIGIN REBASE MOVES NO PAGE'S IDENTITY** (island wave VSM2,
/// clause 1's hazard) — the I4b defect's exact shape, armed where this wave
/// introduced a new key.
///
/// The I4b audit's one on-screen defect was a cache whose key carried the eye
/// bucket, the bands, the caster stamp and the content fold **and not the
/// origin**, so a rebase re-uploaded a merge of the stale half and put every
/// scatter shadow caster a kilometre out of place. `PageIdent` is a new key in the
/// same subsystem, so the question has to be asked of it too — and the answer is
/// a different one, on purpose:
///
/// * the identity is a **world cell**, so a rebase moves neither it nor
///   residency: `re-slotted` and `moved` both read **zero**;
/// * and the atlas still refreshes, because a caster's stamp folds its
///   **render-local** model matrix, which a rebase does move — so every page with
///   a caster in it is charged to `re-cast` and re-drawn. The correctness is in
///   the *fold*, exactly where it was before this wave, and the world-cell key is
///   blind to the rebase without being wrong about it.
#[test]
fn a_floating_origin_rebase_moves_no_pages_identity_and_still_refreshes_the_atlas() {
    let Some(gpu) = gpu_or_skip("the VSM origin rebase") else {
        return;
    };
    let mut s = scene(0, 0.5, 1.0);
    s.instances.push(backdrop());
    s.mark_dirty();
    let set = big_ladder();
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let mut rs = *renderer.settings();
    rs.vsm = set;
    renderer.set_settings(rs);
    let here = view_beside(0.0, 5.0);
    for _ in 0..16 {
        renderer.render(&gpu, &s, &here, &target.view, (FW, FH));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let quiet = renderer.vsm_raster_stats().expect("stats");

    // THE REBASE: the same world eye, a render origin a kilometre away. Nothing
    // in the world moved and every render-local coordinate did.
    let rebased = RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::new(
            inf_math::REBASE_DISTANCE,
            0.0,
            inf_math::REBASE_DISTANCE,
        )),
        ..here
    };
    assert_ne!(
        here.origin.to_render(here.eye_world),
        rebased.origin.to_render(rebased.eye_world),
        "the two views share a render-local eye, so the rebase case was never \
         reached"
    );
    renderer.render(&gpu, &s, &rebased, &target.view, (FW, FH));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let after = renderer.vsm_raster_stats().expect("stats");

    assert_eq!(
        after.cut_flushes, quiet.cut_flushes,
        "the rebase read as a camera cut — `is_camera_cut` takes the f64 WORLD \
         eye, and a flush here would make the two claims below unfalsifiable"
    );
    assert_eq!(
        (
            after.dirty_slot - quiet.dirty_slot,
            after.dirty_geometry - quiet.dirty_geometry
        ),
        (0, 0),
        "a rebase moved a page's identity: the world-cell key is carrying a \
         render-local term and the I4b defect has a second home"
    );
    let recast = after.dirty_casters - quiet.dirty_casters;
    eprintln!("VSM2 REBASE: {recast} pages re-cast, 0 re-slotted, 0 moved");
    assert!(
        recast > 0,
        "the atlas kept every page across a kilometre rebase. The caster fold is \
         the whole of what carries the origin into this key, so if it does not \
         move here it will not move for any render-local change either, and the \
         world-cell key has no second guard"
    );
    // **What this arm does NOT say** (the VSM2 audit). The first write-up's
    // reason — *"the stored depth is render-local, so a page not re-drawn holds
    // depth measured from an origin that has moved"* — is backwards: a page's
    // depth is `(p − centre) · forward` and a rebase moves `p` and `centre` by
    // the same delta, so the stored value is origin-invariant and a page NOT
    // re-drawn would in fact be right. What the re-cast above measures is that
    // the caster fold *notices* the rebase, which is the guard the key needs the
    // day something render-local does reach a page's content — and the cost of
    // it is a whole-atlas burst per `inf_math::REBASE_DISTANCE` of travel, which
    // is the second of the two burst sources the ledger's clause 5 routes.
}

// ── P27.5: THE SKINNED CULL SPHERE, RULED WITH ITS EXACT FIX MEASURED ───────

/// **The skinned caster's bound stays the inflated bind pose, and the exact fix
/// is measured rather than described** (P27.5) — the P27.2/P27.3 remainder.
///
/// The carried sentence was: *"the skinned cull sphere is still the bind pose
/// inflated 50 %, and P27.3 made that margin load-bearing rather than exact — it
/// now costs re-rasterized pages as well as vertex invocations. The exact bound
/// is the posed AABB the renderer is still not handed."*
///
/// # One correction, and it changes where the fix lives
///
/// *"the renderer is still not handed"* implied the fix needs `inf-anim`'s
/// cooperation. It does not. A skinned instance already carries its **joint
/// palette** — the renderer has it, uses it, and folds it into the caster's
/// content stamp — so a bound that follows the pose is computable from data in
/// hand: transform the bind sphere's centre by each joint and take the union,
/// scaling the radius by each matrix's largest axis scale. No new data crosses
/// any boundary, and `inf-anim` stays a dev-dependency.
///
/// # The measurement, and therefore the ruling
///
/// This arm computes both bounds for the pose `a_skinned_casters_cull_sphere_
/// contains_a_pose_that_left_the_bind_pose` already drives, and both contain the
/// posed geometry — so what separates them is cost, which is what a ruling has
/// to be made on. The palette union is measured **tighter**, and the ledger
/// carries the number.
///
/// **Landed in wave NPC1b, and this arm is why it could be.** The shipped bound
/// is `SkinnedCasterGeom::joint_bounds` unioned through the instance's own
/// palette — tighter still than the whole-mesh union computed here, because it
/// carries each joint's OWN vertices rather than the whole bind sphere — and it
/// is opt-in per instance (`SkinnedShadow::Posed`), which is what keeps the
/// margin's page-invalidation behaviour unchanged for everything that has not
/// asked. The reasoning this paragraph used to carry still stands and is now
/// stated where the constant is: a tighter caster sphere changes which pages the
/// cull keeps and therefore which pages a mover invalidates, which is the exact
/// quantity `phase27_gate`'s arm (c) asserts, so making it the DEFAULT is a
/// re-measurement rather than a constant change. This arm stays as the pure
/// statement of the ruling; `the_posed_bound_is_tighter_than_the_margin_and_still_contains_the_pose`
/// is the shipped one, and measures 67 % of the radius on this same fixture.
#[test]
fn the_palette_union_bound_is_tighter_than_the_shipped_pose_margin() {
    // The fixture's bind pose: a quad at ±1 in x and y, so its bind sphere is
    // centred at the origin with radius √2.
    let corners = [
        glam::Vec3::new(-1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, -1.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 0.0),
        glam::Vec3::new(-1.0, 1.0, 0.0),
    ];
    let bind_centre = glam::Vec3::ZERO;
    let bind_radius = corners
        .iter()
        .map(|c| (*c - bind_centre).length())
        .fold(0.0f32, f32::max);
    assert!((bind_radius - std::f32::consts::SQRT_2).abs() < 1e-5);

    // The pose: 0.6 m along x, through one joint — the same palette the device
    // arm above drives.
    let palette = [glam::Mat4::from_translation(glam::Vec3::new(0.6, 0.0, 0.0))];
    let posed: Vec<glam::Vec3> = corners
        .iter()
        .map(|c| palette[0].transform_point3(*c))
        .collect();

    // 1. THE SHIPPED BOUND: the bind sphere, inflated, at the bind centre.
    // , which is how the shipped code spells it: the constant
    // is the INFLATION and not the factor (0.5 means half again).
    let shipped_r = bind_radius * (1.0 + inf_render::SKINNED_POSE_MARGIN);
    for p in &posed {
        assert!(
            (*p - bind_centre).length() <= shipped_r + 1e-4,
            "the shipped margin does not contain the pose — the device arm's \
             premise is wrong"
        );
    }

    // 2. THE PALETTE UNION: each joint's transformed bind sphere, unioned. The
    //    radius scale is the matrix's largest axis length, which is what makes
    //    it conservative under a scaling joint.
    let mut union_centre = glam::Vec3::ZERO;
    let mut union_r = 0.0f32;
    for (i, m) in palette.iter().enumerate() {
        let c = m.transform_point3(bind_centre);
        let scale = [
            m.x_axis.truncate(),
            m.y_axis.truncate(),
            m.z_axis.truncate(),
        ]
        .iter()
        .map(|a| a.length())
        .fold(0.0f32, f32::max);
        let r = bind_radius * scale;
        if i == 0 {
            union_centre = c;
            union_r = r;
            continue;
        }
        // Merge two spheres: the smallest sphere containing both.
        let d = (c - union_centre).length();
        if d + r <= union_r {
            continue;
        }
        let nr = 0.5 * (union_r + d + r);
        union_centre += (c - union_centre) * ((nr - union_r) / d.max(1e-6));
        union_r = nr;
    }
    for p in &posed {
        assert!(
            (*p - union_centre).length() <= union_r + 1e-4,
            "the palette union does not contain the pose, so it is not a legal \
             replacement whatever it costs"
        );
    }

    // THE COMPARISON, which is the ruling's evidence.
    println!(
        "phase27 skinned bound: shipped r {shipped_r:.4} m at {bind_centre:?}, \
         palette union r {union_r:.4} m at {union_centre:?} — {:.0} % of the \
         radius and {:.0} % of the volume",
        100.0 * union_r / shipped_r,
        100.0 * (union_r / shipped_r).powi(3)
    );
    assert!(
        union_r < shipped_r,
        "the palette union ({union_r}) is no tighter than the shipped margin \
         ({shipped_r}), so the carried remainder has no fix in it and the \
         ledger should say so instead"
    );
    // …and the margin is not merely slack: dropping it entirely leaves the pose
    // OUTSIDE the bind sphere, which is why it exists at all.
    assert!(
        posed
            .iter()
            .any(|p| (*p - bind_centre).length() > bind_radius),
        "the fixture's pose never leaves its bind sphere, so neither bound is \
         under test"
    );
    // The same `const` block, for the same reason: a margin of zero would make
    // the shipped bound the bind pose exactly, and the comparison above would
    // silently become "the union is tighter than the thing it equals".
    const {
        assert!(inf_render::SKINNED_POSE_MARGIN > 0.0);
    }
}
