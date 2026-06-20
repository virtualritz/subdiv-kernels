//! Interactive Bevy + egui showcase for `subdiv-kernels`.
//!
//! A regular dodecahedron is subdivided live by Catmull–Clark. The egui panel
//! controls the subdivision level (0–6), which face carries a creased edge ring
//! and how sharp it is, whether to shade with exact limit-surface normals, and a
//! wireframe overlay; drag to orbit, scroll to zoom.
//!
//! Run with: `cargo run --example bevy --features bevy`
//!
//! The kernel stays geometry-agnostic — all Bevy/egui code lives here; the
//! crate exposes no Bevy types.

use bevy::asset::RenderAssetUsages;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use std::collections::HashMap;
use std::num::NonZeroU8;
use subdiv_kernels::{Mesh as Cage, Refiner, Scheme, SchemeOptions, UniformRefine};

const MAX_LEVEL: u8 = 6;
/// A regular dodecahedron has 12 faces.
const FACE_COUNT: usize = 12;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .insert_resource(Params {
            level: 2,
            creased: true,
            crease_face: 0,
            crease_value: 4.0,
            limit_normals: false,
            wireframe: false,
            dirty: false,
        })
        .insert_resource(UiWantsPointer(false))
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_panel)
        .add_systems(Update, (orbit_camera, rebuild_mesh, draw_cage_wire))
        .run();
}

// ── Resources / components ──────────────────────────────────────────────

#[derive(Resource)]
struct Params {
    level: u8,
    creased: bool,
    /// Which face's edge ring to crease (`0..FACE_COUNT`).
    crease_face: usize,
    /// Crease sharpness (1 = soft, 10 ≈ infinitely sharp).
    crease_value: f32,
    /// Shade with exact limit-surface normals instead of per-triangle normals.
    limit_normals: bool,
    /// Draw the triangle wireframe over the surface.
    wireframe: bool,
    dirty: bool,
}

/// Set by the egui system each frame so the orbit camera ignores drags that
/// land on the panel.
#[derive(Resource)]
struct UiWantsPointer(bool);

/// Handle to the surface mesh asset, rebuilt in place on parameter changes.
#[derive(Resource)]
struct SurfaceMesh(Handle<Mesh>);

/// The original cage edges tracked through subdivision (kernel `edge_polylines`)
/// as segments at current vertex positions, for the wireframe overlay. Rebuilt
/// with the mesh.
#[derive(Resource, Default)]
struct CageWire(Vec<[Vec3; 2]>);

/// The control cage (dodecahedron) in `subdiv-kernels` form, plus the edge
/// indices of every face (so any face's ring can be creased).
#[derive(Resource)]
struct CageData {
    vertex_count: u32,
    face_vertex_counts: Vec<u32>,
    face_vertex_indices: Vec<u32>,
    edge_vertices: Vec<[u32; 2]>,
    positions: Vec<[f32; 3]>,
    face_edges: Vec<Vec<usize>>,
}

#[derive(Resource, Clone, Copy)]
struct Orbit {
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Orbit {
    fn transform(&self) -> Transform {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let pos = Vec3::new(
            self.radius * cp * sy,
            self.radius * sp,
            self.radius * cp * cy,
        );
        Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y)
    }
}

// ── Setup ───────────────────────────────────────────────────────────────

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    params: Res<Params>,
) {
    let cage = build_cage();
    let (mesh, wire) = build_surface(&cage, &params);
    let handle = meshes.add(mesh);

    commands.spawn((
        Mesh3d(handle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.6, 0.85),
            perceptual_roughness: 0.55,
            ..default()
        })),
        Transform::default(),
    ));
    commands.insert_resource(SurfaceMesh(handle));
    commands.insert_resource(CageWire(wire));
    commands.insert_resource(cage);

    let orbit = Orbit {
        yaw: 0.7,
        pitch: 0.45,
        radius: 5.5,
    };
    // Bevy 0.19 lighting is per-camera: `AmbientLight` is `#[require(Camera)]`,
    // and the directional light is parented to the camera so it lights the view
    // from a consistent angle as you orbit.
    commands
        .spawn((
            Camera3d::default(),
            orbit.transform(),
            AmbientLight {
                brightness: 220.0,
                ..default()
            },
        ))
        .with_children(|camera| {
            camera.spawn((
                DirectionalLight {
                    illuminance: 6000.0,
                    shadow_maps_enabled: true,
                    ..default()
                },
                // Local to the camera: up-and-left of the view direction.
                Transform::from_xyz(-3.0, 5.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
            ));
        });
    commands.insert_resource(orbit);
}

// ── egui panel ──────────────────────────────────────────────────────────

/// Toggle-button caption for a boolean.
fn on_off(b: bool) -> &'static str {
    if b { "On" } else { "Off" }
}

fn ui_panel(
    mut contexts: EguiContexts,
    mut params: ResMut<Params>,
    mut ui_wants: ResMut<UiWantsPointer>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("subdiv-kernels")
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.label("Catmull–Clark · Dodecahedron");
            ui.separator();
            let mut changed = false;
            egui::Grid::new("controls")
                .num_columns(2)
                .spacing([24.0, 8.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Level");
                    changed |= ui
                        .add(egui::Slider::new(&mut params.level, 0..=MAX_LEVEL))
                        .changed();
                    ui.end_row();

                    ui.label("Creased Ring");
                    let caption = on_off(params.creased);
                    changed |= ui.toggle_value(&mut params.creased, caption).changed();
                    ui.end_row();

                    // Crease controls disable when the ring is off.
                    let creased = params.creased;
                    ui.add_enabled(creased, egui::Label::new("Crease Face"));
                    changed |= ui
                        .add_enabled(
                            creased,
                            egui::Slider::new(&mut params.crease_face, 0..=FACE_COUNT - 1),
                        )
                        .changed();
                    ui.end_row();

                    ui.add_enabled(creased, egui::Label::new("Crease Sharpness"));
                    changed |= ui
                        .add_enabled(
                            creased,
                            egui::Slider::new(&mut params.crease_value, 1.0..=10.0),
                        )
                        .changed();
                    ui.end_row();

                    ui.label("Limit Normals");
                    let caption = on_off(params.limit_normals);
                    changed |= ui
                        .toggle_value(&mut params.limit_normals, caption)
                        .changed();
                    ui.end_row();

                    // Render-only: overlays the original cage edges tracked through
                    // subdivision (edge_polylines), not the render triangulation.
                    ui.label("Cage Wireframe");
                    let caption = on_off(params.wireframe);
                    ui.toggle_value(&mut params.wireframe, caption);
                    ui.end_row();
                });
            if changed {
                params.dirty = true;
            }
            ui.separator();
            ui.label("Drag to Orbit · Scroll to Zoom");
        });
    ui_wants.0 = ctx.egui_wants_pointer_input();
    Ok(())
}

// ── Orbit camera ────────────────────────────────────────────────────────

fn orbit_camera(
    ui_wants: Res<UiWantsPointer>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<Orbit>,
    mut camera: Single<&mut Transform, With<Camera3d>>,
) {
    // Drain readers regardless, so leaving the panel doesn't cause a jump.
    let drag: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel.read().map(|w| w.y).sum();
    if ui_wants.0 {
        return;
    }

    let mut changed = false;
    if buttons.pressed(MouseButton::Left) && drag != Vec2::ZERO {
        orbit.yaw -= drag.x * 0.005;
        orbit.pitch = (orbit.pitch + drag.y * 0.005).clamp(-1.4, 1.4);
        changed = true;
    }
    if scroll != 0.0 {
        orbit.radius = (orbit.radius - scroll * 0.4).clamp(2.0, 20.0);
        changed = true;
    }
    if changed {
        **camera = orbit.transform();
    }
}

// ── Rebuild on parameter change ─────────────────────────────────────────

fn rebuild_mesh(
    mut params: ResMut<Params>,
    cage: Res<CageData>,
    surface: Res<SurfaceMesh>,
    mut wire: ResMut<CageWire>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !params.dirty {
        return;
    }
    params.dirty = false;
    let (mesh, edges) = build_surface(&cage, &params);
    // The handle is created in `setup` and never removed, so this can't fail.
    let _ = meshes.insert(surface.0.id(), mesh);
    wire.0 = edges;
}

/// Draw the original cage edges (subdivided) when the overlay is enabled.
fn draw_cage_wire(params: Res<Params>, wire: Res<CageWire>, mut gizmos: Gizmos) {
    if !params.wireframe {
        return;
    }
    let color = Color::srgb(0.05, 0.05, 0.08);
    for [a, b] in &wire.0 {
        gizmos.line(*a, *b, color);
    }
}

// ── Subdivision → Bevy mesh ─────────────────────────────────────────────

/// Subdivide the cage to `params.level` (0 = base cage) and build a triangulated
/// Bevy mesh, plus the overlay: the **original** cage edges tracked through
/// subdivision (`UniformRefine::edge_polylines` — each input edge becomes a
/// polyline of refined vertices), as curves on the refined/limit surface — not
/// the full fine edge set. Normals are either exact limit-surface normals
/// (`tangent1 × tangent2` from the sectored limit stencils) or averaged from the
/// triangles.
fn build_surface(cage: &CageData, params: &Params) -> (Mesh, Vec<[Vec3; 2]>) {
    let crease_ring = &cage.face_edges[params.crease_face.min(FACE_COUNT - 1)];
    let edge_creases: Vec<f32> = (0..cage.edge_vertices.len())
        .map(|i| {
            if params.creased && crease_ring.contains(&i) {
                params.crease_value
            } else {
                0.0
            }
        })
        .collect();

    let mesh = Cage {
        vertex_count: cage.vertex_count,
        face_vertex_counts: cage.face_vertex_counts.clone(),
        face_vertex_indices: cage.face_vertex_indices.clone(),
        edge_vertices: cage.edge_vertices.clone(),
        edge_creases,
        vertex_corners: vec![0.0; cage.vertex_count as usize],
    };

    // `positions/normals/tris` build the render mesh; `wire` is the original-cage
    // edge overlay. Level 0 is the base cage; it has no refinement result, so
    // limit normals fall back to calculated ones and the overlay is the raw cage
    // edges.
    #[allow(clippy::type_complexity)]
    let (positions, normals, tris, wire): (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<u32>,
        Vec<[Vec3; 2]>,
    ) = if params.level == 0 {
        let tris = fan_triangulate(&cage.face_vertex_counts, &cage.face_vertex_indices);
        let normals = smooth_normals(&cage.positions, &tris);
        let wire = cage
            .edge_vertices
            .iter()
            .map(|e| {
                [
                    Vec3::from(cage.positions[e[0] as usize]),
                    Vec3::from(cage.positions[e[1] as usize]),
                ]
            })
            .collect();
        (cage.positions.clone(), normals, tris, wire)
    } else {
        let refiner =
            Refiner::new(mesh, Scheme::CatmullClark, SchemeOptions::default()).expect("valid cage");
        // `edge_polylines` tracks the refined vertices lying along each *input*
        // edge — i.e. the original cage edges as they subdivide.
        let req = UniformRefine {
            levels: NonZeroU8::new(params.level).expect("level >= 1"),
            edge_polylines: true,
            ..Default::default()
        };
        let result = refiner.refine_uniform(&req).expect("refinement");

        let (positions, normals, tris, vertex_pos) = if params.limit_normals {
            // Push vertices onto the exact limit surface and shade with the limit
            // normal (tangent1 × tangent2). Sectored stencils give per-corner
            // tangents, so each side of a crease shades with its own normal — a
            // single per-vertex normal would be wrong on one side of the ridge.
            let limit = result
                .compose_sectored_limit_stencils(cage.positions.len())
                .expect("limit stencils");
            let vertex_pos = limit.position.interpolate(&cage.positions); // per vertex
            let st1 = limit.tangent1.interpolate(&cage.positions); // per sector
            let st2 = limit.tangent2.interpolate(&cage.positions);
            let sector_n: Vec<[f32; 3]> = st1
                .iter()
                .zip(&st2)
                .map(|(a, b)| {
                    Vec3::from(*a)
                        .cross(Vec3::from(*b))
                        .normalize_or_zero()
                        .to_array()
                })
                .collect();

            // Expand to per-corner vertices so each corner carries its sector's
            // normal, then fan-triangulate each face.
            let counts = &result.topology.face_vertex_counts;
            let fvi = &result.topology.face_vertex_indices;
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            let mut tris = Vec::new();
            let mut corner = 0usize;
            for &n in counts {
                let n = n as usize;
                let base = positions.len() as u32;
                for k in 0..n {
                    let fc = corner + k;
                    positions.push(vertex_pos[fvi[fc] as usize]);
                    normals.push(sector_n[limit.corner_sector[fc] as usize]);
                }
                for k in 1..n - 1 {
                    tris.extend_from_slice(&[base, base + k as u32, base + (k + 1) as u32]);
                }
                corner += n;
            }
            (positions, normals, tris, vertex_pos)
        } else {
            let vertex_pos = result.interpolate(&cage.positions);
            let tris = fan_triangulate(
                &result.topology.face_vertex_counts,
                &result.topology.face_vertex_indices,
            );
            let normals = smooth_normals(&vertex_pos, &tris);
            (vertex_pos.clone(), normals, tris, vertex_pos)
        };

        // Connect consecutive vertices of each input-edge polyline into segments.
        let wire = result
            .edge_polylines
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .flat_map(|poly| {
                poly.windows(2).map(|w| {
                    [
                        Vec3::from(vertex_pos[w[0] as usize]),
                        Vec3::from(vertex_pos[w[1] as usize]),
                    ]
                })
            })
            .collect();
        (positions, normals, tris, wire)
    };

    let render = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(tris));
    (render, wire)
}

fn fan_triangulate(counts: &[u32], indices: &[u32]) -> Vec<u32> {
    let mut tris = Vec::new();
    let mut o = 0usize;
    for &n in counts {
        let n = n as usize;
        for k in 1..n - 1 {
            tris.extend_from_slice(&[indices[o], indices[o + k], indices[o + k + 1]]);
        }
        o += n;
    }
    tris
}

fn smooth_normals(positions: &[[f32; 3]], tris: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for t in tris.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let pa = Vec3::from(positions[a]);
        let n = (Vec3::from(positions[b]) - pa).cross(Vec3::from(positions[c]) - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.iter()
        .map(|v| v.normalize_or_zero().to_array())
        .collect()
}

// ── Dodecahedron cage (built geometrically — correct by construction) ────

fn build_cage() -> CageData {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let inv = 1.0 / phi;

    // 20 vertices: 8 cube corners + 12 from the (0, ±1/φ, ±φ) cyclic set.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(20);
    for &x in &[-1.0_f32, 1.0] {
        for &y in &[-1.0_f32, 1.0] {
            for &z in &[-1.0_f32, 1.0] {
                positions.push([x, y, z]);
            }
        }
    }
    for &a in &[-inv, inv] {
        for &b in &[-phi, phi] {
            positions.push([0.0, a, b]);
            positions.push([a, b, 0.0]);
            positions.push([b, 0.0, a]);
        }
    }

    // 12 face normals: the dodecahedron's face centers point along (0, ±φ, ±1)
    // and its cyclic permutations (the φ component leads, unlike the vertices).
    let mut centers: Vec<Vec3> = Vec::with_capacity(12);
    for &a in &[-1.0_f32, 1.0] {
        for &b in &[-phi, phi] {
            centers.push(Vec3::new(0.0, b, a));
            centers.push(Vec3::new(b, a, 0.0));
            centers.push(Vec3::new(a, 0.0, b));
        }
    }

    // Each face = the 5 vertices nearest its center, ordered CCW about it.
    let faces: Vec<Vec<u32>> = centers
        .iter()
        .map(|c| {
            let cn = c.normalize();
            let mut ring: Vec<usize> = (0..positions.len()).collect();
            ring.sort_by(|&i, &j| {
                let di = Vec3::from(positions[j]).dot(cn);
                let dj = Vec3::from(positions[i]).dot(cn);
                di.partial_cmp(&dj).unwrap()
            });
            ring.truncate(5);
            let u = perp(cn);
            let w = cn.cross(u);
            ring.sort_by(|&i, &j| {
                angle(positions[i], u, w)
                    .partial_cmp(&angle(positions[j], u, w))
                    .unwrap()
            });
            ring.into_iter().map(|i| i as u32).collect()
        })
        .collect();

    // Dedup edges from faces; record each face's edge ring (so any face is
    // creasable).
    let mut edge_index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edge_vertices: Vec<[u32; 2]> = Vec::new();
    let mut face_edge = |a: u32, b: u32| -> usize {
        let key = (a.min(b), a.max(b));
        *edge_index.entry(key).or_insert_with(|| {
            edge_vertices.push([key.0, key.1]);
            edge_vertices.len() - 1
        })
    };

    let mut face_edges: Vec<Vec<usize>> = Vec::with_capacity(faces.len());
    let mut face_vertex_indices = Vec::new();
    for face in &faces {
        let ring = (0..face.len())
            .map(|k| face_edge(face[k], face[(k + 1) % face.len()]))
            .collect();
        face_edges.push(ring);
        face_vertex_indices.extend_from_slice(face);
    }

    CageData {
        vertex_count: positions.len() as u32,
        face_vertex_counts: vec![5; faces.len()],
        face_vertex_indices,
        edge_vertices,
        positions,
        face_edges,
    }
}

/// A unit vector perpendicular to `c`.
fn perp(c: Vec3) -> Vec3 {
    let a = if c.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    (a - c * a.dot(c)).normalize()
}

/// Angle of `v` in the plane spanned by basis `(u, w)`.
fn angle(v: [f32; 3], u: Vec3, w: Vec3) -> f32 {
    let p = Vec3::from(v);
    p.dot(w).atan2(p.dot(u))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(level: u8, creased: bool, crease_face: usize, limit_normals: bool) -> Params {
        Params {
            level,
            creased,
            crease_face,
            crease_value: 4.0,
            limit_normals,
            wireframe: false,
            dirty: false,
        }
    }

    /// Extract a triangle mesh's positions, normals, and indices.
    fn mesh_data(mesh: &Mesh) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        use bevy::mesh::VertexAttributeValues::Float32x3;
        let pos = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(Float32x3(v)) => v.clone(),
            _ => panic!("positions"),
        };
        let nrm = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(Float32x3(v)) => v.clone(),
            _ => panic!("normals"),
        };
        let idx = match mesh.indices() {
            Some(Indices::U32(v)) => v.clone(),
            _ => panic!("indices"),
        };
        (pos, nrm, idx)
    }

    #[test]
    fn limit_normals_smooth_at_crease() {
        // Regression for the limit-normal artifact at the screenshot config
        // (level 4, crease face 11, sharpness 5.0, limit normals): every
        // triangle-corner normal must agree with its own face plane. Non-sectored
        // limit normals dropped the worst agreement to ~0.42 on the crease ridge
        // (the two mis-lit facets); sectored per-corner normals keep it ~1.0.
        let cage = build_cage();
        let mut p = params(4, true, 11, true);
        p.crease_value = 5.0;
        let (mesh, _) = build_surface(&cage, &p);
        let (pos, nrm, idx) = mesh_data(&mesh);

        let mut min_dot = f32::INFINITY;
        for t in idx.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let pa = Vec3::from(pos[a]);
            let face_n = (Vec3::from(pos[b]) - pa)
                .cross(Vec3::from(pos[c]) - pa)
                .normalize_or_zero();
            if face_n == Vec3::ZERO {
                continue; // degenerate sliver; nothing to shade
            }
            for &v in &[a, b, c] {
                min_dot = min_dot.min(Vec3::from(nrm[v]).dot(face_n));
            }
        }
        assert!(
            min_dot > 0.8,
            "limit normals disagree with face planes (min n·face = {min_dot:.3}); \
             crease normals likely not sectored"
        );
    }

    #[test]
    fn cage_is_valid_and_subdivides() {
        let cage = build_cage();
        // Regular dodecahedron: 20 vertices, 12 pentagons, 30 edges
        // (Euler: V - E + F = 20 - 30 + 12 = 2).
        assert_eq!(cage.vertex_count, 20);
        assert_eq!(cage.face_vertex_counts.len(), FACE_COUNT);
        assert!(cage.face_vertex_counts.iter().all(|&n| n == 5));
        assert_eq!(cage.edge_vertices.len(), 30);
        assert_eq!(cage.face_edges.len(), FACE_COUNT);
        assert!(cage.face_edges.iter().all(|r| r.len() == 5));

        // Every level (base + 4), creased and not, calculated and limit normals,
        // across a few faces, must build cleanly (a bad cage or limit-eval would
        // panic in the kernel).
        for creased in [false, true] {
            for limit_normals in [false, true] {
                for level in 0..=MAX_LEVEL {
                    let face = level as usize % FACE_COUNT;
                    let (mesh, wire) =
                        build_surface(&cage, &params(level, creased, face, limit_normals));
                    assert!(mesh.count_vertices() > 0);
                    assert!(!wire.is_empty());
                }
            }
        }
    }
}
