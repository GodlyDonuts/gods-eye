//! Open an interactive window showing a mesh.
//!
//! Usage:
//!   cargo run -p ge-viewer --example view --features window               # demo sphere
//!   cargo run -p ge-viewer --example view --features window -- PATH.ply   # a reconstruction
//!
//! Generate a PLY first with e.g. `cargo run -p ge-cli -- reconstruct`.

fn main() -> anyhow::Result<()> {
    let arg = std::env::args().nth(1);
    let (mesh, title) = match arg.as_deref() {
        None | Some("sphere") => (
            ge_mesh::demo::sphere_mesh(0.7),
            "Gods Eye — demo sphere".to_string(),
        ),
        Some(path) => (
            ge_viewer::load_ply(std::path::Path::new(path))?,
            format!("Gods Eye — {path}"),
        ),
    };
    println!(
        "viewing {} vertices / {} triangles — drag to orbit, scroll to zoom",
        mesh.vertex_count(),
        mesh.triangle_count()
    );
    ge_viewer::view_mesh(&mesh, &title)
}
