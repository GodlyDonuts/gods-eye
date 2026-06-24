//! Live mesh viewer.
//!
//! Two implementations are planned behind the [`Viewer`] trait: `rerun`
//! out-of-process for development (rich, but needs a 16 GB dev box), and
//! `three-d` + `egui` in-process for the shipped experience (fits 8 GB). M0
//! ships [`NullViewer`] plus a counting viewer for headless CI.

use ge_mesh::Mesh;

/// A sink for live mesh updates.
pub trait Viewer {
    /// Log/replace the current mesh. Called once per extraction tick.
    fn log_mesh(&mut self, mesh: &Mesh) -> anyhow::Result<()>;
}

/// Discards everything. For headless runs where no view is needed.
pub struct NullViewer;

impl Viewer for NullViewer {
    fn log_mesh(&mut self, _mesh: &Mesh) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Records how many times it was called and the last triangle count — used by
/// headless CI to assert the spine actually produced mesh updates.
#[derive(Default)]
pub struct CountingViewer {
    pub updates: usize,
    pub last_triangles: usize,
}

impl Viewer for CountingViewer {
    fn log_mesh(&mut self, mesh: &Mesh) -> anyhow::Result<()> {
        self.updates += 1;
        self.last_triangles = mesh.triangle_count();
        Ok(())
    }
}
