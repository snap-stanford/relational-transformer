use pyo3::prelude::*;

mod common;
pub mod fly;

#[pymodule]
fn rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;

    Ok(())
}
