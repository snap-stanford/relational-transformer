use pyo3::prelude::*;

mod common;
pub mod fly;
#[cfg(feature = "pre")]
mod pre;

/// Preprocess a relbench-3.0.0-layout dataset dir (parquet -> rustler's on-disk
/// rkyv format). Only built into wheels with the `pre` feature. Releases the GIL.
#[cfg(feature = "pre")]
#[pyfunction]
#[pyo3(signature = (dataset_dir, out_dir, *, source=None, skip_tasks=false, skip_db=false))]
fn preprocess(
    py: Python<'_>,
    dataset_dir: String,
    out_dir: String,
    source: Option<String>,
    skip_tasks: bool,
    skip_db: bool,
) -> PyResult<()> {
    py.allow_threads(move || {
        pre::main(pre::Cli {
            dataset_dir,
            out_dir,
            source,
            skip_tasks,
            skip_db,
        })
    });
    Ok(())
}

#[pymodule]
fn _rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;
    m.add_function(wrap_pyfunction!(fly::column_sem_types, m)?)?;
    #[cfg(feature = "pre")]
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;

    Ok(())
}
