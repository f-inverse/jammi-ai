use pyo3::prelude::*;

#[pymodule]
fn jammi_python(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
