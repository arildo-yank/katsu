use pyo3::prelude::*;

/// Estrutura de empacotamento. A classe Tensor será exposta na Parte 4.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
