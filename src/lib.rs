use console_error_panic_hook;
use js_sys::{Array, Object};
use rusvm::kernel::cache;
use std::panic;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(value: &str);
}

// macro_rules! console_log {
//     ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
// }

mod getters;
use getters::*;
mod prepare;
use prepare::*;

#[wasm_bindgen]
pub fn smo(
    x: &Array,
    y: &[f64],
    params_problem: &Object,
    params_smo: &Object,
) -> Result<JsValue, serde_wasm_bindgen::Error> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    // get parameters
    let (params_smo, cache_size) = extract_params_smo(params_smo);
    // prepare problem
    let problem = prepare_problem(&y, params_problem).unwrap();
    // prepare kernel
    let data = extract_data(x);
    let gamma = get_nonan(params_problem, "gamma", 1.0);

    let base = Box::new(rusvm::kernel::gaussian_from_vecs(
        data.iter().map(|v| v.as_slice()).collect::<Vec<&[f64]>>(),
        gamma,
    ));
    let mut kernel = cache(base, cache_size);
    let result = rusvm::smo::solve(problem.as_ref(), kernel.as_mut(), &params_smo, None);
    let (result_sv, svs) = result.find_support(&data);
    serde_wasm_bindgen::to_value(&(result.opt_status, result_sv, svs))
}

#[wasm_bindgen]
pub fn newton(
    x: &Array,
    y: &[f64],
    params_problem: &Object,
    params_newton: &Object,
) -> Result<JsValue, serde_wasm_bindgen::Error> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    // get parameters
    let (params_newton, cache_size) = extract_params_newton(params_newton);
    // prepare problem
    let problem = prepare_problem(&y, params_problem).unwrap();
    // prepare kernel
    let data = extract_data(x);
    let gamma = get_nonan(params_problem, "gamma", 1.0);

    let base = Box::new(rusvm::kernel::gaussian_from_vecs(
        data.iter().map(|v| v.as_slice()).collect::<Vec<&[f64]>>(),
        gamma,
    ));
    let mut kernel = cache(base, cache_size);
    let result = rusvm::newton::solve(problem.as_ref(), kernel.as_mut(), &params_newton, None);
    let (result_sv, svs) = result.find_support(&data);
    serde_wasm_bindgen::to_value(&(result.opt_status, result_sv, svs))
}
