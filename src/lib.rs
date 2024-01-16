use console_error_panic_hook;
use js_sys::Object;
use rusvm::{
    kernel::{cache, gaussian, KernelFunction},
    Status,
};
use std::panic;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(value: &str);
}

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

mod getters;
use getters::*;
mod prepare;
use prepare::*;

#[wasm_bindgen]
pub fn predict(
    status: JsValue,
    data: JsValue,
    params_problem: &Object,
    other: JsValue,
) -> Result<JsValue, serde_wasm_bindgen::Error> {
    let lmbda = get_nonan(
        params_problem,
        "lmbda",
        rusvm::problem::Params::DEFAULT_LAMBDA,
    );
    let gamma = get_nonan(params_problem, "gamma", 1.0);
    let stat = serde_wasm_bindgen::from_value::<Status>(status)?;
    let svs = serde_wasm_bindgen::from_value::<Vec<Vec<f64>>>(data)?;
    let sv_slices: Vec<_> = svs.iter().map(|xi| xi.as_slice()).collect();
    let other_data = serde_wasm_bindgen::from_value::<Vec<Vec<f64>>>(other)?;
    let kernel_function: KernelFunction<&[f64]> =
        Box::from(move |xi: &&_, xj: &&_| gaussian::kernel(xi, xj, gamma));
    let predictions: Vec<_> = other_data
        .iter()
        .map(|other_vec| {
            rusvm::predict(
                &other_vec.as_slice(),
                &sv_slices,
                &stat,
                lmbda,
                &kernel_function,
            )
        })
        .collect();
    serde_wasm_bindgen::to_value(&predictions)
}

#[wasm_bindgen]
pub fn smo(
    x: JsValue,
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
    let data = serde_wasm_bindgen::from_value::<Vec<Vec<f64>>>(x)?;
    let base = create_kernel(&data, get_nonan(params_problem, "gamma", 1.0));
    let mut kernel = cache(base, cache_size);
    let result = rusvm::smo::solve(problem.as_ref(), kernel.as_mut(), &params_smo, None);
    let (result_sv, svs) = result.find_support(&data);
    serde_wasm_bindgen::to_value(&(result.opt_status, result_sv, svs))
}

#[wasm_bindgen]
pub fn newton(
    x: JsValue,
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
    let data = serde_wasm_bindgen::from_value::<Vec<Vec<f64>>>(x)?;
    let base = create_kernel(&data, get_nonan(params_problem, "gamma", 1.0));
    let mut kernel = cache(base, cache_size);
    let result = rusvm::newton::solve(problem.as_ref(), kernel.as_mut(), &params_newton, None);
    let (result_sv, svs) = result.find_support(&data);
    serde_wasm_bindgen::to_value(&(result.opt_status, result_sv, svs))
}
