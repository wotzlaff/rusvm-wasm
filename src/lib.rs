use js_sys::{Array, Object, Reflect};
use rusvm::kernel::cache;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(value: &str);
}

// macro_rules! console_log {
//     ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
// }

use console_error_panic_hook;
use std::panic;

mod getters;
use getters::*;

fn extract_params_smo(params: &Object) -> (rusvm::smo::Params, usize) {
    let mut p = rusvm::smo::Params::new();
    let cache_size: usize = if params.is_object() {
        p.tol = get_nonan(params, "tol", p.tol);
        p.max_steps = get_usize(params, "max_steps", p.max_steps);
        p.verbose = get_usize(params, "verbose", p.verbose);
        p.log_objective = get_bool(params, "log_objective", p.log_objective);
        p.second_order = get_bool(params, "second_order", p.second_order);
        p.shrinking_period = get_usize(params, "shrinking_period", p.shrinking_period);
        p.shrinking_threshold = get_nonan(params, "shrinking_threshold", p.shrinking_threshold);
        p.time_limit = get_nonan(params, "time_limit", p.time_limit);
        get_usize(params, "cache_size", 0)
    } else {
        0
    };
    (p, cache_size)
}

fn extract_params_newton(params: &Object) -> (rusvm::newton::Params, usize) {
    let mut p = rusvm::newton::Params::new();
    let cache_size: usize = if params.is_object() {
        p.tol = get_nonan(params, "tol", p.tol);
        p.max_steps = get_usize(params, "max_steps", p.max_steps);
        p.verbose = get_usize(params, "verbose", p.verbose);
        p.time_limit = get_nonan(params, "time_limit", p.time_limit);
        p.sigma = get_nonan(params, "sigma", p.sigma);
        p.eta = get_nonan(params, "eta", p.eta);
        p.max_back_steps = get_usize(params, "max_back_steps", p.max_back_steps);
        get_usize(params, "cache_size", 0)
    } else {
        0
    };
    (p, cache_size)
}

fn extract_params_problem(params: &Object) -> rusvm::problem::Params {
    let mut p = rusvm::problem::Params::new();
    if params.is_object() {
        p.lambda = get_nonan(params, "lmbda", p.lambda);
        p.smoothing = get_nonan(params, "smoothing", p.smoothing);
        p.max_asum = get_nonan(params, "max_asum", p.max_asum);
        p.regularization = get_nonan(params, "regularization", p.regularization);
    }
    p
}

fn prepare_problem<'a>(
    y: &'a &[f64],
    params: &Object,
) -> Option<Box<dyn rusvm::problem::Problem + 'a>> {
    let kind = if let Ok(kind) = Reflect::get(&params, &JsValue::from("kind")) {
        kind.as_string().unwrap()
    } else {
        String::from("classification")
    };
    match kind.as_str() {
        "classification" => {
            // check_params(
            //     params,
            //     vec!["kind", "lmbda", "smoothing", "max_asum", "shift"].as_slice(),
            // )?;
            let mut problem =
                rusvm::problem::Classification::new(y, extract_params_problem(params));
            problem.shift = get_nonan(params, "shift", problem.shift);
            Some(Box::new(problem))
        }
        "regression" => {
            // check_params(
            //     params,
            //     vec!["kind", "lmbda", "smoothing", "max_asum", "epsilon"].as_slice(),
            // )?;
            let mut problem = rusvm::problem::Regression::new(y, extract_params_problem(params));
            problem.epsilon = get_nonan(params, "epsilon", problem.epsilon);
            Some(Box::new(problem))
        }
        "lssvm" => {
            // check_params(params, vec!["kind", "lmbda"].as_slice())?;
            let problem = rusvm::problem::LSSVM::new(y, extract_params_problem(params));
            Some(Box::new(problem))
        }
        "poisson" => {
            // check_params(params, vec!["kind", "lmbda"].as_slice())?;
            let problem = rusvm::problem::Poisson::new(y, extract_params_problem(params));
            Some(Box::new(problem))
        }
        &_ => None,
    }
}

fn extract_data(x: &Array) -> Vec<Vec<f64>> {
    let mut data = Vec::new();
    let mut nft = u32::MAX;
    for i in 0..x.length() {
        let xi: Array = x.get(i).into();
        let nft_i = xi.length();
        if nft == u32::MAX {
            nft = nft_i;
        }
        assert!(nft_i == nft);
        let mut arr = Vec::with_capacity(nft.try_into().unwrap());
        for j in 0..xi.length() {
            arr.push(xi.get(j).unchecked_into_f64());
        }
        data.push(arr);
    }
    data
}

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
