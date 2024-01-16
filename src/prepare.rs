use js_sys::{Object, Reflect};
use rusvm::kernel::Kernel;
use wasm_bindgen::prelude::*;

use super::getters::*;

pub fn extract_params_smo(params: &Object) -> (rusvm::smo::Params, usize) {
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

pub fn extract_params_newton(params: &Object) -> (rusvm::newton::Params, usize) {
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

pub fn extract_params_problem(params: &Object) -> rusvm::problem::Params {
    let mut p = rusvm::problem::Params::new();
    if params.is_object() {
        p.lambda = get_nonan(params, "lmbda", p.lambda);
        p.smoothing = get_nonan(params, "smoothing", p.smoothing);
        p.max_asum = get_nonan(params, "max_asum", p.max_asum);
        p.regularization = get_nonan(params, "regularization", p.regularization);
    }
    p
}

pub fn prepare_problem<'a>(
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

pub fn create_kernel<'a>(data: &'a Vec<Vec<f64>>, gamma: f64) -> Box<impl Kernel + 'a> {
    Box::new(rusvm::kernel::gaussian::from_vecs(
        data.iter().map(|v| v.as_slice()).collect::<Vec<&[f64]>>(),
        gamma,
    ))
}
