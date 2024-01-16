use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;

pub fn get_nonan(params: &Object, key: &str, default: f64) -> f64 {
    let val = Reflect::get(&params, &JsValue::from(key))
        .unwrap_or(JsValue::from(default))
        .unchecked_into_f64();
    if val.is_nan() {
        default
    } else {
        val
    }
}

pub fn get_usize(params: &Object, key: &str, default: usize) -> usize {
    if let Ok(val) = Reflect::get(&params, &JsValue::from(key)) {
        let val_f64 = val.unchecked_into_f64();
        if val_f64.is_nan() {
            default
        } else {
            val_f64 as usize
        }
    } else {
        default
    }
}

pub fn get_bool(params: &Object, key: &str, default: bool) -> bool {
    if let Ok(val) = Reflect::get(&params, &JsValue::from(key)) {
        val.as_bool().unwrap_or(default)
    } else {
        default
    }
}
