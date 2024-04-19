# Use WASM to get Rust-powered SVM training to the browser

## Usage
```
wasm-pack build --target web
cd example
ln -s ../pkg .
python3 -m http.server
```

The example is also available [online](https://rusvm-wasm-example.surge.sh/).