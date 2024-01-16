function generateData (n, seed) {
  Math.seedrandom(seed)
  let x = Array.from(Array(n).keys(), _ => {
    let xi = Math.random()
    return [xi]
  })
  let y = x.map(xi => {
    return Math.sin(2.0 * 3.0 * Math.PI * xi[0])
  })
  return { x, y }
}

function linspace (a, b, n) {
  return Array.from(Array(n).keys(), i => {
    return a + ((b - a) / (n - 1)) * i
  })
}
