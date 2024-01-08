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

function kernel (x0, x1, gamma) {
  let dsqr = 0.0
  for (let i = 0; i < x0.length; ++i) {
    let di = x0[i] - x1[i]
    dsqr += di * di
  }
  return Math.exp(-gamma * dsqr)
}

function linspace (a, b, n) {
  return Array.from(Array(n).keys(), i => {
    return a + ((b - a) / (n - 1)) * i
  })
}

function decisionFunction (xk, status, svs, params) {
  const { lmbda, gamma } = params
  let d = 0.0
  for (let i = 0; i < svs.length; ++i) {
    d += status.a[i] * kernel(svs[i], xk, gamma)
  }
  return d / lmbda + status.b
}
