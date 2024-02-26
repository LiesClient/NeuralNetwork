const activations = {
  lrelu: function lrelu(x) { return (x < 0) ? x * 0.05 : x },
  relu: function relu(x) { return (x < 0) ? 0 : x },
  sigmoid: function sigmoid(x) { return 1 / (1 + Math.exp(-x)) },
  atan: Math.atan,
  tanh: Math.tanh,
  sinh: Math.sinh,
  asinh: Math.asinh,
  softplus: function softplus(x) { return Math.log(1 + Math.exp(x)) },
  silu: function sili(x) { return x / (1 + Math.exp(-x)) },
  gaussian: function gaussian(x) { return Math.exp(-x * x) },
  elu: function elu(x) { return (x < 0) ? Math.expm1(x) : x },
}

const derivatives = {
  lrelu: x => (x < 0) ? 0.05 : 1,
  relu: x => (x < 0) ? 0 : 1,
  sigmoid: x => Math.exp(x) / (Math.exp(x) ** 2),
  atan: x => 1 / (x * x + 1),
  tanh: x => 4 / ((Math.exp(-x) + Math.exp(x)) ** 2),
  sinh: Math.cosh,
  asinh: x => 1 / ((x * x + 1) ** 2),
  softplus: x => activations.sigmoid(x),
  silu: x => (x - 1) / (Math.exp(x) + 1),
  gaussian: x => -2 * x * activations.gaussian(x),
  elu: x => (x < 0) ? Math.exp(x) : 1,
}

const colors = [
  "red", // lrelu
  "green", // relu
  "blue", // sigmoid
  "yellow", // atan
  "cyan", // tanh
  "indigo", // sinh
  "maroon", // asinh
  "gray", // softplus
  "brown", // silu
  "gold", // gaussian
  "orange", // elu
];

/* 
lrelu
relu
sigmoid
atan
atanh
tanh
sinh
asinh
step
softplus
silu
gaussian
elu
*/

if(typeof module != 'undefined') module.exports = { activations, derivatives };
