const rand = (mn, mx) => Math.random() * (mx - mn) + mn;
const rn = () => Math.random() * 2 - 1;
// const rn = () => 0;

class Layer {
  inputLength = 1;
  outputLength = 1;

  weights = [];
  biases = [];

  gradientW = [];
  gradientB = [];

  activation = function fn(x) { return x };
  derivative = (x) => 1;

  constructor(inputLength = this.inputLength, outputLength = this.outputLength, activation = this.activation, derivative = this.derivative) {
    this.inputLength = inputLength;
    this.outputLength = outputLength;
    this.activation = activation;
    this.derivative = derivative;

    this.weights = new Array(inputLength * outputLength).fill(1)
      .map(rn);
    this.gradientW = new Array(inputLength * outputLength).fill(0)
      .map(rn);
    this.biases = new Array(outputLength).fill(0)
      .map(rn);
    this.gradientB = new Array(outputLength).fill(0)
      .map(rn);
  }

  calculate(inputs = []) {
    let activations = new Array(this.outputLength);

    for (let nodeOut = 0; nodeOut < this.outputLength; nodeOut++) {
      let weightedInput = this.biases[nodeOut];

      for (let nodeIn = 0; nodeIn < this.inputLength; nodeIn++) {
        weightedInput += inputs[nodeIn] * this.weights[nodeOut * this.inputLength + nodeIn];
      }

      activations[nodeOut] = this.activation(weightedInput);
    }

    return activations;
  }

  saveValues(inputs = []) {
    let weightedInputs = new Array(this.outputLength);
    let activations = new Array(this.outputLength);

    for (let nodeOut = 0; nodeOut < this.outputLength; nodeOut++) {
      let weightedInput = this.biases[nodeOut];

      for (let nodeIn = 0; nodeIn < this.inputLength; nodeIn++) {
        weightedInput += inputs[nodeIn] * this.weights[nodeOut * this.inputLength + nodeIn];
      }

      weightedInputs[nodeOut] = weightedInput;
      activations[nodeOut] = this.activation(weightedInput);
    }

    return { inputs: inputs, weighted: weightedInputs, activations: activations }
  }

  outputLayerNodeValues(data, expected, costDerivative) {
    let nodeValues = new Array(this.outputLength);
    for (let i = 0; i < nodeValues.length; i++) {
      let costDeriv = costDerivative(data.activations[i], expected[i]);
      let actDeriv = this.derivative(data.weighted[i]);
      nodeValues[i] = costDeriv * actDeriv;
    }
    return nodeValues;
  }

  hiddenLayerNodeValues(data, weights, oldValues) {
    let nodeValues = new Array(this.outputLength).fill(0);

    for (let i = 0; i < this.outputLength; i++) {
      let nodeValue = 0;

      for (let j = 0; j < oldValues.length; j++)
        nodeValue += weights[j * this.outputLength + i] * oldValues[j];

      nodeValues[i] = nodeValue * this.derivative(data.weighted[i]);
    }

    return nodeValues;
  }

  applyGradients(learnRate = 0.01) {
    for (let i = 0; i < this.weights.length; i++) {
      let delta = this.gradientW[i] * learnRate;
      this.weights[i] += delta;
    }

    for (let i = 0; i < this.biases.length; i++) {
      let delta = this.gradientB[i] * learnRate;
      this.biases[i] += delta;
    }
  }

  updateGradients(inputs = [], values = []) {
    // weight gradient updates
    for (let i = 0; i < this.outputLength; i++) {
      let value = values[i];

      for (let j = 0; j < this.inputLength; j++) {
        let partialDeriv = inputs[j] * value;
        this.gradientW[i * this.inputLength + j] -= partialDeriv;
      }

      this.gradientB[i] -= value;
    }
  }

  mutate(scale) {
    let newLayer = Layer.clone(this);

    for (let i = 0; i < newLayer.weights.length; i++)
      newLayer.weights[i] += Math.random() * scale * 2 - scale;

    for (let i = 0; i < newLayer.biases.length; i++)
      newLayer.biases[i] += Math.random() * scale * 2 - scale;
    
    return newLayer;
  }

  static clone(layer) {
    let newLayer = new Layer(layer.inputLength, layer.outputLength, layer.activation, layer.derivative);
    
    for (let i = 0; i < newLayer.weights.length; i++)
      newLayer.weights[i] = layer.weights[i];

    for (let i = 0; i < newLayer.biases.length; i++)
      newLayer.biases[i] = layer.biases[i];

    return newLayer;
  }

  compile(gpu, f32out = false) {
    /// todo: this
    return (gpu.createKernel((input) => {
      
    })
      .setPipeline(!f32out)
      .setOutput(this.outputLength));
  }
}

class Network {
  layers = [];
  shape = [1, 1];

  activation = (x) => x;
  derivative = (x) => 1;

  constructor(shape = this.shape, activation = this.activation, derivative = this.derivative) {
    this.shape = shape;
    this.layers = new Array(shape.length - 1);
    this.activation = activation;
    this.derivative = derivative;

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i] = new Layer(shape[i], shape[i + 1], activation, derivative);
    }
  }

  run(inputs) {
    for (let i = 0; i < this.layers.length; i++) {
      inputs = this.layers[i].calculate(inputs);
    }

    return inputs;
  }

  train(data, learnRate = 0.01) {
    if (!data.length) return;

    let t0 = performance.now();

    for (let i = 0; i < data.length; i++) {
      this.updateGradients(data[i]);
    }

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].applyGradients(learnRate / data.length);
    }

    let error = 0;

    for (let i = 0; i < this.layers.length; i++) {
      let layer = this.layers[i];
      
      for (let j = 0; j < layer.gradientW.length; j++) {
        error += layer.gradientW[j];
        layer.gradientW[j] = 0;
      }

      for (let j = 0; j < layer.gradientB.length; j++) {
        error += layer.gradientB[j];
        layer.gradientB[j] = 0;
      }
    }

    if (error < 0) throw "t : Training Complete. Catch this error with .startsWith('t')"
    if (Math.abs(error) >= 2 ** 32) throw "Network has gone off the rails... restart or stop training.";
    if (isNaN(error)) 
      throw "NaN err... restart training. Remember to stop training after accuracy demands are met.";

    let t1 = performance.now();

    return `t = ${(t1 - t0).toFixed(2)}ms, d = ${-error}`;
  }

  updateGradients(datapoint) {
    let inputsToNextLayer = datapoint.input;
    let learnData = new Array(this.layers.length);
    let totalError = 0;

    for (let i = 0; i < this.layers.length; i++) {
      let layerData = this.layers[i].saveValues(inputsToNextLayer);
      learnData[i] = layerData;
      inputsToNextLayer = layerData.activations;
    }

    let outputLayerIndex = this.layers.length - 1;
    let outputLayer = this.layers[outputLayerIndex];
    let outputLearnData = learnData[outputLayerIndex];

    let outputNodeValues = outputLayer.outputLayerNodeValues(outputLearnData, datapoint.output, this.costDeriv);
    outputLayer.updateGradients(outputLearnData.inputs, outputLearnData.activations);
    learnData[outputLayerIndex].nodeValues = outputNodeValues;

    for (let i = outputLayerIndex - 1; i >= 0; i--) {
      let layerLearnData = learnData[i];
      let hiddenLayer = this.layers[i];

      let nodeValues = hiddenLayer.hiddenLayerNodeValues(layerLearnData, this.layers[i + 1].weights, learnData[i + 1].nodeValues);
      hiddenLayer.updateGradients(layerLearnData.inputs, nodeValues);

      learnData[i].nodeValues = nodeValues;
    }

    return totalError;
  }

  cost(a, y) {
    let e = a - y;
    return e * e;
  }

  costDeriv(a, y) {
    return 2 * (a - y);
  }

  mutate (scale) {
    let newNetwork = new Network(this.shape, this.activation, this.derivative);

    for(let i = 0; i < newNetwork.layers.length; i++)
      newNetwork.layers[i] = this.layers[i].mutate(scale);

    return newNetwork;
  }

  static clone (network) {
    let newNetwork = new Network(network.shape, network.activation, network.derivative);

    for(let i = 0; i < newNetwork.layers.length; i++)
      newNetwork.layers[i] = Layer.clone(network.layers[i]);

    return newNetwork;
  }

  compile (gpu) {
    let kernels = new Array(this.layers.length);

    for(let i = 0; i < kernels.length - 1; i++)
      kernels[i] = this.layers[i].compile(gpu);

    kernels[kernels.length - 1] = this.layers[i].compile(gpu, true);

    return (inputs) => {
      let currTexture = inputs;
      for(let i = 0; i < kernels.length; i++)
        currTexture = kernels[i](inputs);
      return currTexture;
    }
  }
}

if (typeof module != 'undefined') module.exports = { Layer, Network };
