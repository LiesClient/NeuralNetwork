const canvas = document.getElementById("display"); canvas.height = window.innerHeight, canvas.width = window.innerWidth * .7;
const w = canvas.width, h = canvas.height / 2;
const logs = document.getElementById("log");
const ctx = canvas.getContext("2d");
const activationFunction = "sigmoid";
let net = new Network([2, 3, 2], activations[activationFunction], derivatives[activationFunction]);
let iter = 0;
let lr = -0.5;
let data = [];
let t0 = performance.now();
let iters = 1000;
let looping = true;

init();

function init() {
  for (let i = 0; i < 1000; i++) {
    let x = Math.random() * 2 - 1;
    let y = Math.random() * 2 - 1;
    let c1 = x > 0;
    let c2 = 1 - c1;

    data.push({
      input: [x, y],
      output: [c1, c2]
    });
  }

  ctx.fillStyle = "white";
  ctx.strokeStyle = "gray";
  ctx.lineWidth = 2;

  renderNetwork(net);
  renderActivation(net.activation);

  ctx.font = `16px monospace`;
  ctx.fillText("Learn Rate: " + -lr, 0, 16);


  line(0, h, w, h);

  // // render activations by color
  let i = 0;
  for (let act of Object.values(activations)) {
    renderActivation(act, colors[i++]);
  }

  renderDatapoints();

  console.time(`${iters} iter bench`);
  t0 = performance.now();

  document.addEventListener("keydown", (e) => (e.key.toLowerCase() == "w") ? looping = !looping : null);
  
  loop();
}

function loop() {
  if (!looping) return requestAnimationFrame(loop);
  console.time(`Iteration ${++iter}/${iters}: `);

  let cmp = false;
  
  try {
    log(`Iteration ${iter}/${iters}: ${net.train(data, lr)}`);
  } catch (e) {
    if (e.startsWith('t')) {
      log(`Training complete.`);
      cmp = true;
    }
  }

  console.timeEnd(`Iteration ${iter}/${iters}: `);
  renderDatapoints();

  if (iter >= iters || cmp) {
    let end = performance.now();
    console.timeEnd(`${iters} iter bench`);
    log(`${iter} iter bench took ${end - t0} ms`);
    log(`Avg iteration time ${(end - t0) / iter} ms`);
    return;
  }

  setTimeout(loop, 0);
}

function renderDatapoints() {
  ctx.strokeStyle = "white";
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 1.5, w / 4, h / 4, 0, 0, Math.PI * 2);
  ctx.stroke();

  for (let i = 0; i < data.length; i++) {
    let x = data[i].input[0];
    let y = data[i].input[1];
    let origRaw = data[i].output;
    let orig = origRaw.indexOf(1);
    let calcRaw = net.run(data[i].input).map(Math.round);
    let calc = calcRaw.indexOf(1);
    let actualX = ((x + 1) / 2) * w;
    let actualY = h + ((y + 1) / 2) * h;

    // coloring data based on class
    if (orig[0]) ctx.fillStyle = "purple";
    else ctx.fillStyle = "yellow";
    circle(actualX, actualY, 3);

    // coloring data based on accuracy
    if (calc == orig) ctx.fillStyle = "red";
    else ctx.fillStyle = "green";
    circle(actualX, actualY, 2)
  }

  ctx.globalAlpha = 1;
}

function testDerivative(funct, deriv) {
  console.log({ fName, brute, calc, diff });
}

function renderActivation(func = (x) => x, color = "lightgray") {
  const unitWidth = 10;
  const unitHeight = 2;

  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "lightgray";

  line(w / 2, 0, w / 2, h);
  line(0, h / 2, w, h / 2);

  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;

  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    let y = func(unitWidth * ((x / w) * 2 - 1));
    ctx.lineTo(x, (h / 2) + (h / 2) * (-y / unitHeight));
  }
  ctx.stroke();
}

function renderNetwork(net, /* input */) {
  // let curr = net.layers[0].calculate(curr); 
  let shape = net.shape;
  let sx = w / (shape.length + 1);

  for (let i = 0; i < shape[0]; i++) {
    let sy = h / (shape[0] + 1);
    circle(sx, sy * (i + 1), 12);
  }

  for (let i = 1; i < shape.length; i++) {
    // curr = net.layers[i].calculate(curr);
    let lsy = h / (shape[i - 1] + 1);
    let sy = h / (shape[i] + 1);
    for (let j = 0; j < shape[i]; j++) {
      circle(sx * (i + 1), sy * (j + 1), 12);
      for (let k = 0; k < shape[i - 1]; k++)
        line(sx * (i + 1), sy * (j + 1), sx * i, lsy * (k + 1), 12);
    }
  }
}

function circle(x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

function line(x1, y1, x2, y2, d = 0) {
  let ang = Math.atan2(y2 - y1, x2 - x1);

  ctx.beginPath();
  ctx.moveTo(Math.round(x1 + d * Math.cos(ang)), Math.round(y1 + d * Math.sin(ang)));
  ctx.lineTo(Math.round(x2 - d * Math.cos(ang)), Math.round(y2 - d * Math.sin(ang)));
  ctx.stroke();
}

function log(...inpRaw) {
  let text = document.createElement("p");
  let cont = inpRaw.map(format).join(",");
  if (cont.startsWith('"') && cont.endsWith('"'))
    cont = cont.substring(1, cont.length - 1);
  text.textContent = cont;
  logs.appendChild(text);
  logs.scrollTop = logs.scrollHeight;
}

function format(inp, ctabs = 0) {
  let first = true;
  let out = "";
  let c = ctabs;
  let s = " ";
  let tabs = (n) => s.repeat(n);
  let prim = (val) => (typeof val != "object" && typeof val != "function");

  if (prim(inp) || Array.isArray(inp)) return tabs(c) + JSON.stringify(inp);

  function formtarr(val, tbs) {
    let out = "";
    if (Array.isArray(val[0])) {
      out += "[\n";
      val.forEach(el => {
        out += tabs(1) + formtarr(el, tbs + 1) + "\n";
      });
      out += "],";
    } else out += JSON.stringify(val, null) + ",";
    return out.split("\n")
      .map((v) => (str.replaceAll(" ") == "") ? "" : tabs(c) + v)
      .join("\n");
  }

  for (let key in inp) {
    if (!first) out += "\n";
    else first = false
    out += `${key}:`;

    let val = inp[key];
    if (prim(val)) out += " " + JSON.stringify(val) + ",";
    else if (Array.isArray(val) && Array.isArray(val[0])) out += "\n" + formtarr(val, c);
    else if (Array.isArray(val)) out += formtarr(val, 0);
    else out += " {\n" + format(val, c + 1) + "},";
  }

  return (out + "\n").split("\n")
    .map((v) => (v.replaceAll(" ") == "") ? "" : tabs(c) + v)
    .join("\n");
}

function dist(p1, p2) {
  let dx = p2.x - p1.x, dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function diff(a, b) {
  let delta = 0;

  for (let i = 0; i < a.length; i++)
    delta += (b[i] - a[i]) ** 2;

  return delta;
}
