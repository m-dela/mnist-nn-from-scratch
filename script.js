// ============================================================================
// 1. CANVAS SETUP & DRAWING
// ============================================================================
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');

// Set pen properties
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000';

// Drawing state variables
let isDrawing = false;
let lastX = 0;
let lastY = 0;

function getCoordinates(e) {
  const rect = canvas.getBoundingClientRect();
  // Calculate coordinate relative to the entire screen (finger or mouse)
  const clientX = e.touches ? e.touches[0].clientX : e.clientX; 
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;

  // Return coordinates relative to the canvas
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

function startDrawing(e) {
  if (e.type.includes('touch')) {
    e.preventDefault();
  }

  isDrawing = true;
  const { x, y } = getCoordinates(e);
  lastX = x;
  lastY = y;

  ctx.beginPath();
  ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}

function draw(e) {
  if (!isDrawing) return;
  e.preventDefault();

  const { x, y } = getCoordinates(e);

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  lastX = x;
  lastY = y;
}

function stopDrawing() {
  isDrawing = false;
}

// Attach Mouse Events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// Attach Touch Events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);


// ============================================================================
// 2. IMAGE PROCESSING (PREPARING THE DRAWING FOR THE AI)
// ============================================================================

// Step A: Get minimal box in which we have drawn the number
function getBoundingBox() {
  const imgData = ctx.getImageData(0, 0, 280, 280);
  const pixels = imgData.data; // 1D array with [R, G, B, A]

  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y++){
    for (let x = 0; x < canvas.width; x++){
      const index = (y * canvas.width + x) * 4;
      const alpha = pixels[index + 3];

      if (alpha > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX === -1) return null; // No drawing exists

  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  };
}

// Step B: Crop digit and scale to 20x20
function cropAndScaleTo20x20(sourceCanvas, bbox) {
  if (!bbox) return null;

  const targetCanvas = document.createElement('canvas');
  targetCanvas.width = 28;
  targetCanvas.height = 28;
  const targetCtx = targetCanvas.getContext('2d');

  targetCtx.imageSmoothingEnabled = true;
  targetCtx.imageSmoothingQuality = 'medium';

  const maxDim = Math.max(bbox.width, bbox.height);
  const scale = 20 / maxDim;

  const scaledW = bbox.width * scale;
  const scaledH = bbox.height * scale;
  const dX = (28 - scaledW) / 2;
  const dY = (28 - scaledH) / 2;

  targetCtx.drawImage(
    sourceCanvas,
    bbox.x, bbox.y, bbox.width, bbox.height, 
    dX, dY, scaledW, scaledH
  );

  return targetCanvas;
}

// Step C: Calculate center of mass and move digit
function centerByCenterOfMass(targetCanvas) {
  const ctx = targetCanvas.getContext('2d');
  const width = targetCanvas.width;
  const height = targetCanvas.height;
  const data = ctx.getImageData(0, 0, width, height).data;

  let totalMass = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alphaIndex = (y * width + x) * 4;
      const mass = data[alphaIndex + 3];

      if (mass > 0) {
        totalMass += mass;
        sumX += x * mass;
        sumY += y * mass;
      }
    }
  }

  if (totalMass === 0) return targetCanvas;

  const cX = sumX / totalMass;
  const cY = sumY / totalMass;
  const shiftX = Math.round(13.5 - cX);
  const shiftY = Math.round(13.5 - cY);

  const centeredCanvas = document.createElement('canvas');
  centeredCanvas.width = 28;
  centeredCanvas.height = 28;
  centeredCanvas.getContext('2d').drawImage(targetCanvas, shiftX, shiftY);

  return centeredCanvas;
}

// Step D: Transform the canvas into a normalized vector for the NN
function canvasToNormalizedVector(centeredCanvas) {
  const ctx = centeredCanvas.getContext('2d');
  const data = ctx.getImageData(0, 0, 28, 28).data;
  const inputVector = new Float32Array(784);

  for (let i = 0; i < 784; i++) {
    const alpha = data[i * 4 + 3];
    inputVector[i] = alpha / 255.0;
  }

  return inputVector;
}


// ============================================================================
// 3. NEURAL NETWORK (THE AI BRAIN)
// ============================================================================
let model = null;

// Download weights and biases
async function loadModel() {
  try {
    console.log("Loading weights.json...");
    const response = await fetch('weights.json');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    return {
      w1: new Float32Array(data.w1),             
      b1: new Float32Array(data.b1),             
      w2: new Float32Array(data.w2),             
      b2: new Float32Array(data.b2),             
      hiddenSize: data.hidden_size               
    };

  } catch (error) {
    console.error("Failed to load weights.json:", error);
    return null;
  }
}

// Start downloading model as soon as script runs
async function initApp() {
  if (predictBtn) predictBtn.disabled = true;

  model = await loadModel();

  if (model) {
    console.log("Model successfully loaded!");
    if (predictBtn) predictBtn.disabled = false;
  }
}
initApp();

// Calculate probabilities using Math library
function forwardPass(x, model) {
  if (!model) return null;

  const { w1, b1, w2, b2, hiddenSize } = model;
  const inputSize = 784;
  const outputSize = 10;

  let a1 = new Float32Array(hiddenSize);
  let logits = new Float32Array(outputSize);

  // Layer 1: Linear + ReLU
  for (let i = 0; i < hiddenSize; i++) {
    let sum = b1[i];
    for (let j = 0; j < inputSize; j++) {
      sum += x[j] * w1[j * hiddenSize + i];
    }
    a1[i] = Math.max(0, sum);
  }

  // Layer 2: Linear
  for (let i = 0; i < outputSize; i++) {
    let sum = b2[i];
    for (let j = 0; j < hiddenSize; j++) {
      sum += a1[j] * w2[j * outputSize + i]; 
    }
    logits[i] = sum;
  }
  
  // Layer 3: Softmax
  let maxLogit = logits[0];
  for (let i = 1; i < outputSize; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const probs = new Float32Array(outputSize);
  let sumExp = 0;
  
  for (let i = 0; i < outputSize; i++) {
    probs[i] = Math.exp(logits[i] - maxLogit);
    sumExp += probs[i];
  }
  
  for (let j = 0; j < outputSize; j++) {
    probs[j] /= sumExp;
  }

  return probs;
}


// ============================================================================
// 4. BUTTON CLICK EVENTS
// ============================================================================

// CLEAR BUTTON
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i <= 9; i++) {
    const bar = document.getElementById(`bar-${i}`);
    const prob = document.getElementById(`prob-${i}`);
    if (bar) bar.style.width = '0%';
    if (prob) prob.textContent = '0.0%';
  }
  
  const existingPreview = document.getElementById('debug-preview');
  if (existingPreview) existingPreview.remove();
});


// PREDICT BUTTON
predictBtn.addEventListener('click', () => {
  const bbox = getBoundingBox();
  if (!bbox) {
    console.warn('Canvas is blank!');
    return;
  }

  // Chain the image processing steps
  const scaledCanvas = cropAndScaleTo20x20(canvas, bbox);
  const finalCanvas = centerByCenterOfMass(scaledCanvas);
  const inputVector = canvasToNormalizedVector(finalCanvas);

  // Run the math prediction
  const probabilities = forwardPass(inputVector, model);
  if (probabilities) {
    console.log(probabilities);
  }

  updateProababilityChart(probabilities);

  // Visual Verification preview
  const existingPreview = document.getElementById('debug-preview');
  if (existingPreview) existingPreview.remove();

  finalCanvas.id = 'debug-preview';
  finalCanvas.style.width = '140px';
  finalCanvas.style.height = '140px';
  finalCanvas.style.border = '2px dashed #0969da';
  finalCanvas.style.marginTop = '16px';
  finalCanvas.style.display = 'block';
  finalCanvas.style.marginLeft = 'auto';
  finalCanvas.style.marginRight = 'auto';
  finalCanvas.style.imageRendering = 'pixelated';

  document.querySelector('.canvas-card').appendChild(finalCanvas);
});

function updateProababilityChart(probabilities) {
  if (!probabilities || probabilities.length !==10) return;

  // Calculate highest probability
  let bestDigit = 0;
  let maxProb = probabilities[0];

  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      bestDigit = i;
    }
  }

  //Update chart
  for (let i = 0; i <= 9; i++) {
    const prob = probabilities[i];
    const percentage = (prob * 100).toFixed(1); //1 decimal place

    const bar = document.getElementById(`bar-${i}`);
    const label = document.getElementById(`prob-${i}`);
    if (bar) bar.style.width = `${percentage}%`;
    if (label) label.textContent = `${percentage}%`;
  }
}