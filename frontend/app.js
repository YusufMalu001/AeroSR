// DOM Elements
const imageSelector = document.getElementById('image-selector');
const processBtn = document.getElementById('process-btn');
const noiseSlider = document.getElementById('noise-slider');
const blurSlider = document.getElementById('blur-slider');
const hazeSlider = document.getElementById('haze-slider');
const toggleBoxes = document.getElementById('toggle-boxes');

const noiseVal = document.getElementById('noise-val');
const blurVal = document.getElementById('blur-val');
const hazeVal = document.getElementById('haze-val');

// Chart instances
let barChart = null;
let lossChart = null;
let currentResults = null;

// Initialize
async function init() {
    // Sliders event listeners
    noiseSlider.addEventListener('input', (e) => noiseVal.textContent = e.target.value);
    blurSlider.addEventListener('input', (e) => blurVal.textContent = e.target.value);
    hazeSlider.addEventListener('input', (e) => hazeVal.textContent = e.target.value);
    
    toggleBoxes.addEventListener('change', drawAllBoxes);
    processBtn.addEventListener('click', processImage);

    await fetchImages();
    await fetchBenchmarks();
}

async function fetchImages() {
    try {
        const response = await fetch('/api/images');
        const images = await response.json();
        
        imageSelector.innerHTML = '';
        if (images.length === 0) {
            imageSelector.innerHTML = '<option value="">No images available</option>';
            return;
        }
        
        images.forEach(img => {
            const opt = document.createElement('option');
            opt.value = img;
            opt.textContent = img;
            imageSelector.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to load images:", err);
    }
}

function setLoaders(show) {
    const display = show ? 'block' : 'none';
    document.getElementById('loader-orig').style.display = display;
    document.getElementById('loader-deg').style.display = display;
    document.getElementById('loader-bicubic').style.display = display;
    document.getElementById('loader-sr').style.display = display;
    processBtn.disabled = show;
}

async function processImage() {
    const imageName = imageSelector.value;
    if (!imageName) return;

    setLoaders(true);
    
    const reqData = {
        image_name: imageName,
        noise_sigma: parseFloat(noiseSlider.value),
        blur_kernel: parseInt(blurSlider.value),
        haze_t: parseFloat(hazeSlider.value)
    };

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData)
        });
        
        currentResults = await response.json();
        
        updateView('orig', currentResults.original);
        updateView('deg', currentResults.degraded);
        updateView('bicubic', currentResults.bicubic);
        updateView('sr', currentResults.sr);
        
    } catch (err) {
        console.error("Failed to process image:", err);
        alert("Error processing image. Check console.");
    } finally {
        setLoaders(false);
    }
}

function updateView(idPrefix, data) {
    const imgEl = document.getElementById(`img-${idPrefix}`);
    imgEl.src = `data:image/jpeg;base64,${data.image}`;
    
    // Wait for image to render to get proper dims
    imgEl.onload = () => {
        drawBoxes(idPrefix, data.boxes);
    };
    
    document.getElementById(`stats-${idPrefix}`).textContent = `Detections: ${data.boxes.length}`;
}

function drawAllBoxes() {
    if (!currentResults) return;
    drawBoxes('orig', currentResults.original.boxes);
    drawBoxes('deg', currentResults.degraded.boxes);
    drawBoxes('bicubic', currentResults.bicubic.boxes);
    drawBoxes('sr', currentResults.sr.boxes);
}

function drawBoxes(idPrefix, boxes) {
    const canvas = document.getElementById(`canvas-${idPrefix}`);
    const img = document.getElementById(`img-${idPrefix}`);
    
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!toggleBoxes.checked) return;
    
    // Scale factors because naturalImage might be 800x800 but displayed at 300x300
    // Actually, bounding boxes returned from YOLO are relative to the 800x800 image size
    // So we need scale factors
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;

    boxes.forEach(box => {
        const x = box.x1 * scaleX;
        const y = box.y1 * scaleY;
        const w = (box.x2 - box.x1) * scaleX;
        const h = (box.y2 - box.y1) * scaleY;
        
        ctx.strokeStyle = '#f59e0b'; // box-color
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        // Label
        ctx.fillStyle = '#f59e0b';
        const label = `${box.class} ${(box.conf*100).toFixed(0)}%`;
        ctx.font = '12px Outfit';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillRect(x, y > 15 ? y - 15 : y, textWidth + 4, 15);
        ctx.fillStyle = '#000';
        ctx.fillText(label, x + 2, y > 15 ? y - 3 : y + 12);
    });
}

async function fetchBenchmarks() {
    try {
        const response = await fetch('/api/benchmarks');
        const data = await response.json();
        
        renderBarChart(data.benchmarks);
        renderLossChart(data.loss_history);
    } catch (err) {
        console.error("Failed to load benchmarks:", err);
    }
}

function renderBarChart(benchmarks) {
    const ctx = document.getElementById('barChart').getContext('2d');
    
    // Default empty if benchmarks aren't ready
    const labels = ['Original HR', 'Degraded LR', 'Bicubic', 'SR Enhanced'];
    let map50 = [0, 0, 0, 0];
    
    if (Object.keys(benchmarks).length > 0) {
        map50 = [
            benchmarks['original']?.mAP50 || 0,
            benchmarks['degraded']?.mAP50 || 0,
            benchmarks['bicubic']?.mAP50 || 0,
            benchmarks['sr_enhanced']?.mAP50 || 0
        ];
    }
    
    if (barChart) barChart.destroy();
    
    barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'mAP@50',
                data: map50,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.6)',
                    'rgba(239, 68, 68, 0.6)',
                    'rgba(245, 158, 11, 0.6)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(16, 185, 129, 1)'
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'YOLOv8 Detection Accuracy (mAP@50)', color: '#f8fafc' },
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderLossChart(lossData) {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    const labels = lossData.map(d => `Ep ${d.epoch}`);
    const data = lossData.map(d => d.loss);
    
    if (lossChart) lossChart.destroy();
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Training Loss (L1)',
                data: data,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'ESPCN Model Training Loss', color: '#f8fafc' },
                legend: { display: false }
            },
            scales: {
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 45 },
                    grid: { display: false }
                }
            }
        }
    });
}

// Start app
init();
window.addEventListener('resize', drawAllBoxes);
