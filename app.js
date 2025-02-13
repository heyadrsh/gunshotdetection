// Audio Context and Processing Variables
let audioContext;
let mediaStream;
let analyzer;
let dataArray;
let isRecording = false;
let socket;
let probabilityChart;

// DOM Elements
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusBadge = document.getElementById('statusBadge');
const detectionHistory = document.getElementById('detectionHistory');
const gunTypeLabel = document.getElementById('gunTypeLabel');
const currentSounds = document.getElementById('currentSounds');
const micPermissionModal = document.getElementById('micPermissionModal');
const micPermissionBtn = document.getElementById('micPermissionBtn');
const audioVisualizer = document.getElementById('audioVisualizer');
const visualizerCtx = audioVisualizer.getContext('2d');

// Initialize WebSocket connection
function initializeWebSocket() {
    socket = new WebSocket('ws://localhost:8000/ws');
    
    socket.onopen = () => {
        console.log('WebSocket connection established');
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection Error', 'error');
    };
    
    socket.onclose = () => {
        console.log('WebSocket connection closed');
        if (isRecording) {
            stopRecording();
        }
    };
}

// Initialize Chart.js
function initializeChart() {
    const ctx = document.getElementById('gunProbChart').getContext('2d');
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Confidence (%)',
                data: [],
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 12
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Handle messages from the server
function handleServerMessage(data) {
    switch (data.type) {
        case 'gunshot_detected':
            handleGunshotDetection(data);
            break;
        case 'gun_type':
            updateGunType(data);
            break;
        case 'current_sounds':
            updateCurrentSounds(data.sounds);
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}

// Handle gunshot detection
function handleGunshotDetection(data) {
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'detection-entry mb-4 p-4 bg-red-50 border border-red-500 rounded-lg';
    entry.innerHTML = `
        <div class="font-bold text-red-600">[${timestamp}] !!! GUNSHOT DETECTED !!!</div>
        <div class="mt-2">Detected sounds:</div>
        ${data.detections.map(d => `<div class="ml-4">→ ${d}</div>`).join('')}
        <div class="mt-2 border-t border-red-200"></div>
    `;
    
    detectionHistory.insertBefore(entry, detectionHistory.firstChild);
    statusBadge.textContent = 'ALERT: Gunshot Detected';
    statusBadge.className = 'alert-active mt-4 px-4 py-2 rounded-lg';
    
    // Reset status after 3 seconds
    setTimeout(() => {
        if (isRecording) {
            statusBadge.textContent = 'Monitoring Active';
            statusBadge.className = 'status-active mt-4 px-4 py-2 rounded-lg';
        }
    }, 3000);
}

// Update gun type display
function updateGunType(data) {
    gunTypeLabel.textContent = `${data.gun_type} (${data.confidence.toFixed(1)}%)`;
    gunTypeLabel.className = 'gun-type-detected p-4 rounded-lg text-center mb-4';
    
    // Update chart
    probabilityChart.data.labels = data.all_probabilities.map(p => p.name);
    probabilityChart.data.datasets[0].data = data.all_probabilities.map(p => p.probability);
    probabilityChart.update();
}

// Update current sounds display
function updateCurrentSounds(sounds) {
    currentSounds.innerHTML = sounds
        .map(sound => `<div class="mb-2">• ${sound}</div>`)
        .join('');
}

// Initialize audio processing
async function initializeAudio() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const source = audioContext.createMediaStreamSource(mediaStream);
        analyzer = audioContext.createAnalyser();
        analyzer.fftSize = 2048;
        
        source.connect(analyzer);
        dataArray = new Uint8Array(analyzer.frequencyBinCount);
        
        // Start audio visualization
        drawAudioVisualization();
        
        return true;
    } catch (error) {
        console.error('Error initializing audio:', error);
        updateStatus('Microphone access denied', 'error');
        return false;
    }
}

// Draw audio visualization
function drawAudioVisualization() {
    if (!isRecording) return;
    
    requestAnimationFrame(drawAudioVisualization);
    analyzer.getByteTimeDomainData(dataArray);
    
    visualizerCtx.fillStyle = 'rgb(249, 250, 251)';
    visualizerCtx.fillRect(0, 0, audioVisualizer.width, audioVisualizer.height);
    
    visualizerCtx.lineWidth = 2;
    visualizerCtx.strokeStyle = 'rgb(59, 130, 246)';
    visualizerCtx.beginPath();
    
    const sliceWidth = audioVisualizer.width / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * audioVisualizer.height / 2;
        
        if (i === 0) {
            visualizerCtx.moveTo(x, y);
        } else {
            visualizerCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    visualizerCtx.lineTo(audioVisualizer.width, audioVisualizer.height / 2);
    visualizerCtx.stroke();
}

// Start recording
async function startRecording() {
    if (await initializeAudio()) {
        isRecording = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        statusBadge.textContent = 'Monitoring Active';
        statusBadge.className = 'status-active mt-4 px-4 py-2 rounded-lg';
        
        // Start sending audio data to server
        sendAudioData();
    }
}

// Stop recording
function stopRecording() {
    isRecording = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    statusBadge.textContent = 'Monitoring Stopped';
    statusBadge.className = 'status-stopped mt-4 px-4 py-2 rounded-lg';
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'stop' }));
    }
}

// Send audio data to server
function sendAudioData() {
    if (!isRecording) return;
    
    analyzer.getByteTimeDomainData(dataArray);
    const audioData = Array.from(dataArray);
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'audio_data',
            data: audioData
        }));
    }
    
    setTimeout(sendAudioData, 100); // Send data every 100ms
}

// Update status display
function updateStatus(message, type) {
    statusBadge.textContent = message;
    statusBadge.className = `mt-4 px-4 py-2 rounded-lg ${
        type === 'error' ? 'status-stopped' : 'status-active'
    }`;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    micPermissionModal.classList.remove('hidden');
});

micPermissionBtn.addEventListener('click', async () => {
    micPermissionModal.classList.add('hidden');
    initializeWebSocket();
});

startButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (isRecording) {
        stopRecording();
    }
    if (socket) {
        socket.close();
    }
}); 