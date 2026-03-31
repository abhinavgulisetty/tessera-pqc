document.addEventListener('DOMContentLoaded', function() {
    initSmoothScrolling();
    initAttackDemo();
    initAnimations();
});

function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPos = target.offsetTop - navHeight - 20;
                window.scrollTo({
                    top: targetPos,
                    behavior: 'smooth'
                });
            }
        });
    });
}

const SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
];

function hammingWeight(x) {
    let count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

function generateTraces(nTraces, traceLength, keyByte, noiseLevel) {
    const traces = [];
    const plaintexts = [];
    
    for (let i = 0; i < nTraces; i++) {
        const plaintext = Math.floor(Math.random() * 256);
        plaintexts.push(plaintext);
        
        const sboxOut = SBOX[plaintext ^ keyByte];
        const hw = hammingWeight(sboxOut);
        
        const trace = [];
        for (let j = 0; j < traceLength; j++) {
            let value = (Math.random() - 0.5) * noiseLevel * 2;
            if (j === Math.floor(traceLength / 2)) {
                value += hw;
            }
            trace.push(value);
        }
        traces.push(trace);
    }
    
    return { traces, plaintexts, keyByte };
}

function pearsonCorrelation(x, y) {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return den === 0 ? 0 : num / den;
}

function cpaAttack(traces, plaintexts, targetPoint) {
    const correlations = new Array(256).fill(0);
    const traceColumn = traces.map(t => t[targetPoint]);
    
    for (let keyGuess = 0; keyGuess < 256; keyGuess++) {
        const hypotheses = plaintexts.map(p => hammingWeight(SBOX[p ^ keyGuess]));
        correlations[keyGuess] = Math.abs(pearsonCorrelation(traceColumn, hypotheses));
    }
    
    let maxCorr = 0;
    let recoveredKey = 0;
    for (let i = 0; i < 256; i++) {
        if (correlations[i] > maxCorr) {
            maxCorr = correlations[i];
            recoveredKey = i;
        }
    }
    
    return { recoveredKey, maxCorrelation: maxCorr, correlations };
}

let attackRunning = false;

function initAttackDemo() {
    window.runAttackDemo = async function() {
        if (attackRunning) return;
        attackRunning = true;
        
        const statusEl = document.getElementById('demo-status');
        const trueKeyEl = document.getElementById('true-key');
        const recoveredKeyEl = document.getElementById('recovered-key');
        const correlationEl = document.getElementById('correlation');
        const canvas = document.getElementById('attack-canvas');
        const ctx = canvas.getContext('2d');
        
        statusEl.textContent = 'Generating traces...';
        statusEl.style.color = '#ffd43b';
        
        await sleep(300);
        
        const trueKey = Math.floor(Math.random() * 256);
        trueKeyEl.textContent = '0x' + trueKey.toString(16).padStart(2, '0').toUpperCase();
        recoveredKeyEl.textContent = '0x??';
        recoveredKeyEl.style.color = '#8b949e';
        correlationEl.textContent = '-';
        
        const nTraces = 500;
        const traceLength = 32;
        const noiseLevel = 1.5;
        
        const data = generateTraces(nTraces, traceLength, trueKey, noiseLevel);
        
        statusEl.textContent = 'Running CPA attack...';
        
        ctx.fillStyle = '#161b22';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const targetPoint = Math.floor(traceLength / 2);
        
        await animateCorrelations(ctx, canvas, data.traces, data.plaintexts, targetPoint, trueKey);
        
        const result = cpaAttack(data.traces, data.plaintexts, targetPoint);
        
        recoveredKeyEl.textContent = '0x' + result.recoveredKey.toString(16).padStart(2, '0').toUpperCase();
        correlationEl.textContent = result.maxCorrelation.toFixed(4);
        
        if (result.recoveredKey === trueKey) {
            statusEl.textContent = 'Key recovered successfully!';
            statusEl.style.color = '#51cf66';
            recoveredKeyEl.style.color = '#51cf66';
        } else {
            statusEl.textContent = 'Attack completed (need more traces)';
            statusEl.style.color = '#ff6b6b';
            recoveredKeyEl.style.color = '#ff6b6b';
        }
        
        attackRunning = false;
    };
}

async function animateCorrelations(ctx, canvas, traces, plaintexts, targetPoint, trueKey) {
    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;
    
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Key Hypothesis (0-255)', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(12, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Correlation', 0, 0);
    ctx.restore();
    
    const traceColumn = traces.map(t => t[targetPoint]);
    const correlations = new Array(256).fill(0);
    
    const batchSize = 16;
    for (let batch = 0; batch < 256; batch += batchSize) {
        for (let keyGuess = batch; keyGuess < Math.min(batch + batchSize, 256); keyGuess++) {
            const hypotheses = plaintexts.map(p => hammingWeight(SBOX[p ^ keyGuess]));
            correlations[keyGuess] = Math.abs(pearsonCorrelation(traceColumn, hypotheses));
        }
        
        ctx.fillStyle = '#161b22';
        ctx.fillRect(padding + 1, padding, graphWidth - 1, graphHeight - 1);
        
        const barWidth = graphWidth / 256;
        for (let i = 0; i <= batch + batchSize && i < 256; i++) {
            const x = padding + i * barWidth;
            const barHeight = correlations[i] * graphHeight;
            const y = height - padding - barHeight;
            
            if (i === trueKey) {
                ctx.fillStyle = '#51cf66';
            } else {
                ctx.fillStyle = '#4a9eff';
            }
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        }
        
        await sleep(20);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function initAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.overview-card, .attack-card, .cm-card, .flow-container').forEach(el => {
        observer.observe(el);
    });
    
    const perfBars = document.querySelectorAll('.perf-bar');
    const perfObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.style.width;
                bar.style.width = '0';
                setTimeout(() => {
                    bar.style.width = width;
                }, 100);
            }
        });
    }, observerOptions);
    
    perfBars.forEach(bar => perfObserver.observe(bar));
}
