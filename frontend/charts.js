/* ============================================
   Canvas Charts — Training Curves
   ============================================ */

let currentChart = 'reward';

function switchChart(type) {
    currentChart = type;
    document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    drawChart();
}

function drawChart() {
    const canvas = document.getElementById('training-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top: 40, right: 30, bottom: 50, left: 70 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    if (typeof TRAINING_DATA === 'undefined' || !TRAINING_DATA.length) return;

    let data, label, color, gradColor;
    if (currentChart === 'reward') {
        data = TRAINING_DATA.map(d => d.rw);
        label = 'Total Reward';
        color = '#4a9eff';
        gradColor = 'rgba(74,158,255,';
    } else if (currentChart === 'burned') {
        data = TRAINING_DATA.map(d => d.bn);
        label = 'Cells Burned';
        color = '#ff6b35';
        gradColor = 'rgba(255,107,53,';
    } else {
        data = TRAINING_DATA.map(d => d.eps);
        label = 'Epsilon (ε)';
        color = '#a855f7';
        gradColor = 'rgba(168,85,247,';
    }

    const episodes = TRAINING_DATA.map(d => d.ep);
    const minY = Math.min(...data);
    const maxY = Math.max(...data);
    const rangeY = maxY - minY || 1;

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        let y = pad.top + plotH * (1 - i / 5);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        let val = minY + rangeY * (i / 5);
        let y = pad.top + plotH * (1 - i / 5);
        ctx.fillText(Math.round(val * 10) / 10, pad.left - 10, y + 4);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
        let idx = Math.floor(i / 4 * (episodes.length - 1));
        let x = pad.left + (idx / (episodes.length - 1)) * plotW;
        ctx.fillText(episodes[idx], x, H - pad.bottom + 20);
    }

    // Axis titles
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Episode', pad.left + plotW / 2, H - 8);
    ctx.save();
    ctx.translate(16, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(label, 0, 0);
    ctx.restore();

    // Moving average (window=5 of sampled data)
    const avgWin = 5;
    let smoothed = [];
    for (let i = 0; i < data.length; i++) {
        let start = Math.max(0, i - avgWin + 1);
        let slice = data.slice(start, i + 1);
        smoothed.push(slice.reduce((a, b) => a + b) / slice.length);
    }

    function toX(i) { return pad.left + (i / (data.length - 1)) * plotW; }
    function toY(v) { return pad.top + plotH * (1 - (v - minY) / rangeY); }

    // Gradient fill under smooth line
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(smoothed[0]));
    for (let i = 1; i < smoothed.length; i++) ctx.lineTo(toX(i), toY(smoothed[i]));
    ctx.lineTo(toX(smoothed.length - 1), pad.top + plotH);
    ctx.lineTo(toX(0), pad.top + plotH);
    ctx.closePath();
    let grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, gradColor + '0.2)');
    grad.addColorStop(1, gradColor + '0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Raw data (dots)
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.15;
    for (let i = 0; i < data.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(i), toY(data[i]), 2, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Smooth line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(smoothed[0]));
    for (let i = 1; i < smoothed.length; i++) ctx.lineTo(toX(i), toY(smoothed[i]));
    ctx.stroke();

    // Title
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label + ' over Training Episodes (exp-qlearning-1)', pad.left, pad.top - 16);

    // Legend dot
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(pad.left + plotW - 100, pad.top - 20, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Moving Avg', pad.left + plotW - 90, pad.top - 16);
}

function initCharts() {
    drawChart();
}
