/* ============================================
   PyroShieldAI — Main Application
   ============================================ */

// --- Particle Background ---
function initParticles() {
    const canvas = document.getElementById('particle-bg');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const count = 60;

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: -Math.random() * 0.5 - 0.1,
            size: Math.random() * 2 + 0.5,
            opacity: Math.random() * 0.3 + 0.05,
            color: Math.random() > 0.5 ? '#ff6b35' : '#ff9f43'
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.y < -10) { p.y = canvas.height + 10; p.x = Math.random() * canvas.width; }
            if (p.x < -10) p.x = canvas.width + 10;
            if (p.x > canvas.width + 10) p.x = -10;

            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;
        requestAnimationFrame(animate);
    }
    animate();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// --- Hero Fire Canvas ---
function initHeroFire() {
    const canvas = document.getElementById('hero-fire-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const gridSize = 10;
    const cellW = W / gridSize, cellH = H / gridSize;

    // Simple animated fire grid
    let grid = Array.from({ length: gridSize }, () =>
        Array.from({ length: gridSize }, () => Math.random() < 0.8 ? 1 : 0)
    );
    // Initial fires
    grid[3][4] = 2; grid[3][5] = 2; grid[4][4] = 2;
    grid[6][7] = 2; grid[7][7] = 2;

    const colors = {
        0: '#0d0d20', 1: '#1a3a15', 2: '#ff4500', 3: '#1a1a2e', 4: '#2a5aff'
    };

    function tick() {
        let ng = grid.map(r => [...r]);
        for (let r = 0; r < gridSize; r++) for (let c = 0; c < gridSize; c++) {
            if (grid[r][c] === 2) {
                if (Math.random() < 0.08) ng[r][c] = 3;
                for (let [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
                    let nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize && grid[nr][nc] === 1) {
                        if (Math.random() < 0.15) ng[nr][nc] = 2;
                    }
                }
            }
        }
        grid = ng;
        // Reset if all burned
        if (!grid.flat().includes(2) && !grid.flat().includes(1)) {
            grid = Array.from({ length: gridSize }, () =>
                Array.from({ length: gridSize }, () => Math.random() < 0.8 ? 1 : 0)
            );
            let fr = Math.floor(Math.random() * gridSize), fc = Math.floor(Math.random() * gridSize);
            grid[fr][fc] = 2;
        }
    }

    function render() {
        ctx.clearRect(0, 0, W, H);
        for (let r = 0; r < gridSize; r++) for (let c = 0; c < gridSize; c++) {
            let val = grid[r][c];
            let x = c * cellW, y = r * cellH;
            ctx.fillStyle = colors[val] || '#0d0d20';
            ctx.beginPath();
            ctx.roundRect(x + 2, y + 2, cellW - 4, cellH - 4, 4);
            ctx.fill();
            if (val === 2) {
                let f = 0.4 + Math.random() * 0.5;
                let g = ctx.createRadialGradient(x + cellW / 2, y + cellH / 2, 0, x + cellW / 2, y + cellH / 2, cellW);
                g.addColorStop(0, `rgba(255,69,0,${f})`);
                g.addColorStop(1, 'transparent');
                ctx.fillStyle = g;
                ctx.fillRect(x - 5, y - 5, cellW + 10, cellH + 10);
            }
        }
    }

    function loop() {
        tick();
        render();
    }
    setInterval(loop, 400);
    render();
}

// --- Counter Animation ---
function animateCounters() {
    document.querySelectorAll('.counter').forEach(el => {
        const target = parseFloat(el.dataset.target);
        const isFloat = target % 1 !== 0;
        const duration = 2000;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    });
}

// --- Scroll Spy Nav ---
function initScrollSpy() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(l => l.classList.remove('active'));
                let id = entry.target.id;
                let link = document.querySelector(`.nav-link[data-section="${id}"]`);
                if (link) link.classList.add('active');
            }
        });
    }, { threshold: 0.3 });

    sections.forEach(s => observer.observe(s));
}

// --- Intersection Observer for animations ---
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.glass-card, .stat-card, .comp-card, .sdg-card, .insight-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// --- Nav scroll effect ---
function initNavScroll() {
    const nav = document.getElementById('main-nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.padding = '10px 40px';
            nav.style.background = 'rgba(10,10,26,0.95)';
        } else {
            nav.style.padding = '14px 40px';
            nav.style.background = 'rgba(10,10,26,0.85)';
        }
    });
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initHeroFire();
    initSimulator();
    initCharts();
    animateCounters();
    initScrollSpy();
    initScrollAnimations();
    initNavScroll();
});
