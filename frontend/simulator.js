/* ============================================
   Wildfire Grid Simulator (Browser Canvas)
   ============================================ */

// Cell states
const EMPTY = 0, TREE = 1, BURNING = 2, BURNED = 3, FIREBREAK = 4;

// Colors
const CELL_COLORS = {
    [EMPTY]: '#0d0d20',
    [TREE]: '#2d5a27',
    [BURNING]: '#ff4500',
    [BURNED]: '#1a1a2e',
    [FIREBREAK]: '#4a9eff'
};

const NEIGHBOR_OFFSETS = [[-1,0],[1,0],[0,1],[0,-1],[-1,1],[-1,-1],[1,1],[1,-1]];

const WIND_VECTORS = {
    N:[-1,0], S:[1,0], E:[0,1], W:[0,-1],
    NE:[-1,1], NW:[-1,-1], SE:[1,1], SW:[1,-1]
};

class WildfireSim {
    constructor(canvas, size=10) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.size = size;
        this.grid = [];
        this.step = 0;
        this.maxSteps = 50;
        this.totalBurned = 0;
        this.totalReward = 0;
        this.running = false;
        this.done = false;
        this.agentMode = 'random';
        this.windDir = 'N';
        this.fireCount = 2;
        this.speed = 5;
        this.baseSpread = 0.3;
        this.windBonus = 0.2;
        this.deployCell = null;
        this.particleEffects = [];
        this.reset();
    }

    reset() {
        this.step = 0;
        this.totalBurned = 0;
        this.totalReward = 0;
        this.done = false;
        this.deployCell = null;
        this.particleEffects = [];
        this.grid = Array.from({length:this.size}, () =>
            Array.from({length:this.size}, () => Math.random() < 0.85 ? TREE : EMPTY)
        );
        // Place fires
        let treeCells = [];
        for(let r=0;r<this.size;r++) for(let c=0;c<this.size;c++)
            if(this.grid[r][c]===TREE) treeCells.push([r,c]);
        for(let i=0; i<Math.min(this.fireCount, treeCells.length); i++) {
            let idx = Math.floor(Math.random()*treeCells.length);
            let [r,c] = treeCells.splice(idx,1)[0];
            this.grid[r][c] = BURNING;
        }
        this.render();
        this.updateMetrics();
    }

    getWindMap() {
        let wv = WIND_VECTORS[this.windDir] || [0,0];
        let m = {};
        for(let [dr,dc] of NEIGHBOR_OFFSETS) {
            let dot = dr*wv[0] + dc*wv[1];
            m[`${dr},${dc}`] = Math.max(0, dot) * this.windBonus;
        }
        return m;
    }

    getSectorState() {
        // 2x2 sectors
        let half = Math.floor(this.size / 2);
        let sectors = [[0,half,0,half],[0,half,half,this.size],
                       [half,this.size,0,half],[half,this.size,half,this.size]];
        return sectors.map(([r0,r1,c0,c1]) => {
            let burning=0, trees=0;
            for(let r=r0;r<r1;r++) for(let c=c0;c<c1;c++) {
                if(this.grid[r][c]===BURNING) burning++;
                if(this.grid[r][c]===TREE) trees++;
            }
            return {burning, trees, r0, r1, c0, c1};
        });
    }

    chooseAction() {
        let sectors = this.getSectorState();
        if(this.agentMode === 'random') {
            return Math.floor(Math.random() * 4);
        }
        // RL-like heuristic: pick sector with most burning cells
        let best = 0, bestScore = -1;
        sectors.forEach((s, i) => {
            let score = s.burning * 3 + (s.trees > 10 ? 1 : 0);
            if(score > bestScore) { bestScore = score; best = i; }
        });
        return best;
    }

    deployToSector(sectorIdx) {
        let sectors = this.getSectorState();
        let s = sectors[sectorIdx];
        let suppressed = 0;
        // Find burning cell in sector
        let target = null;
        for(let r=s.r0;r<s.r1;r++) for(let c=s.c0;c<s.c1;c++) {
            if(this.grid[r][c]===BURNING) { target=[r,c]; break; }
            if(target) break;
        }
        if(target) {
            this.grid[target[0]][target[1]] = FIREBREAK;
            suppressed++;
            this.deployCell = target;
            this.addParticle(target[0], target[1], '#4a9eff');
        } else {
            // Firebreak on tree near fire
            for(let r=s.r0;r<s.r1;r++) for(let c=s.c0;c<s.c1;c++) {
                if(this.grid[r][c]===TREE) {
                    for(let [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                        let nr=r+dr, nc=c+dc;
                        if(nr>=0&&nr<this.size&&nc>=0&&nc<this.size && this.grid[nr][nc]===BURNING) {
                            this.grid[r][c] = FIREBREAK;
                            this.deployCell = [r,c];
                            this.addParticle(r, c, '#4a9eff');
                            return suppressed;
                        }
                    }
                }
            }
            let mr=Math.floor((s.r0+s.r1)/2), mc=Math.floor((s.c0+s.c1)/2);
            if(this.grid[mr][mc]===TREE) { this.grid[mr][mc]=FIREBREAK; this.deployCell=[mr,mc]; }
        }
        // Suppress neighbors
        if(this.deployCell) {
            let [dr2, dc2] = this.deployCell;
            for(let [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                let nr=dr2+dr, nc=dc2+dc;
                if(nr>=0&&nr<this.size&&nc>=0&&nc<this.size && this.grid[nr][nc]===BURNING && Math.random()<0.35) {
                    this.grid[nr][nc]=FIREBREAK; suppressed++;
                }
            }
        }
        return suppressed;
    }

    spreadFire() {
        let wmap = this.getWindMap();
        let newGrid = this.grid.map(r=>[...r]);
        let burnedBefore = this.grid.flat().filter(c=>c===BURNED).length;

        for(let r=0;r<this.size;r++) for(let c=0;c<this.size;c++) {
            if(this.grid[r][c]!==BURNING) continue;
            if(Math.random()<0.15) newGrid[r][c]=BURNED;
            for(let [dr,dc] of NEIGHBOR_OFFSETS) {
                let nr=r+dr, nc=c+dc;
                if(nr>=0&&nr<this.size&&nc>=0&&nc<this.size && this.grid[nr][nc]===TREE) {
                    let p = this.baseSpread + (wmap[`${dr},${dc}`]||0);
                    if(Math.random()<p) {
                        newGrid[nr][nc] = BURNING;
                        this.addParticle(nr, nc, '#ff6b35');
                    }
                }
            }
        }
        this.grid = newGrid;
        let burnedAfter = this.grid.flat().filter(c=>c===BURNED||c===BURNING).length;
        return Math.max(0, burnedAfter - burnedBefore);
    }

    addParticle(r, c, color) {
        let cellW = this.canvas.width / this.size;
        for(let i=0;i<3;i++) {
            this.particleEffects.push({
                x: c*cellW + cellW/2, y: r*cellW + cellW/2,
                vx: (Math.random()-0.5)*3, vy: -Math.random()*3-1,
                life: 1, color, size: Math.random()*4+2
            });
        }
    }

    tick() {
        if(this.done) return;
        let action = this.chooseAction();
        let suppressed = this.deployToSector(action);
        let newBurned = this.spreadFire();
        this.totalBurned += newBurned;
        this.totalReward += -newBurned + suppressed * 2;
        this.step++;
        let noFire = this.grid.flat().filter(c=>c===BURNING).length === 0;
        if(noFire || this.step >= this.maxSteps) this.done = true;
        this.render();
        this.updateMetrics();
    }

    render() {
        let ctx = this.ctx;
        let w = this.canvas.width, h = this.canvas.height;
        let cellW = w / this.size, cellH = h / this.size;
        ctx.clearRect(0, 0, w, h);

        // Draw grid
        for(let r=0;r<this.size;r++) for(let c=0;c<this.size;c++) {
            let val = this.grid[r][c];
            let x=c*cellW, y=r*cellH;

            // Base color
            ctx.fillStyle = CELL_COLORS[val];
            ctx.beginPath();
            ctx.roundRect(x+2, y+2, cellW-4, cellH-4, 4);
            ctx.fill();

            // Fire glow
            if(val === BURNING) {
                let flicker = 0.6 + Math.random()*0.4;
                let grad = ctx.createRadialGradient(x+cellW/2, y+cellH/2, 0, x+cellW/2, y+cellH/2, cellW);
                grad.addColorStop(0, `rgba(255,69,0,${flicker})`);
                grad.addColorStop(0.5, `rgba(255,140,0,${flicker*0.4})`);
                grad.addColorStop(1, 'transparent');
                ctx.fillStyle = grad;
                ctx.fillRect(x-cellW/3, y-cellH/3, cellW*1.6, cellH*1.6);
            }

            // Deploy highlight
            if(this.deployCell && this.deployCell[0]===r && this.deployCell[1]===c) {
                ctx.strokeStyle = '#ffdd57';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.roundRect(x+1, y+1, cellW-2, cellH-2, 5);
                ctx.stroke();
            }
        }

        // Sector lines
        let half = Math.floor(this.size/2);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5,5]);
        ctx.beginPath();
        ctx.moveTo(half*cellW, 0); ctx.lineTo(half*cellW, h);
        ctx.moveTo(0, half*cellH); ctx.lineTo(w, half*cellH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Particles
        this.particleEffects = this.particleEffects.filter(p => {
            p.x += p.vx; p.y += p.vy; p.life -= 0.04; p.size *= 0.97;
            if(p.life <= 0) return false;
            ctx.globalAlpha = p.life;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
            ctx.fill();
            ctx.globalAlpha = 1;
            return true;
        });
    }

    updateMetrics() {
        let flat = this.grid.flat();
        let burning = flat.filter(c=>c===BURNING).length;
        let burned = flat.filter(c=>c===BURNED).length;
        let trees = flat.filter(c=>c===TREE).length;
        let breaks = flat.filter(c=>c===FIREBREAK).length;
        let total = this.size * this.size;

        document.getElementById('metric-burning').textContent = burning;
        document.getElementById('metric-burned').textContent = burned;
        document.getElementById('metric-trees').textContent = trees;
        document.getElementById('metric-firebreaks').textContent = breaks;
        document.getElementById('metric-reward').textContent = Math.round(this.totalReward);
        document.getElementById('metric-total-burned').textContent = this.totalBurned;
        document.getElementById('sim-step').textContent = this.step;

        document.getElementById('bar-burning').style.width = (burning/total*100)+'%';
        document.getElementById('bar-burned').style.width = (burned/total*100)+'%';
        document.getElementById('bar-trees').style.width = (trees/total*100)+'%';
        document.getElementById('bar-firebreaks').style.width = (breaks/total*100)+'%';

        let dot = document.getElementById('status-dot');
        let txt = document.getElementById('sim-status-text');
        if(this.running && !this.done) { dot.className='status-dot running'; txt.textContent='Running'; }
        else if(this.done) { dot.className='status-dot done'; txt.textContent='Complete'; }
        else { dot.className='status-dot'; txt.textContent='Ready'; }
    }
}

// Global sim instance
let sim;
let simInterval;

function initSimulator() {
    let canvas = document.getElementById('sim-canvas');
    sim = new WildfireSim(canvas, 10);
}

function startSimulation() {
    if(!sim) initSimulator();
    if(sim.done) sim.reset();
    sim.running = true;
    let delay = Math.max(50, 600 - sim.speed * 55);
    if(simInterval) clearInterval(simInterval);
    simInterval = setInterval(() => {
        if(sim.done) { clearInterval(simInterval); sim.running=false; sim.updateMetrics(); return; }
        sim.tick();
    }, delay);
}

function resetSimulation() {
    if(simInterval) clearInterval(simInterval);
    sim.running = false;
    sim.reset();
}

function setAgentMode(mode) {
    sim.agentMode = mode;
    document.getElementById('btn-random').classList.toggle('active', mode==='random');
    document.getElementById('btn-rl').classList.toggle('active', mode==='rl');
}

function setWind(dir) {
    sim.windDir = dir;
    document.querySelectorAll('.wind-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.dir === dir);
    });
}

function setSpeed(val) {
    sim.speed = parseInt(val);
    document.getElementById('speed-val').textContent = val + 'x';
    if(sim.running) { startSimulation(); }
}

function setFireCount(val) {
    sim.fireCount = parseInt(val);
    document.getElementById('fires-val').textContent = val;
}
