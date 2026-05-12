// ========== Save / Load Spline Configuration ==========
document.getElementById('saveFigBtn').addEventListener('click', () => {
    const data = {
        format: 'symmetry-set-spline',
        version: 1,
        curves: curves.map(c => c.map(p => ({ x: p.x, y: p.y }))),
        curveOpen: curveOpen.slice(),
        activeCurveIdx: activeCurveIdx,
        view: { x: view.x, y: view.y, scale: view.scale }
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.download = `figure-${ts}.sspl`;
    a.click();
    URL.revokeObjectURL(url);
    updateStatus('Saved figure');
});

document.getElementById('loadFigBtn').addEventListener('click', () => {
    document.getElementById('loadFigInput').click();
});

document.getElementById('loadFigInput').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
        try {
            applyFigureData(JSON.parse(ev.target.result));
        } catch (err) {
            updateStatus('Failed to parse file');
        }
        e.target.value = '';
    };
    reader.readAsText(file);
});

// ========== CSV Upload ==========
document.getElementById('uploadCSVBtn').addEventListener('click', () => {
    document.getElementById('csvFileInput').click();
});

document.getElementById('csvFileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const csvText = event.target.result;
            const points = parseCSV(csvText);
            
            if (points.length < 3) {
                updateStatus('CSV needs at least 3 points');
                return;
            }
            
            loadCSVCurve(points);
        } catch (err) {
            updateStatus('Error parsing CSV: ' + err.message);
            console.error(err);
        }
    };
    reader.readAsText(file);
    
    // Reset file input so same file can be re-uploaded
    e.target.value = '';
});

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) throw new Error('CSV must have header and data rows');
    
    // Parse header to find x and y columns
    const header = lines[0].toLowerCase().split(',').map(h => h.trim());
    const xIdx = header.findIndex(h => h === 'x');
    const yIdx = header.findIndex(h => h === 'y');
    
    if (xIdx === -1 || yIdx === -1) {
        throw new Error('CSV must have "x" and "y" columns');
    }
    
    const points = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = line.split(',').map(v => v.trim());
        const x = parseFloat(values[xIdx]);
        const y = parseFloat(values[yIdx]);
        
        if (!isNaN(x) && !isNaN(y)) {
            points.push({ x, y });
        }
    }
    
    return points;
}

function loadCSVCurve(points) {
    saveState();

    // Enable fixed curve mode for CSV data; respect the active curve's open flag.
    fixedCurveMode = true;
    const isOpen = !!curveOpen[activeCurveIdx];
    curves = [[]];
    curveOpen = [isOpen];
    activeCurveIdx = 0;

    // Generate curve data from CSV points
    const n = points.length;
    cachedCurveData = [];

    for (let i = 0; i < n; i++) {
        // Open: forward/backward differences at the endpoints; central inside.
        // Closed: cyclic central difference everywhere.
        let prev, curr, next;
        if (isOpen && i === 0) {
            prev = points[0]; curr = points[0]; next = points[1];
        } else if (isOpen && i === n - 1) {
            prev = points[n - 2]; curr = points[n - 1]; next = points[n - 1];
        } else {
            prev = points[(i - 1 + n) % n];
            curr = points[i];
            next = points[(i + 1) % n];
        }

        const dx = (next.x - prev.x) / 2;
        const dy = (next.y - prev.y) / 2;
        const vel = Math.sqrt(dx*dx + dy*dy);

        if (vel < 1e-10) continue;

        const Tx = dx / vel;
        const Ty = dy / vel;
        const Nx = -Ty;
        const Ny = Tx;

        // Curvature from second difference (skip endpoints in open mode — no
        // meaningful 2nd derivative there).
        let curvature = 0;
        if (!(isOpen && (i === 0 || i === n - 1))) {
            const ddx = next.x - 2*curr.x + prev.x;
            const ddy = next.y - 2*curr.y + prev.y;
            curvature = (dx * ddy - dy * ddx) / (vel * vel * vel);
        }

        cachedCurveData.push({
            p: new Point(curr.x, curr.y),
            T: new Point(Tx, Ty),
            N: new Point(Nx, Ny),
            curvature: curvature,
            curveId: 0
        });
    }
    
    // Compute symmetry set
    cachedSSData = computeSymmetrySet(cachedCurveData, ssStepSize); sortedSSIndices = null;
    cachedFocalData = computeFocalSet(cachedCurveData);
    
    // Auto-fit view to data
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const pt of points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }
    
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    const maxRange = Math.max(rangeX, rangeY);
    
    // Set view to fit curve with some padding
    view.scale = Math.min(canvas.width, canvas.height) / (maxRange * 1.5);
    view.x = -centerX * view.scale;
    view.y = centerY * view.scale;
    
    updateUI();
    updateStatus(`CSV loaded: ${cachedCurveData.length} pts, SS: ${cachedSSData.length}`);
    draw();
}

document.getElementById('resolution').addEventListener('input', e => {
    ssStepSize = 6 - parseInt(e.target.value);
});

document.getElementById('vineyardModeBtn').addEventListener('click', () => {
    vineyardMode = !vineyardMode;
    if (vineyardMode) {
        customLoopMode = false;
        document.getElementById('drawLoopBtn').classList.remove('active');
    }
    updateModeIndicator();
    const btn = document.getElementById('vineyardModeBtn');
    if (vineyardMode) {
        btn.classList.add('active');
        btn.innerText = 'Click Canvas...';
        updateStatus('Click to place center');
    } else {
        btn.classList.remove('active');
        btn.innerText = '⬡ Place Center';
    }
});

document.getElementById('vineyardRadius').addEventListener('input', e => {
    vineyardRadius = parseFloat(e.target.value) || 1;
    if (vineyardCenter) draw();
});

document.getElementById('vineyardRadius').addEventListener('change', e => {
    vineyardRadius = parseFloat(e.target.value) || 1;
    if (vineyardRadius <= 0) {
        vineyardRadius = 0.1;
        e.target.value = 0.1;
    }
    if (vineyardCenter) draw();
});

document.getElementById('vineyardSamples').addEventListener('input', e => {
    vineyardSamples = parseInt(e.target.value);
    document.getElementById('samplesVal').innerText = vineyardSamples;
});

document.getElementById('computeVineyardBtn').addEventListener('click', computeVineyard);

document.getElementById('playVineyardBtn').addEventListener('click', () => {
    if (!vineyardData) return;
    vineyardAnimPlaying = !vineyardAnimPlaying;
    document.getElementById('playVineyardBtn').innerText = vineyardAnimPlaying ? '⏸' : '▶';
    if (vineyardAnimPlaying) vineyardAnimLoop();
});

document.getElementById('stopVineyardBtn').addEventListener('click', stopVineyardAnim);

document.getElementById('vineyardAnimSlider').addEventListener('input', e => {
    if (!vineyardCenters.length) return;
    const pct = parseFloat(e.target.value) / 1000;
    vineyardAnimIdx = Math.floor(pct * (vineyardCenters.length - 1));
    updatePersistenceDiagram(vineyardAnimIdx);
    updateVineyardPlot();
    draw();
});

// Loop type selector
document.getElementById('vineyardLoopType').addEventListener('change', e => {
    vineyardLoopType = e.target.value;
    const circularControls = document.getElementById('circularLoopControls');
    const customControls = document.getElementById('customLoopControls');
    
    if (vineyardLoopType === 'circular') {
        circularControls.style.display = 'block';
        customControls.style.display = 'none';
        customLoopMode = false;
        document.getElementById('drawLoopBtn').classList.remove('active');
    } else {
        circularControls.style.display = 'none';
        customControls.style.display = 'block';
        vineyardMode = false;
        document.getElementById('vineyardModeBtn').classList.remove('active');
    }
    draw();
});

// Draw loop button
document.getElementById('drawLoopBtn').addEventListener('click', () => {
    customLoopMode = !customLoopMode;
    document.getElementById('drawLoopBtn').classList.toggle('active', customLoopMode);
    if (customLoopMode) {
        vineyardMode = false;
        document.getElementById('vineyardModeBtn').classList.remove('active');
    }
    updateModeIndicator();
    draw();
});

// Clear loop button
document.getElementById('clearLoopBtn').addEventListener('click', () => {
    customLoopPoints = [];
    updateLoopStatus();
    draw();
});

function updateLoopStatus() {
    const status = document.getElementById('loopStatus');
    if (customLoopPoints.length === 0) {
        status.innerText = 'No loop drawn';
        status.style.color = 'var(--text-muted)';
    } else if (customLoopPoints.length < 3) {
        status.innerText = `${customLoopPoints.length} points (need 3+)`;
        status.style.color = '#f59e0b';
    } else {
        status.innerText = `${customLoopPoints.length} points ✓`;
        status.style.color = '#22c55e';
    }
}

// Sample points along the custom loop spline (auto-closed)
function sampleCustomLoop(numSamples) {
    if (customLoopPoints.length < 3) return [];
    
    const pts = customLoopPoints;
    const C = solveControlPoints(pts);
    const n = C.length;
    // For closed loop, wrap around
    const drawC = [C[n-1], ...C, C[0], C[1]];
    
    const numSegments = pts.length;
    const pointsPerSeg = Math.ceil(numSamples / numSegments);
    const samples = [];
    
    for (let i = 0; i < numSegments; i++) {
        const p0 = drawC[i], p1 = drawC[i+1], p2 = drawC[i+2], p3 = drawC[i+3];
        
        const segSamples = (i === numSegments - 1) ? 
            numSamples - samples.length : pointsPerSeg;
        
        for (let j = 0; j < segSamples; j++) {
            const t = j / segSamples;
            const P = bSplineEval(p0, p1, p2, p3, t);
            samples.push({ x: P.x, y: P.y });
        }
    }
    
    return samples;
}

// Compute vineyard along custom loop
['showCurve', 'showControls', 'showFocal', 'showSS', 'showVineyardCircle', 'showSweepPreview', 'showBirthDeathCircles'].forEach(id => {
    document.getElementById(id).addEventListener('change', draw);
});

document.getElementById('showVineyardPlot').addEventListener('change', e => {
    document.getElementById('vineyardPanel').classList.toggle('active', e.target.checked && vineyardData);
});

document.getElementById('showPD').addEventListener('change', e => {
    document.getElementById('persistencePanel').classList.toggle('active', e.target.checked && vineyardData);
});

let vineyardExpanded = false;
function clearPanelInlineSize(panel) {
    panel.style.width = '';
    panel.style.height = '';
    panel.style.left = '';
    panel.style.top = '';
    panel.style.right = '';
    panel.style.bottom = '';
    panel.style.position = '';
}

document.getElementById('expandVineyardBtn').addEventListener('click', () => {
    vineyardExpanded = !vineyardExpanded;
    const panel = document.getElementById('vineyardPanel');
    const btn = document.getElementById('expandVineyardBtn');

    clearPanelInlineSize(panel);
    if (vineyardExpanded) {
        panel.classList.add('expanded');
        btn.innerText = '⤡';
    } else {
        panel.classList.remove('expanded');
        btn.innerText = '⤢';
    }

    setTimeout(() => {
        if (vineyardData) {
            Plotly.relayout('vineyardPlot', { autosize: true });
        }
    }, 350);
});

let pdExpanded = false;
document.getElementById('expandPDBtn').addEventListener('click', () => {
    pdExpanded = !pdExpanded;
    const panel = document.getElementById('persistencePanel');
    const btn = document.getElementById('expandPDBtn');

    clearPanelInlineSize(panel);
    if (pdExpanded) {
        panel.classList.add('expanded');
        btn.innerText = '⤡';
    } else {
        panel.classList.remove('expanded');
        btn.innerText = '⤢';
    }

    setTimeout(() => {
        if (vineyardData) {
            Plotly.relayout('persistencePlot', { autosize: true });
        }
    }, 350);
});

function stopVineyardAnim() {
    vineyardAnimPlaying = false;
    vineyardAnimIdx = 0;
    document.getElementById('playVineyardBtn').innerText = '▶';
    document.getElementById('vineyardAnimSlider').value = 0;
    if (vineyardData) {
        updatePersistenceDiagram(0);
        updateVineyardPlot();
    }
    draw();
}

function vineyardAnimLoop() {
    if (!vineyardAnimPlaying || !vineyardCenters.length) return;
    
    vineyardAnimIdx = (vineyardAnimIdx + 1) % vineyardCenters.length;
    const pct = vineyardAnimIdx / vineyardCenters.length;
    document.getElementById('vineyardAnimSlider').value = Math.floor(pct * 1000);
    
    updatePersistenceDiagram(vineyardAnimIdx);
    updateVineyardPlot();
    draw();
    
    setTimeout(vineyardAnimLoop, 50);
}

