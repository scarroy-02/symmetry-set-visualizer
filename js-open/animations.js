// ========== Focal Animation ==========
function stopFocalAnim() {
    focalAnimPlaying = false;
    focalAnimIdx = 0;
    document.getElementById('focalPlayBtn').innerText = '▶';
    document.getElementById('focalPlayBtn').style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
    document.getElementById('focalAnimSlider').value = 0;
    draw();
}

function focalAnimLoop() {
    if (!focalAnimPlaying || !cachedCurveData || cachedCurveData.length === 0) return;
    
    focalAnimIdx = (focalAnimIdx + 15) % cachedCurveData.length;
    const pct = focalAnimIdx / cachedCurveData.length;
    document.getElementById('focalAnimSlider').value = Math.floor(pct * 1000);
    
    draw();
    requestAnimationFrame(focalAnimLoop);
}

document.getElementById('focalPlayBtn').addEventListener('click', () => {
    if (!cachedCurveData || cachedCurveData.length === 0) return;
    
    focalAnimPlaying = !focalAnimPlaying;
    const btn = document.getElementById('focalPlayBtn');
    
    if (focalAnimPlaying) {
        btn.innerText = '⏸';
        btn.style.background = 'linear-gradient(135deg, #dc2626, #ef4444)';
        focalAnimLoop();
    } else {
        btn.innerText = '▶';
        btn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
    }
});

document.getElementById('focalStopBtn').addEventListener('click', stopFocalAnim);

document.getElementById('focalAnimSlider').addEventListener('input', e => {
    if (!cachedCurveData || cachedCurveData.length === 0) return;
    const pct = parseFloat(e.target.value) / 1000;
    focalAnimIdx = Math.floor(pct * (cachedCurveData.length - 1));
    draw();
});

// ========== Symmetry Set Animation ==========
function stopSSAnim() {
    ssAnimPlaying = false;
    ssAnimIdx = 0;
    document.getElementById('ssPlayBtn').innerText = '▶';
    document.getElementById('ssPlayBtn').style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    document.getElementById('ssAnimSlider').value = 0;
    draw();
}

let sortedSSIndices = null;

function buildSortedSSIndices() {
    if (!cachedSSData || cachedSSData.length === 0) { sortedSSIndices = null; return; }

    // Fast path: walk the already-ordered branches.
    if (cachedSSData.branches && cachedSSData.branches.length > 0) {
        const ptToIdx = new Map();
        for (let i = 0; i < cachedSSData.length; i++) ptToIdx.set(cachedSSData[i], i);
        const order = [];
        for (const br of cachedSSData.branches) {
            for (const p of br) {
                const i = ptToIdx.get(p);
                if (i !== undefined) order.push(i);
            }
        }
        sortedSSIndices = order;
        return;
    }

    // Fallback: nearest-neighbor (old behaviour)
    const n = cachedSSData.length;
    const used = new Uint8Array(n);
    const order = new Array(n);
    order[0] = 0; used[0] = 1;
    for (let k = 1; k < n; k++) {
        const prev = order[k - 1];
        const px = cachedSSData[prev].x, py = cachedSSData[prev].y;
        let bestDist = Infinity, bestIdx = 0;
        for (let j = 0; j < n; j++) {
            if (used[j]) continue;
            const dx = cachedSSData[j].x - px, dy = cachedSSData[j].y - py;
            const d = dx * dx + dy * dy;
            if (d < bestDist) { bestDist = d; bestIdx = j; }
        }
        order[k] = bestIdx; used[bestIdx] = 1;
    }
    sortedSSIndices = order;
}

function ssAnimLoop() {
    if (!ssAnimPlaying || !cachedSSData || cachedSSData.length === 0) return;
    if (!sortedSSIndices || sortedSSIndices.length !== cachedSSData.length) buildSortedSSIndices();

    ssAnimIdx = (ssAnimIdx + 5) % sortedSSIndices.length;
    const pct = ssAnimIdx / sortedSSIndices.length;
    document.getElementById('ssAnimSlider').value = Math.floor(pct * 1000);

    draw();
    requestAnimationFrame(ssAnimLoop);
}

document.getElementById('ssPlayBtn').addEventListener('click', () => {
    if (!cachedSSData || cachedSSData.length === 0) return;

    ssAnimPlaying = !ssAnimPlaying;
    const btn = document.getElementById('ssPlayBtn');

    if (ssAnimPlaying) {
        btn.innerText = '⏸';
        btn.style.background = 'linear-gradient(135deg, #dc2626, #ef4444)';
        ssAnimLoop();
    } else {
        btn.innerText = '▶';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    }
});

document.getElementById('ssStopBtn').addEventListener('click', stopSSAnim);

document.getElementById('ssAnimSlider').addEventListener('input', e => {
    if (!cachedSSData || cachedSSData.length === 0) return;
    if (!sortedSSIndices) buildSortedSSIndices();
    const pct = parseFloat(e.target.value) / 1000;
    ssAnimIdx = Math.floor(pct * (sortedSSIndices.length - 1));
    draw();
});

// ========== Radius Sweep ==========
async function computeRadiusSweep() {
    if (!vineyardCenter || !cachedCurveData || cachedCurveData.length === 0) {
        updateStatus('Need curve and center!');
        return;
    }
    
    if (!PERSISTENCE_API_URL) {
        updateStatus('Error: PERSISTENCE_API_URL not set!');
        return;
    }
    
    updateStatus('Computing sweep via API...');
    
    const radii = [];
    for (let i = 0; i <= radiusSteps; i++) {
        radii.push(radiusStart + (radiusEnd - radiusStart) * (i / radiusSteps));
    }
    
    radiusSweepData = {
        radii: radii,
        vineyards: [],
        infinityY: 0
    };
    
    let globalInfinityY = 0;
    
    try {
        for (let rIdx = 0; rIdx < radii.length; rIdx++) {
            const r = radii[rIdx];
            
            // Generate centers for this radius
            const centers = [];
            for (let i = 0; i < vineyardSamples; i++) {
                const theta = (i / vineyardSamples) * Math.PI * 2;
                centers.push(new Point(
                    vineyardCenter.x + r * Math.cos(theta),
                    vineyardCenter.y + r * Math.sin(theta)
                ));
            }
            
            // Call API
            const apiResult = await computeVineyardAPI(centers, cachedCurveData);
            
            const infinityY = apiResult.infinityY;
            
            const processEntries = (entries, type) => {
                return (entries || []).map(e => ({
                    birth: e.birth,
                    death: e.death,
                    centerIdx: e.centerIdx,
                    isInfinite: e.isInfinite || false,
                    type: type
                }));
            };
            
            const ord0 = processEntries(apiResult.ord0, 'ord');
            const rel0 = processEntries(apiResult.rel0, 'rel');
            const ext0 = processEntries(apiResult.ext0, 'ext');
            const ord1 = processEntries(apiResult.ord1, 'ord');
            const rel1 = processEntries(apiResult.rel1, 'rel');
            const ext1 = processEntries(apiResult.ext1, 'ext');
            
            const vineyard = {
                h0: [...ord0, ...rel0, ...ext0],
                h1: [...ord1, ...rel1, ...ext1],
                ord0, rel0, ext0, ord1, rel1, ext1,
                centers: centers,
                infinityY: infinityY,
                radius: r
            };
            
            radiusSweepData.vineyards.push(vineyard);
            
            if (infinityY > globalInfinityY) {
                globalInfinityY = infinityY;
            }
            
            if (rIdx % 3 === 0) {
                updateStatus(`Sweep ${Math.round((rIdx / radii.length) * 100)}%`);
            }
        }
        
        radiusSweepData.infinityY = globalInfinityY;
        vineyardMaxVal = globalInfinityY;
        
        updateStatus(`Sweep done! ${radii.length} vineyards`);
        
        radiusSweepIdx = 0;
        updateRadiusSweepDisplay();
        
        // Only show vineyard panel for radius sweep (not persistence diagram)
        document.getElementById('vineyardPanel').classList.add('active');
        document.getElementById('showVineyardPlot').checked = true;
        // Hide persistence panel during radius sweep
        document.getElementById('persistencePanel').classList.remove('active');
        document.getElementById('showPD').checked = false;
        
        draw();
        
    } catch (err) {
        console.error('API error:', err);
        updateStatus(`Error: ${err.message}`);
    }
}

function updateRadiusSweepDisplay() {
    if (!radiusSweepData || !radiusSweepData.vineyards[radiusSweepIdx]) return;
    
    const currentVineyard = radiusSweepData.vineyards[radiusSweepIdx];
    const r = currentVineyard.radius;
    
    document.getElementById('currentSweepRadius').innerText = `R: ${r.toFixed(2)}`;
    
    vineyardRadius = r;
    document.getElementById('vineyardRadius').value = r.toFixed(2);
    
    vineyardCenters = currentVineyard.centers;
    
    vineyardData = {
        h0: currentVineyard.h0,
        h1: currentVineyard.h1,
        ord0: currentVineyard.ord0,
        rel0: currentVineyard.rel0,
        ext0: currentVineyard.ext0,
        ord1: currentVineyard.ord1,
        rel1: currentVineyard.rel1,
        ext1: currentVineyard.ext1,
        infinityY: currentVineyard.infinityY
    };
    vineyardMaxVal = currentVineyard.infinityY;
    
    updateRadiusSweepVineyardPlot();
    // Don't update persistence diagram during radius sweep - only vineyard changes
}

function updateRadiusSweepVineyardPlot() {
    if (!radiusSweepData || !radiusSweepData.vineyards[radiusSweepIdx]) return;
    
    const currentVineyard = radiusSweepData.vineyards[radiusSweepIdx];
    const r = currentVineyard.radius;
    const maxVal = currentVineyard.infinityY;
    
    const ord0 = currentVineyard.ord0 || [];
    const rel0 = currentVineyard.rel0 || [];
    const ext0 = currentVineyard.ext0 || [];
    const ord1 = currentVineyard.ord1 || [];
    const rel1 = currentVineyard.rel1 || [];
    const ext1 = currentVineyard.ext1 || [];
    
    // Six traces for extended persistence types
    const traceOrd0 = {
        x: ord0.map(d => d.birth),
        y: ord0.map(d => d.death),
        z: ord0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ord H₀',
        marker: { size: 3, color: '#ef4444', opacity: 0.7 }
    };

    const traceRel0 = {
        x: rel0.map(d => d.birth),
        y: rel0.map(d => d.death),
        z: rel0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Rel H₀',
        marker: { size: 2.1, color: '#f97316', opacity: 0.7, symbol: 'square' }
    };

    const traceExt0 = {
        x: ext0.map(d => d.birth),
        y: ext0.map(d => d.death),
        z: ext0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ext H₀',
        marker: { size: 2.1, color: '#eab308', opacity: 0.7, symbol: 'diamond' }
    };

    const traceOrd1 = {
        x: ord1.map(d => d.birth),
        y: ord1.map(d => d.death),
        z: ord1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ord H₁',
        marker: { size: 3, color: '#3b82f6', opacity: 0.7 }
    };

    const traceRel1 = {
        x: rel1.map(d => d.birth),
        y: rel1.map(d => d.death),
        z: rel1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Rel H₁',
        marker: { size: 2.1, color: '#06b6d4', opacity: 0.7, symbol: 'square' }
    };

    const traceExt1 = {
        x: ext1.map(d => d.birth),
        y: ext1.map(d => d.death),
        z: ext1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ext H₁',
        marker: { size: 2.1, color: '#a855f7', opacity: 0.7, symbol: 'diamond' }
    };

    const diagPoints = {
        x: [0, maxVal, maxVal, 0],
        y: [0, maxVal, maxVal, 0],
        z: [0, 0, vineyardSamples - 1, vineyardSamples - 1],
        i: [0, 0],
        j: [1, 2],
        k: [2, 3],
        type: 'mesh3d',
        name: 'Diagonal',
        color: 'rgba(100,100,100,0.1)',
        showscale: false,
        hoverinfo: 'skip'
    };
    
    const layout = {
        scene: {
            xaxis: { title: { text: 'Birth', font: { size: 10 } }, color: '#888', gridcolor: '#333', range: [0, maxVal], showspikes: false },
            yaxis: { title: { text: 'Death', font: { size: 10 } }, color: '#888', gridcolor: '#333', range: [0, maxVal], showspikes: false },
            zaxis: { title: { text: 'Time', font: { size: 10 } }, color: '#888', gridcolor: '#333', range: [0, vineyardSamples - 1], showspikes: false },
            bgcolor: 'rgba(0,0,0,0)',
            camera: { eye: { x: 1.6, y: 1.6, z: 1.0 } },
            aspectmode: 'cube'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 0, r: 0, t: 20, b: 0 },
        showlegend: false,
        title: { text: `R = ${r.toFixed(2)}`, font: { color: '#c084fc', size: 11 }, x: 0.5 }
    };
    
    Plotly.newPlot('vineyardPlot', [diagPoints, traceOrd0, traceRel0, traceExt0, traceOrd1, traceRel1, traceExt1], layout, { displayModeBar: false, responsive: true });
}

function updateRadiusSweepPersistenceDiagram() {
    if (!radiusSweepData || !radiusSweepData.vineyards[radiusSweepIdx]) return;
    
    const currentVineyard = radiusSweepData.vineyards[radiusSweepIdx];
    const r = currentVineyard.radius;
    const maxVal = currentVineyard.infinityY;
    
    const h0 = currentVineyard.h0;
    const h1 = currentVineyard.h1;
    
    const traceH0 = {
        x: h0.map(d => d.birth),
        y: h0.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'H₀',
        marker: { size: 6, color: '#ef4444', symbol: 'circle', opacity: 0.6 }
    };

    const traceH1 = {
        x: h1.map(d => d.birth),
        y: h1.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'H₁',
        marker: { size: 6, color: '#3b82f6', symbol: 'triangle-up', opacity: 0.6 }
    };
    
    const diagonal = {
        x: [0, maxVal],
        y: [0, maxVal],
        mode: 'lines',
        type: 'scatter',
        name: 'Diagonal',
        line: { color: '#555', dash: 'dash', width: 1 },
        showlegend: false,
        hoverinfo: 'skip'
    };
    
    const infinityLine = {
        x: [0, maxVal],
        y: [currentVineyard.infinityY, currentVineyard.infinityY],
        mode: 'lines',
        type: 'scatter',
        name: '∞',
        line: { color: '#888', dash: 'dot', width: 1 },
        showlegend: false,
        hoverinfo: 'skip'
    };
    
    const layout = {
        xaxis: { title: { text: 'Birth', font: { size: 11, color: '#aaa' } }, color: '#888', gridcolor: '#333', range: [-maxVal * 0.02, maxVal * 1.02], zeroline: true, zerolinecolor: '#666' },
        yaxis: { title: { text: 'Death', font: { size: 11, color: '#aaa' } }, color: '#888', gridcolor: '#333', range: [-maxVal * 0.02, maxVal * 1.02], zeroline: true, zerolinecolor: '#666', scaleanchor: 'x', scaleratio: 1 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(10,10,15,0.5)',
        margin: { l: 45, r: 10, t: 25, b: 40 },
        legend: { x: 0.75, y: 0.15, font: { color: '#ccc', size: 9 }, bgcolor: 'rgba(0,0,0,0.6)' },
        showlegend: true,
        title: { text: `All PDs R=${r.toFixed(2)}`, font: { color: '#aaa', size: 11 }, x: 0.5 }
    };
    
    Plotly.newPlot('persistencePlot', [diagonal, infinityLine, traceH0, traceH1], layout, { displayModeBar: false, responsive: true });
}

function stopRadiusSweep() {
    radiusSweepPlaying = false;
    document.getElementById('playRadiusSweepBtn').innerText = '▶';
    document.getElementById('playRadiusSweepBtn').style.background = '';
}

function radiusSweepAnimLoop() {
    if (!radiusSweepPlaying || !radiusSweepData) return;
    
    radiusSweepIdx = (radiusSweepIdx + 1) % radiusSweepData.radii.length;
    const pct = radiusSweepIdx / (radiusSweepData.radii.length - 1);
    document.getElementById('radiusSweepSlider').value = Math.floor(pct * 1000);
    
    updateRadiusSweepDisplay();
    draw();
    
    setTimeout(radiusSweepAnimLoop, 200);
}

document.getElementById('radiusStart').addEventListener('input', e => {
    radiusStart = parseFloat(e.target.value) || 0.5;
    if (vineyardCenter) draw();
});

document.getElementById('radiusStart').addEventListener('change', e => {
    radiusStart = parseFloat(e.target.value) || 0.5;
    if (radiusStart <= 0) {
        radiusStart = 0.1;
        e.target.value = 0.1;
    }
    if (vineyardCenter) draw();
});

document.getElementById('radiusEnd').addEventListener('input', e => {
    radiusEnd = parseFloat(e.target.value) || 5;
    if (vineyardCenter) draw();
});

document.getElementById('radiusEnd').addEventListener('change', e => {
    radiusEnd = parseFloat(e.target.value) || 5;
    if (radiusEnd <= 0) {
        radiusEnd = 0.1;
        e.target.value = 0.1;
    }
    if (vineyardCenter) draw();
});

document.getElementById('radiusSteps').addEventListener('input', e => {
    radiusSteps = parseInt(e.target.value);
    document.getElementById('radiusStepsVal').innerText = radiusSteps;
});

document.getElementById('computeRadiusSweepBtn').addEventListener('click', computeRadiusSweep);

document.getElementById('playRadiusSweepBtn').addEventListener('click', () => {
    if (!radiusSweepData) return;
    
    radiusSweepPlaying = !radiusSweepPlaying;
    const btn = document.getElementById('playRadiusSweepBtn');
    
    if (radiusSweepPlaying) {
        btn.innerText = '⏸';
        btn.style.background = 'linear-gradient(135deg, #dc2626, #ef4444)';
        radiusSweepAnimLoop();
    } else {
        btn.innerText = '▶';
        btn.style.background = '';
    }
});

document.getElementById('stopRadiusSweepBtn').addEventListener('click', () => {
    stopRadiusSweep();
    radiusSweepIdx = 0;
    document.getElementById('radiusSweepSlider').value = 0;
    if (radiusSweepData) {
        updateRadiusSweepDisplay();
        draw();
    }
});

document.getElementById('radiusSweepSlider').addEventListener('input', e => {
    if (!radiusSweepData) return;
    const pct = parseFloat(e.target.value) / 1000;
    radiusSweepIdx = Math.floor(pct * (radiusSweepData.radii.length - 1));
    updateRadiusSweepDisplay();
    draw();
});

