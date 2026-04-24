// ========== Persistence Computation via Python API ==========
// All persistence is computed by the GUDHI-based Python server

// Call Python API for vineyard computation
async function computeVineyardAPI(centers, curveData) {
    const response = await fetch(`${PERSISTENCE_API_URL}/vineyard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            centers: centers.map(c => ({ x: c.x, y: c.y })),
            points: curveData.map(d => ({ x: d.p.x, y: d.p.y, curveId: d.curveId || 0 })),
            use_squared_distance: true
        })
    });
    
    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API error ${response.status}: ${errText}`);
    }
    
    return await response.json();
}

async function computeVineyard() {
    if (!cachedCurveData || cachedCurveData.length === 0) {
        updateStatus('Need a curve first!');
        return;
    }
    
    if (!PERSISTENCE_API_URL) {
        updateStatus('Error: PERSISTENCE_API_URL not set! Edit the HTML to set your server URL.');
        return;
    }
    
    let centers = [];
    
    if (vineyardLoopType === 'circular') {
        if (!vineyardCenter) {
            updateStatus('Need center point! Click "Place Center" first.');
            return;
        }
        
        // Generate circular centers
        for (let i = 0; i < vineyardSamples; i++) {
            const theta = (i / vineyardSamples) * Math.PI * 2;
            centers.push(new Point(
                vineyardCenter.x + vineyardRadius * Math.cos(theta),
                vineyardCenter.y + vineyardRadius * Math.sin(theta)
            ));
        }
    } else {
        // Custom loop mode
        if (customLoopPoints.length < 3) {
            updateStatus('Need at least 3 loop points!');
            return;
        }
        
        // Sample points along the loop
        const loopSamples = sampleCustomLoop(vineyardSamples * 10);
        const step = Math.max(1, Math.floor(loopSamples.length / vineyardSamples));
        for (let i = 0; i < loopSamples.length; i += step) {
            centers.push(loopSamples[i]);
        }
    }
    
    vineyardCenters = centers;
    
    updateStatus(`Computing vineyard via API (${centers.length} centers, ${cachedCurveData.length} points)...`);
    
    try {
        const apiResult = await computeVineyardAPI(centers, cachedCurveData);
        
        // Process API result
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
        
        vineyardData = {
            h0: [...ord0, ...rel0, ...ext0],
            h1: [...ord1, ...rel1, ...ext1],
            ord0, rel0, ext0, ord1, rel1, ext1,
            infinityY: infinityY
        };
        vineyardMaxVal = infinityY;
        
        const totalPts = vineyardData.h0.length + vineyardData.h1.length;
        const ordCount = ord0.length + ord1.length;
        const relCount = rel0.length + rel1.length;
        const extCount = ext0.length + ext1.length;
        updateStatus(`Vineyard: ${totalPts} pts (Ord: ${ordCount}, Rel: ${relCount}, Ext: ${extCount})`);
        
        // Reset any prior inline position/size from dragging or resizing so the
        // panels return to their CSS default layout, then activate BEFORE plotting
        // so Plotly measures the real container size (hidden containers are 0x0).
        const vPanel = document.getElementById('vineyardPanel');
        const pPanel = document.getElementById('persistencePanel');
        [vPanel, pPanel].forEach(panel => {
            panel.style.position = '';
            panel.style.left = '';
            panel.style.top = '';
            panel.style.right = '';
            panel.style.bottom = '';
            panel.style.width = '';
            panel.style.height = '';
        });
        vPanel.classList.add('active');
        pPanel.classList.add('active');

        updateVineyardPlot();
        updatePersistenceDiagram(0);

        draw();
        
    } catch (err) {
        console.error('API error:', err);
        updateStatus(`Error: ${err.message}. Is the Python server running?`);
    }
}

function updateVineyardPlot() {
    if (!vineyardData) return;
    
    const ord0 = vineyardData.ord0 || [];
    const rel0 = vineyardData.rel0 || [];
    const ext0 = vineyardData.ext0 || [];
    const ord1 = vineyardData.ord1 || [];
    const rel1 = vineyardData.rel1 || [];
    const ext1 = vineyardData.ext1 || [];
    const h0 = vineyardData.h0;
    const h1 = vineyardData.h1;
    const maxVal = vineyardMaxVal;
    
    // Ordinary H0 - solid red
    const traceOrd0 = {
        x: ord0.map(d => d.birth),
        y: ord0.map(d => d.death),
        z: ord0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ord H₀',
        marker: {
            size: 3,
            color: '#ef4444',
            opacity: 0.7
        }
    };

    // Relative H0 - orange squares
    const traceRel0 = {
        x: rel0.map(d => d.birth),
        y: rel0.map(d => d.death),
        z: rel0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Rel H₀',
        marker: {
            size: 2.1,
            color: '#f97316',  // Orange for relative
            opacity: 0.7,
            symbol: 'square'
        }
    };

    // Extended H0 - yellow diamonds
    const traceExt0 = {
        x: ext0.map(d => d.birth),
        y: ext0.map(d => d.death),
        z: ext0.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ext H₀',
        marker: {
            size: 2.1,
            color: '#eab308',  // Yellow for extended
            opacity: 0.7,
            symbol: 'diamond'
        }
    };

    // Ordinary H1 - solid blue
    const traceOrd1 = {
        x: ord1.map(d => d.birth),
        y: ord1.map(d => d.death),
        z: ord1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ord H₁',
        marker: {
            size: 3,
            color: '#3b82f6',
            opacity: 0.7
        }
    };

    // Relative H1 - cyan squares
    const traceRel1 = {
        x: rel1.map(d => d.birth),
        y: rel1.map(d => d.death),
        z: rel1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Rel H₁',
        marker: {
            size: 2.1,
            color: '#06b6d4',  // Cyan for relative
            opacity: 0.7,
            symbol: 'square'
        }
    };

    // Extended H1 - purple diamonds
    const traceExt1 = {
        x: ext1.map(d => d.birth),
        y: ext1.map(d => d.death),
        z: ext1.map(d => d.centerIdx),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Ext H₁',
        marker: {
            size: 2.1,
            color: '#a855f7',  // Purple for extended
            opacity: 0.7,
            symbol: 'diamond'
        }
    };

    const currentZ = vineyardAnimIdx;
    const currentH0 = h0.filter(d => d.centerIdx === currentZ);
    const currentH1 = h1.filter(d => d.centerIdx === currentZ);
    
    const currentTrace = {
        x: [...currentH0.map(d => d.birth), ...currentH1.map(d => d.birth)],
        y: [...currentH0.map(d => d.death), ...currentH1.map(d => d.death)],
        z: [...currentH0.map(d => d.centerIdx), ...currentH1.map(d => d.centerIdx)],
        mode: 'markers',
        type: 'scatter3d',
        name: 'Current',
        marker: {
            size: 6,
            color: '#fbbf24',
            opacity: 1,
            line: { color: 'white', width: 1 }
        }
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
        color: 'rgba(100,100,100,0.15)',
        showscale: false,
        hoverinfo: 'skip'
    };
    
    const layout = {
        scene: {
            xaxis: { 
                title: { text: 'Birth', font: { size: 10 } },
                color: '#888', 
                gridcolor: '#333',
                range: [0, maxVal],
                showspikes: false
            },
            yaxis: { 
                title: { text: 'Death', font: { size: 10 } },
                color: '#888', 
                gridcolor: '#333',
                range: [0, maxVal],
                showspikes: false
            },
            zaxis: { 
                title: { text: 'Time', font: { size: 10 } },
                color: '#888', 
                gridcolor: '#333',
                range: [0, vineyardSamples - 1],
                showspikes: false
            },
            bgcolor: 'rgba(0,0,0,0)',
            camera: { eye: { x: 1.6, y: 1.6, z: 1.0 } },
            aspectmode: 'cube'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 0, r: 0, t: 5, b: 0 },
        showlegend: false
    };
    
    Plotly.newPlot('vineyardPlot', [diagPoints, traceOrd0, traceRel0, traceExt0, traceOrd1, traceRel1, traceExt1, currentTrace], layout, {
        displayModeBar: false,
        responsive: true
    });
}

function updatePersistenceDiagram(centerIdx) {
    if (!vineyardData || !vineyardCenters[centerIdx]) return;
    
    // Get extended persistence data for current center
    const ord0 = (vineyardData.ord0 || []).filter(d => d.centerIdx === centerIdx);
    const rel0 = (vineyardData.rel0 || []).filter(d => d.centerIdx === centerIdx);
    const ext0 = (vineyardData.ext0 || []).filter(d => d.centerIdx === centerIdx);
    const ord1 = (vineyardData.ord1 || []).filter(d => d.centerIdx === centerIdx);
    const rel1 = (vineyardData.rel1 || []).filter(d => d.centerIdx === centerIdx);
    const ext1 = (vineyardData.ext1 || []).filter(d => d.centerIdx === centerIdx);
    
    const h0 = vineyardData.h0.filter(d => d.centerIdx === centerIdx);
    const h1 = vineyardData.h1.filter(d => d.centerIdx === centerIdx);
    
    currentPersistence = { h0, h1, ord0, rel0, ext0, ord1, rel1, ext1 };
    
    const maxVal = vineyardMaxVal;
    
    // Ordinary H0 - red circles
    const traceOrd0 = {
        x: ord0.map(d => d.birth),
        y: ord0.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Ord H₀',
        marker: { size: 8, color: '#ef4444', symbol: 'circle' }
    };

    // Relative H0 - orange squares
    const traceRel0 = {
        x: rel0.map(d => d.birth),
        y: rel0.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Rel H₀',
        marker: { size: 10, color: '#f97316', symbol: 'square' }
    };

    // Extended H0 - yellow diamonds
    const traceExt0 = {
        x: ext0.map(d => d.birth),
        y: ext0.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Ext H₀',
        marker: { size: 12, color: '#eab308', symbol: 'diamond' }
    };

    // Ordinary H1 - blue triangles
    const traceOrd1 = {
        x: ord1.map(d => d.birth),
        y: ord1.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Ord H₁',
        marker: { size: 8, color: '#3b82f6', symbol: 'triangle-up' }
    };

    // Relative H1 - cyan squares
    const traceRel1 = {
        x: rel1.map(d => d.birth),
        y: rel1.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Rel H₁',
        marker: { size: 10, color: '#06b6d4', symbol: 'square' }
    };

    // Extended H1 - purple diamonds
    const traceExt1 = {
        x: ext1.map(d => d.birth),
        y: ext1.map(d => d.death),
        mode: 'markers',
        type: 'scatter',
        name: 'Ext H₁',
        marker: { size: 12, color: '#a855f7', symbol: 'diamond' }
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
    
    const infinityY = vineyardData.infinityY;
    const infinityLine = {
        x: [0, maxVal],
        y: [infinityY, infinityY],
        mode: 'lines',
        type: 'scatter',
        name: '∞',
        line: { color: '#888', dash: 'dot', width: 1 },
        showlegend: false,
        hoverinfo: 'skip'
    };
    
    const layout = {
        xaxis: { 
            title: { text: 'Birth', font: { size: 11, color: '#aaa' } },
            color: '#888', 
            gridcolor: '#333',
            range: [-maxVal * 0.02, maxVal * 1.02],
            zeroline: true,
            zerolinecolor: '#666'
        },
        yaxis: { 
            title: { text: 'Death', font: { size: 11, color: '#aaa' } },
            color: '#888', 
            gridcolor: '#333',
            range: [-maxVal * 0.02, maxVal * 1.02],
            zeroline: true,
            zerolinecolor: '#666',
            scaleanchor: 'x',
            scaleratio: 1
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(10,10,15,0.5)',
        margin: { l: 45, r: 10, t: 25, b: 40 },
        legend: {
            x: 0.60, y: 0.35,
            font: { color: '#ccc', size: 8 },
            bgcolor: 'rgba(0,0,0,0.6)'
        },
        showlegend: true,
        title: {
            text: `Center ${centerIdx}`,
            font: { color: '#aaa', size: 11 },
            x: 0.5
        }
    };
    
    Plotly.newPlot('persistencePlot', [diagonal, infinityLine, traceOrd0, traceRel0, traceExt0, traceOrd1, traceRel1, traceExt1], layout, {
        displayModeBar: false,
        responsive: true
    });
}

