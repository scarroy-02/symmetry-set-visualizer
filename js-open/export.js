// ========== Export Functionality ==========
const exportMenu = document.getElementById('exportMenu');
const exportBtn = document.getElementById('exportBtn');

exportBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const rect = exportBtn.getBoundingClientRect();
    exportMenu.style.top = (rect.bottom + 4) + 'px';
    exportMenu.style.left = rect.left + 'px';
    exportMenu.classList.toggle('active');
});

document.addEventListener('click', (e) => {
    if (!exportMenu.contains(e.target) && e.target !== exportBtn) {
        exportMenu.classList.remove('active');
    }
});

function downloadDataURL(dataURL, filename) {
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataURL;
    link.click();
}

const exportLineThicknessInput = document.getElementById('exportLineThickness');
const exportLineThicknessVal = document.getElementById('exportLineThicknessVal');
if (exportLineThicknessInput && exportLineThicknessVal) {
    exportLineThicknessInput.addEventListener('input', () => {
        exportLineThicknessVal.textContent = parseFloat(exportLineThicknessInput.value).toFixed(1).replace(/\.0$/, '') + '×';
        if (typeof draw === 'function') draw();
    });
}
function getExportLineThickness() {
    return exportLineThicknessInput ? (parseFloat(exportLineThicknessInput.value) || 1) : 1;
}

function exportCanvasAsPNG() {
    const t = getExportLineThickness();
    // Create a SQUARE high-res export canvas with LIGHT theme
    const exportSize = Math.min(canvas.width, canvas.height);
    const scale = 2; // 2x resolution for better quality
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = exportSize * scale;
    exportCanvas.height = exportSize * scale;
    const exportCtx = exportCanvas.getContext('2d');

    // Scale and redraw
    exportCtx.scale(scale, scale);

    // Draw WHITE background for papers
    exportCtx.fillStyle = '#ffffff';
    exportCtx.fillRect(0, 0, exportSize, exportSize);

    exportCtx.save();
    exportCtx.translate(exportSize / 2 + view.x, exportSize / 2 + view.y);
    exportCtx.scale(view.scale, -view.scale);

    // NO GRID for clean export

    if (cachedCurveData && cachedCurveData.length > 0) {
        // Focal set - darker green for visibility
        if (document.getElementById('showFocal').checked && cachedFocalData) {
            exportCtx.strokeStyle = '#7c3aed';
            exportCtx.lineWidth = 1.5 * t / view.scale;
            exportCtx.lineJoin = 'round';
            exportCtx.lineCap = 'round';
            for (const branch of cachedFocalData.branches) {
                if (branch.length < 2) continue;
                exportCtx.beginPath();
                exportCtx.moveTo(branch[0].x, branch[0].y);
                for (let k = 1; k < branch.length; k++) exportCtx.lineTo(branch[k].x, branch[k].y);
                if (branch.closed) exportCtx.closePath();
                exportCtx.stroke();
            }
        }

        if (document.getElementById('showSS').checked && cachedSSData) {
            if (cachedSSData.branches && cachedSSData.branches.length > 0) {
                exportCtx.strokeStyle = '#3b82f6';
                exportCtx.lineWidth = 1.5 * t / view.scale;
                exportCtx.lineJoin = 'round';
                exportCtx.lineCap = 'round';
                for (const branch of cachedSSData.branches) {
                    if (branch.length < 2) continue;
                    exportCtx.beginPath();
                    exportCtx.moveTo(branch[0].x, branch[0].y);
                    for (let k = 1; k < branch.length; k++) exportCtx.lineTo(branch[k].x, branch[k].y);
                    exportCtx.stroke();
                }
            } else {
                exportCtx.fillStyle = '#3b82f6';
                const s = 1.5 / view.scale;
                for (const p of cachedSSData) exportCtx.fillRect(p.x - s/2, p.y - s/2, s, s);
            }
        }

        // Curve - darker blue
        if (document.getElementById('showCurve').checked) {
            exportCtx.strokeStyle = '#171717';
            exportCtx.lineWidth = 2.5 * t / view.scale;
            let currentId = -1;
            exportCtx.beginPath();
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (currentId !== -1) { if (!curveOpen[currentId]) exportCtx.closePath(); exportCtx.stroke(); exportCtx.beginPath(); }
                    exportCtx.moveTo(pt.p.x, pt.p.y);
                    currentId = pt.curveId;
                } else {
                    exportCtx.lineTo(pt.p.x, pt.p.y);
                }
            }
            if (currentId !== -1) { if (!curveOpen[currentId]) exportCtx.closePath(); exportCtx.stroke(); }
        }
    }

    // Vineyard circle - darker purple
    if (document.getElementById('showVineyardCircle').checked) {
        // Custom loop spline (closed loop or open arc)
        if (customLoopPoints.length >= 3) {
            const loopSamples = sampleCustomLoop(200);
            if (loopSamples.length > 0) {
                exportCtx.strokeStyle = 'rgba(127, 29, 29, 0.8)';
                exportCtx.lineWidth = 2 * t / view.scale;
                exportCtx.setLineDash([]);
                exportCtx.beginPath();
                exportCtx.moveTo(loopSamples[0].x, loopSamples[0].y);
                for (let i = 1; i < loopSamples.length; i++) {
                    exportCtx.lineTo(loopSamples[i].x, loopSamples[i].y);
                }
                if (!customLoopOpen) exportCtx.closePath();
                exportCtx.stroke();
            }
        }

        if (vineyardCenter) {
            exportCtx.fillStyle = '#7f1d1d';
            exportCtx.beginPath();
            exportCtx.arc(vineyardCenter.x, vineyardCenter.y, 6 / view.scale, 0, Math.PI * 2);
            exportCtx.fill();
            
            exportCtx.strokeStyle = 'rgba(127, 29, 29, 0.6)';
            exportCtx.lineWidth = 1.5 * t / view.scale;
            exportCtx.setLineDash([5 / view.scale, 5 / view.scale]);
            exportCtx.beginPath();
            exportCtx.arc(vineyardCenter.x, vineyardCenter.y, vineyardRadius, 0, Math.PI * 2);
            exportCtx.stroke();
            exportCtx.setLineDash([]);
        }
        
        if (vineyardCenters.length > 0) {
            exportCtx.fillStyle = 'rgba(127, 29, 29, 0.5)';
            const s = 3 / view.scale;
            for (const c of vineyardCenters) {
                exportCtx.fillRect(c.x - s/2, c.y - s/2, s, s);
            }
        }
    }

    // Current vineyard position marker — independent of the Observation Loop toggle
    if (vineyardCenters.length > 0 && vineyardAnimIdx < vineyardCenters.length) {
        const curr = vineyardCenters[vineyardAnimIdx];
        exportCtx.fillStyle = '#f59e0b';
        exportCtx.beginPath();
        exportCtx.arc(curr.x, curr.y, 8 / view.scale, 0, Math.PI * 2);
        exportCtx.fill();
    }

    // Birth/Death circles — same as on-canvas, independent of the Observation Loop toggle
    drawBirthDeathCircles(exportCtx);

    // Control points - dark gray/black
    if (document.getElementById('showControls').checked) {
        const baseSize = 7 / view.scale;
        for (let c = 0; c < curves.length; c++) {
            const pts = curves[c];
            for (let i = 0; i < pts.length; i++) {
                const p = pts[i];
                exportCtx.fillStyle = (c === activeCurveIdx) ? '#1a1a1a' : '#888';
                exportCtx.fillRect(p.x - baseSize/2, p.y - baseSize/2, baseSize, baseSize);
            }
        }
    }
    
    exportCtx.restore();
    
    const dataURL = exportCanvas.toDataURL('image/png');
    downloadDataURL(dataURL, 'symmetry_canvas.png');
    exportMenu.classList.remove('active');
}

// Persistence-diagram export: temporarily fatten the markers and the diagonal/
// infinity reference lines (these are trace props, not layout) so they read clearly
// at print resolution. Returns the saved originals; pass them to the restore helper.
function thickenPersistenceForExport(plotId) {
    if (plotId !== 'persistencePlot') return null;
    const gd = document.getElementById(plotId);
    if (!gd || !gd.data) return null;

    const markerIdx = [], markerSizes = [];
    const lineIdx = [], lineWidths = [];
    gd.data.forEach((t, i) => {
        if (t.marker && t.marker.size != null) { markerIdx.push(i); markerSizes.push(t.marker.size); }
        if (t.line && t.line.width != null) { lineIdx.push(i); lineWidths.push(t.line.width); }
    });
    if (markerIdx.length) Plotly.restyle(plotId, { 'marker.size': markerSizes.map(s => s * 1.8) }, markerIdx);
    if (lineIdx.length) Plotly.restyle(plotId, { 'line.width': lineWidths.map(w => Math.max(3.5, w * 3.5)) }, lineIdx);
    return { markerIdx, markerSizes, lineIdx, lineWidths };
}

function restorePersistenceAfterExport(plotId, saved) {
    if (!saved) return;
    if (saved.markerIdx.length) Plotly.restyle(plotId, { 'marker.size': saved.markerSizes }, saved.markerIdx);
    if (saved.lineIdx.length) Plotly.restyle(plotId, { 'line.width': saved.lineWidths }, saved.lineIdx);
}

function exportPlotlyAsPNG(plotId, filename) {
    const plotDiv = document.getElementById(plotId);
    // Read the CURRENT 3D camera (includes any interactive rotation). _fullLayout
    // reflects live camera changes; plotDiv.layout.scene.camera can be stale, which
    // is what made the export snap back to the plot's initial orientation.
    const rawCamera = (plotDiv._fullLayout && plotDiv._fullLayout.scene && plotDiv._fullLayout.scene.camera)
        || (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera);
    const savedCamera = rawCamera ? JSON.parse(JSON.stringify(rawCamera)) : undefined;

    // Export as square with good quality
    const exportSize = 800;
    
    const lightLayout = {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: plotId === 'persistencePlot' ? { family: 'Times New Roman, Times, serif', color: '#1a1a1a', size: 14 } : { color: '#1a1a1a' },
        showlegend: plotId === 'persistencePlot',
        legend: plotId === 'persistencePlot' ? { x: 0.98, y: 0.02, xanchor: 'right', yanchor: 'bottom', font: { family: 'Times New Roman, Times, serif', color: '#1a1a1a', size: 18 }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#333333', borderwidth: 1 } : undefined,
        scene: plotId === 'vineyardPlot' ? {
            xaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Birth', font: { color: '#333' } } },
            yaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Death', font: { color: '#333' } } },
            zaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Time', font: { color: '#333' } } },
            bgcolor: '#ffffff',
            camera: savedCamera // preserve the current (interactive) orientation
        } : undefined,
        xaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', linecolor: '#333', linewidth: 3, tickwidth: 2, ticklen: 7, zerolinewidth: 2.5, gridwidth: 1.5, showline: false, automargin: true, tickfont: { family: 'Times New Roman, Times, serif', size: 14, color: '#333' }, title: { text: 'Birth', font: { family: 'Times New Roman, Times, serif', size: 20, color: '#333' }, standoff: 30 } } : undefined,
        yaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', linecolor: '#333', linewidth: 3, tickwidth: 2, ticklen: 7, zerolinewidth: 2.5, gridwidth: 1.5, showline: false, automargin: true, tickfont: { family: 'Times New Roman, Times, serif', size: 14, color: '#333' }, title: { text: 'Death', font: { family: 'Times New Roman, Times, serif', size: 20, color: '#333' }, standoff: 30 } } : undefined,
        title: { font: { color: '#333' } }
    };
    
    const savedStyle = thickenPersistenceForExport(plotId);

    Plotly.relayout(plotId, lightLayout).then(() => {
        return Plotly.toImage(plotId, { format: 'png', width: exportSize, height: exportSize, scale: 2 });
    }).then(dataURL => {
        restorePersistenceAfterExport(plotId, savedStyle);
        downloadDataURL(dataURL, filename);
        exportMenu.classList.remove('active');
    });
}

function exportVineyardAsOBJ() {
    if (!vineyardData) {
        alert('No vineyard data to export');
        return;
    }
    
    const h0 = vineyardData.h0;
    const h1 = vineyardData.h1;
    const maxVal = vineyardMaxVal;
    
    // Normalize coordinates for better 3D software compatibility
    const scale = 1.0 / Math.max(maxVal, vineyardSamples);
    
    let obj = '# Vineyard 3D Export\n';
    obj += '# Generated by Symmetry Set Visualizer\n';
    obj += `# H0 points: ${h0.length}, H1 points: ${h1.length}\n`;
    obj += `# Axes: X=Birth, Y=Death, Z=Time\n\n`;
    
    // Material file reference
    obj += 'mtllib vineyard.mtl\n\n';
    
    // H0 points (red)
    obj += '# H0 points (red)\n';
    obj += 'g H0\n';
    obj += 'usemtl H0_material\n';
    for (const pt of h0) {
        const x = pt.birth * scale;
        const y = pt.death * scale;
        const z = pt.centerIdx * scale;
        obj += `v ${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}\n`;
    }
    
    // H1 points (blue)
    obj += '\n# H1 points (blue)\n';
    obj += 'g H1\n';
    obj += 'usemtl H1_material\n';
    for (const pt of h1) {
        const x = pt.birth * scale;
        const y = pt.death * scale;
        const z = pt.centerIdx * scale;
        obj += `v ${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}\n`;
    }
    
    // Add point elements (some software needs explicit point definitions)
    obj += '\n# Point elements\n';
    const totalPoints = h0.length + h1.length;
    for (let i = 1; i <= totalPoints; i++) {
        obj += `p ${i}\n`;
    }
    
    // Create MTL file content
    let mtl = '# Vineyard Materials\n\n';
    mtl += 'newmtl H0_material\n';
    mtl += 'Kd 0.937 0.267 0.267\n';  // Red (#ef4444)
    mtl += 'Ka 0.1 0.1 0.1\n';
    mtl += 'd 1.0\n\n';
    
    mtl += 'newmtl H1_material\n';
    mtl += 'Kd 0.231 0.510 0.965\n';  // Blue (#3b82f6)
    mtl += 'Ka 0.1 0.1 0.1\n';
    mtl += 'd 1.0\n';
    
    // Download OBJ file
    const objBlob = new Blob([obj], { type: 'text/plain' });
    const objUrl = URL.createObjectURL(objBlob);
    downloadDataURL(objUrl, 'vineyard.obj');
    URL.revokeObjectURL(objUrl);
    
    // Download MTL file
    setTimeout(() => {
        const mtlBlob = new Blob([mtl], { type: 'text/plain' });
        const mtlUrl = URL.createObjectURL(mtlBlob);
        downloadDataURL(mtlUrl, 'vineyard.mtl');
        URL.revokeObjectURL(mtlUrl);
    }, 100);
    
    exportMenu.classList.remove('active');
}

function exportVineyardAsPLY() {
    if (!vineyardData) {
        alert('No vineyard data to export');
        return;
    }
    
    const h0 = vineyardData.h0;
    const h1 = vineyardData.h1;
    const maxVal = vineyardMaxVal;
    
    // Normalize coordinates
    const scale = 1.0 / Math.max(maxVal, vineyardSamples);
    
    const totalPoints = h0.length + h1.length;
    
    // PLY header
    let ply = 'ply\n';
    ply += 'format ascii 1.0\n';
    ply += 'comment Vineyard 3D Export\n';
    ply += 'comment Generated by Symmetry Set Visualizer\n';
    ply += 'comment Axes: X=Birth, Y=Death, Z=Time\n';
    ply += `element vertex ${totalPoints}\n`;
    ply += 'property float x\n';
    ply += 'property float y\n';
    ply += 'property float z\n';
    ply += 'property uchar red\n';
    ply += 'property uchar green\n';
    ply += 'property uchar blue\n';
    ply += 'property uchar alpha\n';
    ply += 'end_header\n';
    
    // H0 points (red: #ef4444 = 239, 68, 68)
    for (const pt of h0) {
        const x = pt.birth * scale;
        const y = pt.death * scale;
        const z = pt.centerIdx * scale;
        ply += `${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)} 239 68 68 255\n`;
    }
    
    // H1 points (blue: #3b82f6 = 59, 130, 246)
    for (const pt of h1) {
        const x = pt.birth * scale;
        const y = pt.death * scale;
        const z = pt.centerIdx * scale;
        ply += `${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)} 59 130 246 255\n`;
    }
    
    // Download PLY file
    const blob = new Blob([ply], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    downloadDataURL(url, 'vineyard.ply');
    URL.revokeObjectURL(url);
    
    exportMenu.classList.remove('active');
}

function exportAllAsPNG() {
    exportCanvasAsPNG();
    
    setTimeout(() => {
        if (vineyardData) {
            Plotly.toImage('vineyardPlot', { format: 'png', width: 800, height: 600, scale: 2 })
                .then(dataURL => downloadDataURL(dataURL, 'vineyard_3d.png'));
        }
    }, 500);
    
    setTimeout(() => {
        if (vineyardData) {
            Plotly.toImage('persistencePlot', { format: 'png', width: 800, height: 600, scale: 2 })
                .then(dataURL => downloadDataURL(dataURL, 'persistence_diagram.png'));
        }
    }, 1000);
    
    exportMenu.classList.remove('active');
}

document.getElementById('exportCanvas').addEventListener('click', exportCanvasAsPNG);
document.getElementById('exportVineyard').addEventListener('click', () => {
    if (!vineyardData) { alert('No vineyard data to export'); return; }
    exportPlotlyAsPNG('vineyardPlot', 'vineyard_3d.png');
});
document.getElementById('exportVineyardOBJ').addEventListener('click', exportVineyardAsOBJ);
document.getElementById('exportVineyardPLY').addEventListener('click', exportVineyardAsPLY);
document.getElementById('exportPD').addEventListener('click', () => {
    if (!vineyardData) { alert('No persistence data to export'); return; }
    exportPlotlyAsPNG('persistencePlot', 'persistence_diagram.png');
});
document.getElementById('exportAllPNG').addEventListener('click', exportAllAsPNG);

