// ========== Theme Toggle ==========
let isDarkTheme = false;

// Apply light theme by default
document.body.classList.add('light-theme');

function toggleTheme() {
    isDarkTheme = !isDarkTheme;
    document.body.classList.toggle('light-theme', !isDarkTheme);
    document.getElementById('themeToggleBtn').innerText = isDarkTheme ? '🌙' : '☀️';
    draw();
}

document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);

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

function exportCanvasAsPNG() {
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
            exportCtx.lineWidth = 1.5 / view.scale;
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
                exportCtx.lineWidth = 1.5 / view.scale;
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
            exportCtx.lineWidth = 2.5 / view.scale;
            let currentId = -1;
            exportCtx.beginPath();
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); exportCtx.beginPath(); }
                    exportCtx.moveTo(pt.p.x, pt.p.y);
                    currentId = pt.curveId;
                } else {
                    exportCtx.lineTo(pt.p.x, pt.p.y);
                }
            }
            if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); }
        }
    }

    // Vineyard circle - darker purple
    if (document.getElementById('showVineyardCircle').checked) {
        if (vineyardCenter) {
            exportCtx.fillStyle = '#7f1d1d';
            exportCtx.beginPath();
            exportCtx.arc(vineyardCenter.x, vineyardCenter.y, 6 / view.scale, 0, Math.PI * 2);
            exportCtx.fill();
            
            exportCtx.strokeStyle = 'rgba(127, 29, 29, 0.6)';
            exportCtx.lineWidth = 1.5 / view.scale;
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
            
            // Yellow animated center marker
            if (typeof vineyardAnimIdx !== 'undefined' && vineyardAnimIdx < vineyardCenters.length) {
                const curr = vineyardCenters[vineyardAnimIdx];
                exportCtx.fillStyle = '#f59e0b';
                exportCtx.beginPath();
                exportCtx.arc(curr.x, curr.y, 8 / view.scale, 0, Math.PI * 2);
                exportCtx.fill();
            }
        }
    }

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

function exportCanvasAsSVG() {
    // SQUARE export
    const exportSize = Math.min(canvas.width, canvas.height);
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${exportSize}" height="${exportSize}" viewBox="0 0 ${exportSize} ${exportSize}">`;
    // White background for papers
    svg += `<rect width="100%" height="100%" fill="#ffffff"/>`;
    
    const cx = exportSize / 2 + view.x;
    const cy = exportSize / 2 + view.y;
    
    svg += `<g transform="translate(${cx}, ${cy}) scale(${view.scale}, ${-view.scale})">`;
    
    // NO GRID for clean export
    
    if (cachedCurveData && cachedCurveData.length > 0) {
        // Focal set - darker green
        if (document.getElementById('showFocal').checked && cachedFocalData) {
            for (const branch of cachedFocalData.branches) {
                if (branch.length < 2) continue;
                let d = `M ${branch[0].x} ${branch[0].y}`;
                for (let k = 1; k < branch.length; k++) d += ` L ${branch[k].x} ${branch[k].y}`;
                if (branch.closed) d += ' Z';
                svg += `<path d="${d}" stroke="#7c3aed" stroke-width="${1.5/view.scale}" stroke-linejoin="round" stroke-linecap="round" fill="none"/>`;
            }
        }

        if (document.getElementById('showSS').checked && cachedSSData) {
            if (cachedSSData.branches && cachedSSData.branches.length > 0) {
                for (const branch of cachedSSData.branches) {
                    if (branch.length < 2) continue;
                    let d = `M ${branch[0].x} ${branch[0].y}`;
                    for (let k = 1; k < branch.length; k++) d += ` L ${branch[k].x} ${branch[k].y}`;
                    svg += `<path d="${d}" stroke="#3b82f6" stroke-width="${1.5/view.scale}" stroke-linejoin="round" stroke-linecap="round" fill="none"/>`;
                }
            } else {
                const s = 1.5 / view.scale;
                for (const p of cachedSSData) {
                    svg += `<rect x="${p.x - s/2}" y="${p.y - s/2}" width="${s}" height="${s}" fill="#3b82f6"/>`;
                }
            }
        }

        // Curve - darker blue
        if (document.getElementById('showCurve').checked) {
            let currentId = -1;
            let pathData = '';
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (pathData) {
                        svg += `<path d="${pathData} Z" stroke="#171717" stroke-width="${2.5/view.scale}" fill="none"/>`;
                    }
                    pathData = `M ${pt.p.x} ${pt.p.y}`;
                    currentId = pt.curveId;
                } else {
                    pathData += ` L ${pt.p.x} ${pt.p.y}`;
                }
            }
            if (pathData) {
                svg += `<path d="${pathData} Z" stroke="#171717" stroke-width="${2.5/view.scale}" fill="none"/>`;
            }
        }
    }

    // Vineyard - darker purple
    if (document.getElementById('showVineyardCircle').checked && vineyardCenter) {
        svg += `<circle cx="${vineyardCenter.x}" cy="${vineyardCenter.y}" r="${6/view.scale}" fill="#7f1d1d"/>`;
        svg += `<circle cx="${vineyardCenter.x}" cy="${vineyardCenter.y}" r="${vineyardRadius}" stroke="rgba(127,29,29,0.6)" stroke-width="${1.5/view.scale}" fill="none" stroke-dasharray="${5/view.scale} ${5/view.scale}"/>`;
        
        // Vineyard center sample points
        if (vineyardCenters.length > 0) {
            const s = 3 / view.scale;
            for (const c of vineyardCenters) {
                svg += `<rect x="${c.x - s/2}" y="${c.y - s/2}" width="${s}" height="${s}" fill="rgba(127,29,29,0.5)"/>`;
            }
            
            // Yellow animated center marker
            if (typeof vineyardAnimIdx !== 'undefined' && vineyardAnimIdx < vineyardCenters.length) {
                const curr = vineyardCenters[vineyardAnimIdx];
                svg += `<circle cx="${curr.x}" cy="${curr.y}" r="${8/view.scale}" fill="#f59e0b"/>`;
            }
        }
    }

    // Control points - dark
    if (document.getElementById('showControls').checked) {
        const baseSize = 7 / view.scale;
        for (let c = 0; c < curves.length; c++) {
            const pts = curves[c];
            for (const p of pts) {
                const fill = (c === activeCurveIdx) ? '#1a1a1a' : '#888';
                svg += `<rect x="${p.x - baseSize/2}" y="${p.y - baseSize/2}" width="${baseSize}" height="${baseSize}" fill="${fill}"/>`;
            }
        }
    }
    
    svg += `</g></svg>`;
    
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    downloadDataURL(url, 'symmetry_canvas.svg');
    URL.revokeObjectURL(url);
    exportMenu.classList.remove('active');
}

function exportPlotlyAsPNG(plotId, filename) {
    // Get the plot element and save current camera state
    const plotDiv = document.getElementById(plotId);
    let savedCamera = null;
    
    // Save 3D camera position if it's a 3D plot
    if (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera) {
        savedCamera = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
    }
    
    // Export as square with good quality
    const exportSize = 800;
    
    const lightLayout = {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#1a1a1a' },
        showlegend: false,
        scene: plotId === 'vineyardPlot' ? {
            xaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Birth', font: { color: '#333' } } },
            yaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Death', font: { color: '#333' } } },
            zaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Time', font: { color: '#333' } } },
            bgcolor: '#ffffff',
            camera: savedCamera // Preserve camera position
        } : undefined,
        xaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Birth', font: { color: '#333' } } } : undefined,
        yaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Death', font: { color: '#333' } } } : undefined,
        title: { font: { color: '#333' } }
    };
    
    Plotly.relayout(plotId, lightLayout).then(() => {
        return Plotly.toImage(plotId, { format: 'png', width: exportSize, height: exportSize, scale: 2 });
    }).then(dataURL => {
        downloadDataURL(dataURL, filename);
        exportMenu.classList.remove('active');
        // Restore dark theme and camera position if needed
        if (isDarkTheme) {
            const darkLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: plotId === 'persistencePlot' ? 'rgba(10,10,15,0.5)' : 'rgba(0,0,0,0)',
                font: { color: '#ccc' },
                showlegend: false,
                scene: plotId === 'vineyardPlot' ? {
                    xaxis: { color: '#888', gridcolor: '#333', title: { text: 'Birth', font: { color: '#aaa' } } },
                    yaxis: { color: '#888', gridcolor: '#333', title: { text: 'Death', font: { color: '#aaa' } } },
                    zaxis: { color: '#888', gridcolor: '#333', title: { text: 'Time', font: { color: '#aaa' } } },
                    bgcolor: 'rgba(0,0,0,0)',
                    camera: savedCamera
                } : undefined,
                xaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Birth', font: { color: '#aaa' } } } : undefined,
                yaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Death', font: { color: '#aaa' } } } : undefined,
                title: { font: { color: '#aaa' } }
            };
            Plotly.relayout(plotId, darkLayout);
        }
    });
}

function exportPlotlyAsSVG(plotId, filename) {
    const plotDiv = document.getElementById(plotId);
    let savedCamera = null;
    
    if (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera) {
        savedCamera = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
    }
    
    // Export as square
    const exportSize = 800;
    
    const lightLayout = {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#1a1a1a' },
        showlegend: false,
        scene: plotId === 'vineyardPlot' ? {
            xaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Birth', font: { color: '#333' } } },
            yaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Death', font: { color: '#333' } } },
            zaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Time', font: { color: '#333' } } },
            bgcolor: '#ffffff',
            camera: savedCamera
        } : undefined,
        xaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Birth', font: { color: '#333' } } } : undefined,
        yaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Death', font: { color: '#333' } } } : undefined,
        title: { font: { color: '#333' } }
    };
    
    Plotly.relayout(plotId, lightLayout).then(() => {
        return Plotly.toImage(plotId, { format: 'svg', width: exportSize, height: exportSize });
    }).then(dataURL => {
        downloadDataURL(dataURL, filename);
        exportMenu.classList.remove('active');
        if (isDarkTheme) {
            const darkLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: plotId === 'persistencePlot' ? 'rgba(10,10,15,0.5)' : 'rgba(0,0,0,0)',
                font: { color: '#ccc' },
                showlegend: false,
                scene: plotId === 'vineyardPlot' ? {
                    xaxis: { color: '#888', gridcolor: '#333', title: { text: 'Birth', font: { color: '#aaa' } } },
                    yaxis: { color: '#888', gridcolor: '#333', title: { text: 'Death', font: { color: '#aaa' } } },
                    zaxis: { color: '#888', gridcolor: '#333', title: { text: 'Time', font: { color: '#aaa' } } },
                    bgcolor: 'rgba(0,0,0,0)',
                    camera: savedCamera
                } : undefined,
                xaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Birth', font: { color: '#aaa' } } } : undefined,
                yaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Death', font: { color: '#aaa' } } } : undefined,
                title: { font: { color: '#aaa' } }
            };
            Plotly.relayout(plotId, darkLayout);
        }
    });
}

function exportCanvasAsPDF() {
    const { jsPDF } = window.jspdf;
    
    // Create a SQUARE export canvas with good quality
    const exportSize = Math.min(canvas.width, canvas.height);
    const scale = 2;
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = exportSize * scale;
    exportCanvas.height = exportSize * scale;
    const exportCtx = exportCanvas.getContext('2d');
    
    exportCtx.scale(scale, scale);
    // White background for papers
    exportCtx.fillStyle = '#ffffff';
    exportCtx.fillRect(0, 0, exportSize, exportSize);
    
    exportCtx.save();
    // Center the view in the square
    exportCtx.translate(exportSize / 2 + view.x, exportSize / 2 + view.y);
    exportCtx.scale(view.scale, -view.scale);
    
    // NO GRID for clean export

    if (cachedCurveData && cachedCurveData.length > 0) {
        if (document.getElementById('showFocal').checked && cachedFocalData) {
            exportCtx.strokeStyle = '#7c3aed';
            exportCtx.lineWidth = 1.5 / view.scale;
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
                exportCtx.lineWidth = 1.5 / view.scale;
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

        if (document.getElementById('showCurve').checked) {
            exportCtx.strokeStyle = '#171717';
            exportCtx.lineWidth = 2.5 / view.scale;
            let currentId = -1;
            exportCtx.beginPath();
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); exportCtx.beginPath(); }
                    exportCtx.moveTo(pt.p.x, pt.p.y);
                    currentId = pt.curveId;
                } else {
                    exportCtx.lineTo(pt.p.x, pt.p.y);
                }
            }
            if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); }
        }
    }

    if (document.getElementById('showVineyardCircle').checked) {
        if (vineyardCenter) {
            exportCtx.fillStyle = '#7f1d1d';
            exportCtx.beginPath();
            exportCtx.arc(vineyardCenter.x, vineyardCenter.y, 6 / view.scale, 0, Math.PI * 2);
            exportCtx.fill();
            
            exportCtx.strokeStyle = 'rgba(127, 29, 29, 0.6)';
            exportCtx.lineWidth = 1.5 / view.scale;
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
            
            // Yellow animated center marker
            if (typeof vineyardAnimIdx !== 'undefined' && vineyardAnimIdx < vineyardCenters.length) {
                const curr = vineyardCenters[vineyardAnimIdx];
                exportCtx.fillStyle = '#f59e0b';
                exportCtx.beginPath();
                exportCtx.arc(curr.x, curr.y, 8 / view.scale, 0, Math.PI * 2);
                exportCtx.fill();
            }
        }
    }

    if (document.getElementById('showControls').checked) {
        const baseSize = 7 / view.scale;
        for (let c = 0; c < curves.length; c++) {
            const pts = curves[c];
            for (const p of pts) {
                exportCtx.fillStyle = (c === activeCurveIdx) ? '#1a1a1a' : '#888';
                exportCtx.fillRect(p.x - baseSize/2, p.y - baseSize/2, baseSize, baseSize);
            }
        }
    }
    
    exportCtx.restore();
    
    // Use PNG for better quality
    const imgData = exportCanvas.toDataURL('image/png');
    const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'px',
        format: [exportSize, exportSize]
    });
    pdf.addImage(imgData, 'PNG', 0, 0, exportSize, exportSize);
    pdf.save('symmetry_canvas.pdf');
    exportMenu.classList.remove('active');
}

function exportPlotlyAsPDF(plotId, filename) {
    const plotDiv = document.getElementById(plotId);
    let savedCamera = null;
    
    if (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera) {
        savedCamera = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
    }
    
    // Export as square with good quality
    const exportSize = 800;
    
    const lightLayout = {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#1a1a1a' },
        showlegend: false,
        scene: plotId === 'vineyardPlot' ? {
            xaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Birth', font: { color: '#333' } } },
            yaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Death', font: { color: '#333' } } },
            zaxis: { color: '#333', gridcolor: '#ddd', title: { text: 'Time', font: { color: '#333' } } },
            bgcolor: '#ffffff',
            camera: savedCamera
        } : undefined,
        xaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Birth', font: { color: '#333' } } } : undefined,
        yaxis: plotId === 'persistencePlot' ? { color: '#333', gridcolor: '#ddd', zerolinecolor: '#999', title: { text: 'Death', font: { color: '#333' } } } : undefined,
        title: { font: { color: '#333' } }
    };
    
    Plotly.relayout(plotId, lightLayout).then(() => {
        // Use PNG format for better quality
        return Plotly.toImage(plotId, { format: 'png', width: exportSize, height: exportSize, scale: 2 });
    }).then(dataURL => {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'px',
            format: [exportSize, exportSize]
        });
        pdf.addImage(dataURL, 'PNG', 0, 0, exportSize, exportSize);
        pdf.save(filename);
        exportMenu.classList.remove('active');
        if (isDarkTheme) {
            const darkLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: plotId === 'persistencePlot' ? 'rgba(10,10,15,0.5)' : 'rgba(0,0,0,0)',
                font: { color: '#ccc' },
                showlegend: false,
                scene: plotId === 'vineyardPlot' ? {
                    xaxis: { color: '#888', gridcolor: '#333', title: { text: 'Birth', font: { color: '#aaa' } } },
                    yaxis: { color: '#888', gridcolor: '#333', title: { text: 'Death', font: { color: '#aaa' } } },
                    zaxis: { color: '#888', gridcolor: '#333', title: { text: 'Time', font: { color: '#aaa' } } },
                    bgcolor: 'rgba(0,0,0,0)',
                    camera: savedCamera
                } : undefined,
                xaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Birth', font: { color: '#aaa' } } } : undefined,
                yaxis: plotId === 'persistencePlot' ? { color: '#888', gridcolor: '#333', zerolinecolor: '#666', title: { text: 'Death', font: { color: '#aaa' } } } : undefined,
                title: { font: { color: '#aaa' } }
            };
            Plotly.relayout(plotId, darkLayout);
        }
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

async function exportAllAsPDF() {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'px',
        format: [800, 600]
    });
    
    // Canvas
    const scale = 2;
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = canvas.width * scale;
    exportCanvas.height = canvas.height * scale;
    const exportCtx = exportCanvas.getContext('2d');
    
    exportCtx.scale(scale, scale);
    exportCtx.fillStyle = '#0a0a0f';
    exportCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    exportCtx.save();
    exportCtx.translate(canvas.width / 2 + view.x, canvas.height / 2 + view.y);
    exportCtx.scale(view.scale, -view.scale);
    
    exportCtx.strokeStyle = '#1a1a2e';
    exportCtx.lineWidth = 1 / view.scale;
    exportCtx.beginPath();
    exportCtx.moveTo(-1000, 0); exportCtx.lineTo(1000, 0);
    exportCtx.moveTo(0, -1000); exportCtx.lineTo(0, 1000);
    exportCtx.stroke();

    if (cachedCurveData && cachedCurveData.length > 0) {
        if (document.getElementById('showFocal').checked && cachedFocalData) {
            exportCtx.strokeStyle = '#7c3aed';
            exportCtx.lineWidth = 1.5 / view.scale;
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
                exportCtx.lineWidth = 1.5 / view.scale;
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
        if (document.getElementById('showCurve').checked) {
            exportCtx.strokeStyle = '#e0e0e0';
            exportCtx.lineWidth = 2.5 / view.scale;
            let currentId = -1;
            exportCtx.beginPath();
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); exportCtx.beginPath(); }
                    exportCtx.moveTo(pt.p.x, pt.p.y);
                    currentId = pt.curveId;
                } else {
                    exportCtx.lineTo(pt.p.x, pt.p.y);
                }
            }
            if (currentId !== -1) { exportCtx.closePath(); exportCtx.stroke(); }
        }
    }
    if (document.getElementById('showVineyardCircle').checked && vineyardCenter) {
        exportCtx.fillStyle = '#991b1b';
        exportCtx.beginPath();
        exportCtx.arc(vineyardCenter.x, vineyardCenter.y, 6 / view.scale, 0, Math.PI * 2);
        exportCtx.fill();
    }
    if (document.getElementById('showControls').checked) {
        const baseSize = 7 / view.scale;
        for (let c = 0; c < curves.length; c++) {
            for (const p of curves[c]) {
                exportCtx.fillStyle = (c === activeCurveIdx) ? '#fff' : '#666';
                exportCtx.fillRect(p.x - baseSize/2, p.y - baseSize/2, baseSize, baseSize);
            }
        }
    }
    exportCtx.restore();
    
    // Scale canvas to fit PDF page
    const canvasAspect = canvas.width / canvas.height;
    let cw = 800, ch = 600;
    if (canvasAspect > 800/600) {
        ch = 800 / canvasAspect;
    } else {
        cw = 600 * canvasAspect;
    }
    const cx = (800 - cw) / 2;
    const cy = (600 - ch) / 2;
    
    const canvasImg = exportCanvas.toDataURL('image/png');
    pdf.addImage(canvasImg, 'PNG', cx, cy, cw, ch);
    
    if (vineyardData) {
        // Vineyard 3D
        pdf.addPage([800, 600], 'landscape');
        const vineyardImg = await Plotly.toImage('vineyardPlot', { format: 'png', width: 800, height: 600, scale: 2 });
        pdf.addImage(vineyardImg, 'PNG', 0, 0, 800, 600);
        
        // Persistence Diagram
        pdf.addPage([800, 600], 'landscape');
        const pdImg = await Plotly.toImage('persistencePlot', { format: 'png', width: 800, height: 600, scale: 2 });
        pdf.addImage(pdImg, 'PNG', 0, 0, 800, 600);
    }
    
    pdf.save('symmetry_all_figures.pdf');
    exportMenu.classList.remove('active');
}

document.getElementById('exportCanvas').addEventListener('click', exportCanvasAsPNG);
document.getElementById('exportCanvasSVG').addEventListener('click', exportCanvasAsSVG);
document.getElementById('exportCanvasPDF').addEventListener('click', exportCanvasAsPDF);
document.getElementById('exportVineyard').addEventListener('click', () => {
    if (!vineyardData) { alert('No vineyard data to export'); return; }
    exportPlotlyAsPNG('vineyardPlot', 'vineyard_3d.png');
});
document.getElementById('exportVineyardSVG').addEventListener('click', () => {
    if (!vineyardData) { alert('No vineyard data to export'); return; }
    exportPlotlyAsSVG('vineyardPlot', 'vineyard_3d.svg');
});
document.getElementById('exportVineyardPDF').addEventListener('click', () => {
    if (!vineyardData) { alert('No vineyard data to export'); return; }
    exportPlotlyAsPDF('vineyardPlot', 'vineyard_3d.pdf');
});
document.getElementById('exportVineyardOBJ').addEventListener('click', exportVineyardAsOBJ);
document.getElementById('exportVineyardPLY').addEventListener('click', exportVineyardAsPLY);
document.getElementById('exportPD').addEventListener('click', () => {
    if (!vineyardData) { alert('No persistence data to export'); return; }
    exportPlotlyAsPNG('persistencePlot', 'persistence_diagram.png');
});
document.getElementById('exportPDSVG').addEventListener('click', () => {
    if (!vineyardData) { alert('No persistence data to export'); return; }
    exportPlotlyAsSVG('persistencePlot', 'persistence_diagram.svg');
});
document.getElementById('exportPDPDF').addEventListener('click', () => {
    if (!vineyardData) { alert('No persistence data to export'); return; }
    exportPlotlyAsPDF('persistencePlot', 'persistence_diagram.pdf');
});
document.getElementById('exportAllPNG').addEventListener('click', exportAllAsPNG);
document.getElementById('exportAllPDF').addEventListener('click', exportAllAsPDF);

