// ========== Computation ==========
let computeTimer = null;
function triggerComputation(fullCompute = true) {
    if (fixedCurveMode) {
        draw();
        return;
    }
    
    const hasValidCurve = curves.some(c => c.length >= 3);
    if (!hasValidCurve) { draw(); return; }
    
    if (!fullCompute) {
        cachedCurveData = computeSplineData();
        cachedSSData = null; sortedSSIndices = null;
        cachedFocalData = computeFocalSet(cachedCurveData);
        draw();
        return;
    }

    if (computeTimer) clearTimeout(computeTimer);

    computeTimer = setTimeout(() => {
        cachedCurveData = computeSplineData();
        cachedSSData = computeSymmetrySet(cachedCurveData, ssStepSize); sortedSSIndices = null;
        cachedFocalData = computeFocalSet(cachedCurveData);
        updateStatus(`SS: ${cachedSSData.length} pts`);
        draw();
    }, 10);
}

function getLineThickness() {
    const el = document.getElementById('exportLineThickness');
    return el ? (parseFloat(el.value) || 1) : 1;
}

// ========== Drawing ==========
function draw() {
    const lineT = getLineThickness();
    // Theme-aware colors
    const bgColor = isDarkTheme ? '#0a0a0f' : '#f5f5f7';
    const gridColor = isDarkTheme ? '#1a1a2e' : '#e0e0e0';
    const curveColor = isDarkTheme ? '#e0e0e0' : '#171717';
    const focalColor = isDarkTheme ? '#a855f7' : '#7c3aed';
    const ssColor = isDarkTheme ? '#60a5fa' : '#3b82f6';
    const focalAnimColor = isDarkTheme ? '#7c3aed' : '#5b21b6';
    const focalAnimColorAlpha = isDarkTheme ? 'rgba(124, 58, 237, 0.4)' : 'rgba(91, 33, 182, 0.4)';
    const ssAnimColor = isDarkTheme ? '#2563eb' : '#1d4ed8';
    const ssAnimColorAlpha = isDarkTheme ? 'rgba(37, 99, 235, 0.4)' : 'rgba(29, 78, 216, 0.4)';
    const vineyardColor = isDarkTheme ? '#991b1b' : '#7f1d1d';
    const controlActive = isDarkTheme ? '#fff' : '#1a1a1a';
    const controlInactive = isDarkTheme ? '#666' : '#888';
    
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.save();
    ctx.translate(canvas.width / 2 + view.x, canvas.height / 2 + view.y);
    ctx.scale(view.scale, -view.scale);
    
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1 / view.scale;
    ctx.beginPath();
    ctx.moveTo(-1000, 0); ctx.lineTo(1000, 0);
    ctx.moveTo(0, -1000); ctx.lineTo(0, 1000);
    ctx.stroke();

    if (cachedCurveData && cachedCurveData.length > 0) {
        if (document.getElementById('showFocal').checked && cachedFocalData) {
            ctx.strokeStyle = focalColor;
            ctx.lineWidth = 1.5 * lineT / view.scale;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            for (const branch of cachedFocalData.branches) {
                if (branch.length < 2) continue;
                ctx.beginPath();
                ctx.moveTo(branch[0].x, branch[0].y);
                for (let k = 1; k < branch.length; k++) ctx.lineTo(branch[k].x, branch[k].y);
                if (branch.closed) ctx.closePath();
                ctx.stroke();
            }
        }

        if (document.getElementById('showSS').checked && cachedSSData) {
            if (cachedSSData.branches && cachedSSData.branches.length > 0) {
                ctx.strokeStyle = ssColor;
                ctx.lineWidth = 1.5 * lineT / view.scale;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                for (const branch of cachedSSData.branches) {
                    if (branch.length < 2) continue;
                    ctx.beginPath();
                    ctx.moveTo(branch[0].x, branch[0].y);
                    for (let k = 1; k < branch.length; k++) ctx.lineTo(branch[k].x, branch[k].y);
                    ctx.stroke();
                }
            } else {
                ctx.fillStyle = ssColor;
                const s = 1.5 / view.scale;
                for (const p of cachedSSData) ctx.fillRect(p.x - s/2, p.y - s/2, s, s);
            }
        }

        if (document.getElementById('showCurve').checked) {
            ctx.strokeStyle = curveColor;
            ctx.lineWidth = 2.5 * lineT / view.scale;
            
            let currentId = -1;
            ctx.beginPath();
            
            for (let i = 0; i < cachedCurveData.length; i++) {
                const pt = cachedCurveData[i];
                if (pt.curveId !== currentId) {
                    if (currentId !== -1) {
                        if (!curveOpen[currentId]) ctx.closePath();
                        ctx.stroke();
                        ctx.beginPath();
                    }
                    ctx.moveTo(pt.p.x, pt.p.y);
                    currentId = pt.curveId;
                } else {
                    ctx.lineTo(pt.p.x, pt.p.y);
                }
            }
            if (currentId !== -1) {
                if (!curveOpen[currentId]) ctx.closePath();
                ctx.stroke();
            }
        }
        
        if ((focalAnimPlaying || focalAnimIdx > 0) && focalAnimIdx < cachedCurveData.length) {
            const pt = cachedCurveData[focalAnimIdx];
            
            ctx.save();
            ctx.strokeStyle = focalAnimColor;
            ctx.fillStyle = focalAnimColor;
            ctx.lineWidth = 4 / view.scale;

            const s = 18 / view.scale;
            ctx.beginPath();
            ctx.arc(pt.p.x, pt.p.y, s/2, 0, Math.PI * 2);
            ctx.fill();

            const k = pt.curvature;
            if (Math.abs(k) > 1e-8) {
                const R = 1.0 / k;
                if (Math.abs(R) < R_MAX) {
                    const ex = pt.p.x + pt.N.x * R;
                    const ey = pt.p.y + pt.N.y * R;

                    ctx.beginPath();
                    ctx.arc(ex, ey, s/2, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.beginPath();
                    ctx.moveTo(pt.p.x, pt.p.y);
                    ctx.lineTo(ex, ey);
                    ctx.stroke();

                    ctx.strokeStyle = focalAnimColorAlpha;
                    ctx.lineWidth = 3 / view.scale;
                    ctx.beginPath();
                    ctx.arc(ex, ey, Math.abs(R), 0, Math.PI * 2);
                    ctx.stroke();
                }
            }
            ctx.restore();
        }

        // Symmetry Set animation
        if ((ssAnimPlaying || ssAnimIdx > 0) && cachedSSData && sortedSSIndices && ssAnimIdx < sortedSSIndices.length && cachedCurveData) {
            const sp = cachedSSData[sortedSSIndices[ssAnimIdx]];

            ctx.save();
            ctx.fillStyle = ssAnimColor;
            ctx.strokeStyle = ssAnimColor;
            ctx.lineWidth = 4 / view.scale;

            const s = 18 / view.scale;

            // Draw SS center point
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, s/2, 0, Math.PI * 2);
            ctx.fill();

            // Draw the two generating curve points and connecting lines
            if (sp.i1 != null && sp.i2 != null && sp.i1 < cachedCurveData.length && sp.i2 < cachedCurveData.length) {
                const p1 = cachedCurveData[sp.i1];
                const p2 = cachedCurveData[sp.i2];

                // Curve contact points
                ctx.beginPath();
                ctx.arc(p1.p.x, p1.p.y, s/2, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.arc(p2.p.x, p2.p.y, s/2, 0, Math.PI * 2);
                ctx.fill();

                // Lines from SS center to curve points
                ctx.beginPath();
                ctx.moveTo(sp.x, sp.y);
                ctx.lineTo(p1.p.x, p1.p.y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(sp.x, sp.y);
                ctx.lineTo(p2.p.x, p2.p.y);
                ctx.stroke();

                // Bitangent circle
                if (sp.r != null && sp.r < R_MAX) {
                    ctx.strokeStyle = ssAnimColorAlpha;
                    ctx.lineWidth = 3 / view.scale;
                    ctx.beginPath();
                    ctx.arc(sp.x, sp.y, sp.r, 0, Math.PI * 2);
                    ctx.stroke();
                }
            }

            ctx.restore();
        }
    }

    if (document.getElementById('showVineyardCircle').checked) {
        // Draw custom loop if exists
        if (customLoopPoints.length > 0) {
            // Draw spline curve through control points (auto-closed when 3+ points)
            if (customLoopPoints.length >= 3) {
                const loopSamples = sampleCustomLoop(200);
                if (loopSamples.length > 0) {
                    ctx.strokeStyle = isDarkTheme ? 'rgba(153, 27, 27, 0.7)' : 'rgba(127, 29, 29, 0.8)';
                    ctx.lineWidth = 2 / view.scale;
                    
                    ctx.beginPath();
                    ctx.moveTo(loopSamples[0].x, loopSamples[0].y);
                    for (let i = 1; i < loopSamples.length; i++) {
                        ctx.lineTo(loopSamples[i].x, loopSamples[i].y);
                    }
                    ctx.closePath();
                    ctx.stroke();
                }
            } else if (customLoopPoints.length === 2) {
                // Draw line between 2 points
                ctx.strokeStyle = isDarkTheme ? 'rgba(153, 27, 27, 0.5)' : 'rgba(127, 29, 29, 0.6)';
                ctx.lineWidth = 1.5 / view.scale;
                ctx.setLineDash([5 / view.scale, 5 / view.scale]);
                ctx.beginPath();
                ctx.moveTo(customLoopPoints[0].x, customLoopPoints[0].y);
                ctx.lineTo(customLoopPoints[1].x, customLoopPoints[1].y);
                ctx.stroke();
                ctx.setLineDash([]);
            }
            
            // Draw control points
            for (let i = 0; i < customLoopPoints.length; i++) {
                const pt = customLoopPoints[i];
                const isHover = (i === customLoopHoverIdx);
                const size = (isHover ? 8 : 6) / view.scale;
                
                ctx.fillStyle = isHover ? '#fbbf24' : vineyardColor;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, size, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Label showing loop status
            if (customLoopPoints.length > 0) {
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                const firstPt = customLoopPoints[0];
                const statusX = firstPt.x * view.scale + view.x + canvas.width / 2;
                const statusY = canvas.height / 2 - firstPt.y * view.scale + view.y - 15;
                ctx.font = '10px JetBrains Mono, monospace';
                const isReady = customLoopPoints.length >= 3;
                ctx.fillStyle = isReady ? '#22c55e' : '#f59e0b';
                ctx.fillText(isReady ? `Loop (${customLoopPoints.length} pts)` : `${customLoopPoints.length}/3 pts`, statusX, statusY);
                ctx.restore();
            }
        }
        
        // Draw circular vineyard (only when in circular mode)
        if (vineyardLoopType === 'circular' && vineyardCenter) {
            ctx.fillStyle = vineyardColor;
            ctx.beginPath();
            ctx.arc(vineyardCenter.x, vineyardCenter.y, 6 / view.scale, 0, Math.PI * 2);
            ctx.fill();
            
            // Main vineyard radius - purple dashed
            ctx.strokeStyle = isDarkTheme ? 'rgba(153, 27, 27, 0.7)' : 'rgba(127, 29, 29, 0.8)';
            ctx.lineWidth = 2 / view.scale;
            ctx.setLineDash([5 / view.scale, 5 / view.scale]);
            ctx.beginPath();
            ctx.arc(vineyardCenter.x, vineyardCenter.y, vineyardRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Label for main radius
            ctx.save();
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            const rLabelX = (vineyardCenter.x + vineyardRadius) * view.scale + view.x + canvas.width / 2;
            const rLabelY = canvas.height / 2 - vineyardCenter.y * view.scale + view.y;
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.fillStyle = isDarkTheme ? 'rgba(153, 27, 27, 0.9)' : 'rgba(127, 29, 29, 1)';
            ctx.fillText(`R=${vineyardRadius.toFixed(2)}`, rLabelX + 5, rLabelY);
            ctx.restore();
            
            // Radius sweep preview circles (start and end radii)
            if (document.getElementById('showSweepPreview').checked) {
                const startR = parseFloat(document.getElementById('radiusStart').value) || 0.5;
                const endR = parseFloat(document.getElementById('radiusEnd').value) || 5;
                
                // Start radius - cyan/teal dashed
                ctx.strokeStyle = isDarkTheme ? 'rgba(34, 211, 238, 0.6)' : 'rgba(6, 182, 212, 0.7)';
                ctx.lineWidth = 1.5 / view.scale;
                ctx.setLineDash([3 / view.scale, 3 / view.scale]);
                ctx.beginPath();
                ctx.arc(vineyardCenter.x, vineyardCenter.y, startR, 0, Math.PI * 2);
                ctx.stroke();
                
                // Label for start radius
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                const startLabelX = (vineyardCenter.x + startR * 0.707) * view.scale + view.x + canvas.width / 2;
                const startLabelY = canvas.height / 2 - (vineyardCenter.y + startR * 0.707) * view.scale + view.y;
                ctx.font = '9px JetBrains Mono, monospace';
                ctx.fillStyle = isDarkTheme ? 'rgba(34, 211, 238, 0.9)' : 'rgba(6, 182, 212, 1)';
                ctx.fillText(`Start=${startR.toFixed(2)}`, startLabelX + 3, startLabelY - 3);
                ctx.restore();
                
                // End radius - orange dashed
                ctx.strokeStyle = isDarkTheme ? 'rgba(251, 146, 60, 0.6)' : 'rgba(234, 88, 12, 0.7)';
                ctx.beginPath();
                ctx.arc(vineyardCenter.x, vineyardCenter.y, endR, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
                
                // Label for end radius
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                const endLabelX = (vineyardCenter.x + endR * 0.707) * view.scale + view.x + canvas.width / 2;
                const endLabelY = canvas.height / 2 - (vineyardCenter.y + endR * 0.707) * view.scale + view.y;
                ctx.font = '9px JetBrains Mono, monospace';
                ctx.fillStyle = isDarkTheme ? 'rgba(251, 146, 60, 0.9)' : 'rgba(234, 88, 12, 1)';
                ctx.fillText(`End=${endR.toFixed(2)}`, endLabelX + 3, endLabelY - 3);
                ctx.restore();
            }
        }
        
        if (vineyardCenters.length > 0) {
            ctx.fillStyle = isDarkTheme ? 'rgba(153, 27, 27, 0.4)' : 'rgba(127, 29, 29, 0.5)';
            const s = 3 / view.scale;
            for (let i = 0; i < vineyardCenters.length; i++) {
                const c = vineyardCenters[i];
                ctx.fillRect(c.x - s/2, c.y - s/2, s, s);
            }
            
            if (vineyardAnimIdx < vineyardCenters.length) {
                const curr = vineyardCenters[vineyardAnimIdx];
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                ctx.arc(curr.x, curr.y, 8 / view.scale, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw birth/death circles if enabled
                if (document.getElementById('showBirthDeathCircles').checked && vineyardData) {
                    const ord0 = (vineyardData.ord0 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    const rel0 = (vineyardData.rel0 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    const ext0 = (vineyardData.ext0 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    const ord1 = (vineyardData.ord1 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    const rel1 = (vineyardData.rel1 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    const ext1 = (vineyardData.ext1 || []).filter(d => d.centerIdx === vineyardAnimIdx);
                    
                    // Helper function to draw birth/death circles
                    const drawBDCircle = (pt, birthColor, deathColor, lineWidth, prefix) => {
                        const birthR = Math.sqrt(pt.birth);
                        const deathR = pt.isInfinite ? null : Math.sqrt(pt.death);
                        
                        // Birth circle
                        ctx.strokeStyle = birthColor;
                        ctx.lineWidth = lineWidth / view.scale;
                        ctx.setLineDash([]);
                        ctx.beginPath();
                        ctx.arc(curr.x, curr.y, birthR, 0, Math.PI * 2);
                        ctx.stroke();
                        
                        // Death circle
                        if (deathR !== null) {
                            ctx.strokeStyle = deathColor;
                            ctx.lineWidth = (lineWidth * 0.75) / view.scale;
                            ctx.setLineDash([4 / view.scale, 4 / view.scale]);
                            ctx.beginPath();
                            ctx.arc(curr.x, curr.y, deathR, 0, Math.PI * 2);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }
                    };
                    
                    // Draw Ordinary H0 (red, thin)
                    for (const pt of ord0) {
                        drawBDCircle(pt, 'rgba(239, 68, 68, 0.8)', 'rgba(239, 68, 68, 0.5)', 2, '');
                    }
                    
                    // Draw Relative H0 (orange, medium)
                    for (const pt of rel0) {
                        drawBDCircle(pt, 'rgba(249, 115, 22, 0.9)', 'rgba(249, 115, 22, 0.6)', 2.5, 'R');
                    }
                    
                    // Draw Extended H0 (yellow, thick)
                    for (const pt of ext0) {
                        drawBDCircle(pt, 'rgba(234, 179, 8, 0.9)', 'rgba(234, 179, 8, 0.6)', 3, 'E');
                    }
                    
                    // Draw Ordinary H1 (blue, thin)
                    for (const pt of ord1) {
                        drawBDCircle(pt, 'rgba(59, 130, 246, 0.8)', 'rgba(59, 130, 246, 0.5)', 2, '');
                    }
                    
                    // Draw Relative H1 (cyan, medium)
                    for (const pt of rel1) {
                        drawBDCircle(pt, 'rgba(6, 182, 212, 0.9)', 'rgba(6, 182, 212, 0.6)', 2.5, 'R');
                    }
                    
                    // Draw Extended H1 (purple, thick)
                    for (const pt of ext1) {
                        drawBDCircle(pt, 'rgba(153, 27, 27, 0.9)', 'rgba(153, 27, 27, 0.6)', 3, 'E');
                    }
                }
                
                if (cachedCurveData && cachedCurveData.length > 0 && !document.getElementById('showBirthDeathCircles').checked) {
                    ctx.strokeStyle = 'rgba(251, 191, 36, 0.15)';
                    ctx.lineWidth = 0.5 / view.scale;
                    const step = Math.max(1, Math.floor(cachedCurveData.length / 50));
                    for (let i = 0; i < cachedCurveData.length; i += step) {
                        ctx.beginPath();
                        ctx.moveTo(curr.x, curr.y);
                        ctx.lineTo(cachedCurveData[i].p.x, cachedCurveData[i].p.y);
                        ctx.stroke();
                    }
                }
            }
        }
    }

    if (document.getElementById('showControls').checked) {
        const baseSize = 7 / view.scale;
        const hoverSize = 10 / view.scale;
        
        for (let c = 0; c < curves.length; c++) {
            const pts = curves[c];
            for (let i = 0; i < pts.length; i++) {
                const p = pts[i];
                const isHover = (hoverInfo && hoverInfo.cIdx === c && hoverInfo.pIdx === i) || 
                               (dragInfo && dragInfo.cIdx === c && dragInfo.pIdx === i);
                const isSelected = (selectedInfo && selectedInfo.cIdx === c && selectedInfo.pIdx === i);
                const isActiveCurve = (c === activeCurveIdx);
                
                if (isSelected) ctx.fillStyle = curveColor;
                else if (isHover) ctx.fillStyle = '#fbbf24';
                else ctx.fillStyle = isActiveCurve ? controlActive : controlInactive;

                const s = (isHover || isSelected) ? hoverSize : baseSize;
                ctx.fillRect(p.x - s/2, p.y - s/2, s, s);
                
                if (isSelected) {
                    ctx.strokeStyle = curveColor;
                    ctx.lineWidth = 1.5 / view.scale;
                    ctx.strokeRect(p.x - s/2 - 2/view.scale, p.y - s/2 - 2/view.scale, s + 4/view.scale, s + 4/view.scale);
                }
            }
        }
    }
    
    ctx.restore();
}

