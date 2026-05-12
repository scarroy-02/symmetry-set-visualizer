// ========== Per-curve list (open/closed checkbox per curve) ==========
function rebuildCurveList() {
    const host = document.getElementById('curveList');
    if (!host) return;
    host.innerHTML = '';
    for (let i = 0; i < curves.length; i++) {
        const row = document.createElement('div');
        row.className = 'toggle-row';
        row.style.cursor = 'pointer';

        const label = document.createElement('span');
        label.innerText = `Curve ${i}` + (i === activeCurveIdx ? ' ★' : '');
        label.style.flex = '1';
        label.addEventListener('click', () => {
            activeCurveIdx = i;
            selectedInfo = null;
            updateUI();
            updateStatus();
            draw();
        });

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!curveOpen[i];
        cb.title = 'Open (non-closed) curve';
        cb.addEventListener('change', () => {
            saveState();
            curveOpen[i] = cb.checked;
            triggerComputation(true);
            rebuildCurveList();
        });

        const tag = document.createElement('span');
        tag.innerText = 'Open';
        tag.style.fontSize = '9px';
        tag.style.color = 'var(--text-muted)';
        tag.style.marginRight = '4px';

        row.appendChild(label);
        row.appendChild(tag);
        row.appendChild(cb);
        host.appendChild(row);
    }
}

// ========== UI Helpers ==========
function updateStatus(msg) {
    const pts = curves[activeCurveIdx] || [];
    document.getElementById('status').innerText = msg || `Curve ${activeCurveIdx}: ${pts.length} pts`;
}

function updateUI() {
    document.getElementById('curveCountBadge').innerText = curves.length;
    // Keep curveOpen[] length in sync with curves[]
    while (curveOpen.length < curves.length) curveOpen.push(false);
    if (curveOpen.length > curves.length) curveOpen.length = curves.length;
    if (typeof rebuildCurveList === 'function') rebuildCurveList();
}

function updateModeIndicator() {
    const indicator = document.getElementById('modeIndicator');
    if (vineyardMode) {
        indicator.innerText = 'Vineyard';
        indicator.className = 'mode-badge mode-vineyard';
    } else if (customLoopMode) {
        indicator.innerText = 'Loop';
        indicator.className = 'mode-badge mode-vineyard';
    } else {
        indicator.innerText = 'Draw';
        indicator.className = 'mode-badge mode-draw';
    }
}

function screenToWorld(sx, sy) {
    const rect = canvas.getBoundingClientRect();
    const cx = canvas.width/2 + view.x;
    const cy = canvas.height/2 + view.y;
    return new Point((sx - rect.left - cx) / view.scale, -(sy - rect.top - cy) / view.scale);
}

function worldToScreen(p) {
    const cx = canvas.width/2 + view.x;
    const cy = canvas.height/2 + view.y;
    return { x: cx + p.x * view.scale, y: cy - p.y * view.scale };
}

function findHitControlPoint(sx, sy) {
    const threshSq = 144;
    const rect = canvas.getBoundingClientRect();
    for(let c=0; c<curves.length; c++) {
        for(let i=0; i<curves[c].length; i++) {
            const sp = worldToScreen(curves[c][i]);
            const dx = sp.x - (sx - rect.left), dy = sp.y - (sy - rect.top);
            if (dx*dx + dy*dy < threshSq) return { cIdx: c, pIdx: i };
        }
    }
    return null;
}

function pointToSegmentDistSq(p, v, w) {
    const l2 = (v.x - w.x)**2 + (v.y - w.y)**2;
    if (l2 === 0) return (p.x - v.x)**2 + (p.y - v.y)**2;
    let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    const projX = v.x + t * (w.x - v.x);
    const projY = v.y + t * (w.y - v.y);
    return (p.x - projX)**2 + (p.y - projY)**2;
}

function insertPointSmart(newPt) {
    const pts = curves[activeCurveIdx];
    const isOpen = !!curveOpen[activeCurveIdx];

    if (pts.length < 3) {
        pts.push(newPt);
        selectedInfo = { cIdx: activeCurveIdx, pIdx: pts.length - 1 };
        return;
    }

    // Open: only score real segments (i, i+1) — no wrap.
    // Closed: include the closing segment (n-1, 0).
    const lastSegIdx = isOpen ? pts.length - 2 : pts.length - 1;
    let bestIdx = -1, minDistSq = Infinity;
    for (let i = 0; i <= lastSegIdx; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const distSq = pointToSegmentDistSq(newPt, p1, p2);
        if (distSq < minDistSq) { minDistSq = distSq; bestIdx = i; }
    }

    pts.splice(bestIdx + 1, 0, newPt);
    selectedInfo = { cIdx: activeCurveIdx, pIdx: bestIdx + 1 };
}

function findHitVineyardCenter(sx, sy) {
    if (!vineyardCenter) return false;
    const rect = canvas.getBoundingClientRect();
    const sp = worldToScreen(vineyardCenter);
    const dx = sp.x - (sx - rect.left), dy = sp.y - (sy - rect.top);
    return dx*dx + dy*dy < 200;
}

// ========== Event Handlers ==========
canvas.addEventListener('mousedown', e => {
    e.preventDefault();
    const mousePos = screenToWorld(e.clientX, e.clientY);
    
    mouseStart = { x: e.clientX, y: e.clientY };
    viewStart = { x: view.x, y: view.y };
    isInteractionActive = true;

    if (e.button === 2 || e.altKey) {
        interactionType = 'PAN';
        canvas.style.cursor = 'grabbing';
        return;
    }

    if (e.button === 0) {
        // Check for custom loop point drag FIRST (before adding new points)
        if (customLoopPoints.length > 0) {
            const hitIdx = findHitCustomLoopPoint(e.clientX, e.clientY);
            if (hitIdx >= 0) {
                saveState();
                customLoopDragIdx = hitIdx;
                interactionType = 'DRAG_LOOP_POINT';
                canvas.style.cursor = 'grabbing';
                return;
            }
        }
        
        // Custom loop mode - add new point only if not hitting existing point
        if (customLoopMode) {
            saveState();
            customLoopPoints.push(new Point(mousePos.x, mousePos.y));
            updateLoopStatus();
            isInteractionActive = false;
            draw();
            return;
        }
        
        if (vineyardMode) {
            saveState();
            vineyardCenter = mousePos;
            vineyardMode = false;
            updateModeIndicator();
            document.getElementById('vineyardModeBtn').classList.remove('active');
            document.getElementById('vineyardModeBtn').innerText = '⬡ Place Center';
            updateStatus('Center placed!');
            isInteractionActive = false;
            draw();
            return;
        }
        
        if (findHitVineyardCenter(e.clientX, e.clientY)) {
            vineyardDragging = true;
            interactionType = 'DRAG_VINEYARD';
            canvas.style.cursor = 'grabbing';
            return;
        }
        
        const hit = findHitControlPoint(e.clientX, e.clientY);
        if (hit !== null) {
            selectedInfo = { ...hit };
            activeCurveIdx = hit.cIdx;
            saveState();
            dragInfo = hit;
            interactionType = 'DRAG_POINT';
            draw();
            return;
        }
        
        interactionType = 'POTENTIAL_PAN';
    }
});

// Find hit custom loop point
function findHitCustomLoopPoint(sx, sy) {
    const worldPos = screenToWorld(sx, sy);
    const hitRadius = 12 / view.scale;
    
    for (let i = 0; i < customLoopPoints.length; i++) {
        const pt = customLoopPoints[i];
        const dx = worldPos.x - pt.x;
        const dy = worldPos.y - pt.y;
        if (dx*dx + dy*dy < hitRadius * hitRadius) {
            return i;
        }
    }
    return -1;
}

canvas.addEventListener('mousemove', e => {
    const dx = e.clientX - mouseStart.x;
    const dy = e.clientY - mouseStart.y;
    const distSq = dx*dx + dy*dy;

    if (interactionType === 'DRAG_VINEYARD' && vineyardDragging) {
        const p = screenToWorld(e.clientX, e.clientY);
        vineyardCenter = p;
        if (vineyardCenters.length > 0) {
            vineyardCenters = [];
            for (let i = 0; i < vineyardSamples; i++) {
                const theta = (i / vineyardSamples) * Math.PI * 2;
                vineyardCenters.push(new Point(
                    vineyardCenter.x + vineyardRadius * Math.cos(theta),
                    vineyardCenter.y + vineyardRadius * Math.sin(theta)
                ));
            }
        }
        draw();
        return;
    }
    
    if (interactionType === 'DRAG_LOOP_POINT' && customLoopDragIdx >= 0) {
        const p = screenToWorld(e.clientX, e.clientY);
        customLoopPoints[customLoopDragIdx] = new Point(p.x, p.y);
        draw();
        return;
    }

    if (interactionType === 'DRAG_POINT' && dragInfo) {
        const p = screenToWorld(e.clientX, e.clientY);
        curves[dragInfo.cIdx][dragInfo.pIdx] = p;
        triggerComputation(false);
        return;
    }

    if (interactionType === 'PAN') {
        view.x = viewStart.x + dx;
        view.y = viewStart.y + dy;
        draw();
        return;
    }

    if (interactionType === 'POTENTIAL_PAN' && distSq > 25) {
        interactionType = 'PAN';
        canvas.style.cursor = 'grabbing';
        return;
    }

    if (!isInteractionActive) {
        // Check custom loop point hover
        if (customLoopPoints.length > 0) {
            const hitIdx = findHitCustomLoopPoint(e.clientX, e.clientY);
            if (hitIdx >= 0) {
                customLoopHoverIdx = hitIdx;
                canvas.style.cursor = 'grab';
                draw();
                return;
            }
            customLoopHoverIdx = -1;
        }
        
        if (findHitVineyardCenter(e.clientX, e.clientY)) {
            canvas.style.cursor = 'grab';
            return;
        }
        
        const hit = findHitControlPoint(e.clientX, e.clientY);
        hoverInfo = hit;
        if (hit) {
            canvas.style.cursor = 'grab';
            updateStatus(`Curve ${hit.cIdx}, Point ${hit.pIdx}`);
        } else {
            canvas.style.cursor = (vineyardMode || customLoopMode) ? 'cell' : 'crosshair';
        }
        draw();
    }
});

canvas.addEventListener('mouseup', e => {
    if (!isInteractionActive) return;

    if (interactionType === 'DRAG_VINEYARD') {
        vineyardDragging = false;
        if (vineyardData && cachedCurveData) {
            computeVineyard();
        }
    }
    
    if (interactionType === 'DRAG_LOOP_POINT') {
        customLoopDragIdx = -1;
    }

    if (interactionType === 'DRAG_POINT') {
        dragInfo = null;
        triggerComputation(true);
    }
    
    if (interactionType === 'POTENTIAL_PAN' && !vineyardMode && !customLoopMode) {
        if (fixedCurveMode) {
            updateStatus('Fixed curve mode - Clear to draw');
        } else {
            saveState();
            const mousePos = screenToWorld(e.clientX, e.clientY);
            insertPointSmart(mousePos);
            updateStatus();
            triggerComputation(true);
        }
    }

    isInteractionActive = false;
    interactionType = null;
    canvas.style.cursor = (vineyardMode || customLoopMode) ? 'cell' : 'crosshair';
    draw();
});

canvas.addEventListener('contextmenu', e => e.preventDefault());

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = Math.exp(e.deltaY > 0 ? -0.1 : 0.1);
    view.scale *= factor;
    draw();
}, { passive: false });

window.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        performUndo();
        return;
    }

    if (e.key === 'Delete' || e.key === 'Backspace') {
        // Delete custom loop point if hovering
        if (customLoopHoverIdx >= 0 && customLoopPoints.length > 0) {
            saveState();
            customLoopPoints.splice(customLoopHoverIdx, 1);
            customLoopHoverIdx = -1;
            updateLoopStatus();
            draw();
            return;
        }
        
        if (selectedInfo) {
            const pts = curves[selectedInfo.cIdx];
            if (selectedInfo.pIdx < pts.length) {
                saveState();
                pts.splice(selectedInfo.pIdx, 1);
                if (pts.length === 0 && curves.length > 1) {
                    curves.splice(selectedInfo.cIdx, 1);
                    curveOpen.splice(selectedInfo.cIdx, 1);
                    activeCurveIdx = Math.max(0, activeCurveIdx - 1);
                }
                selectedInfo = null;
                updateUI();
                updateStatus();
                triggerComputation(true);
            }
        }
    }
    
    if (e.key === 'Escape') {
        vineyardMode = false;
        customLoopMode = false;
        updateModeIndicator();
        document.getElementById('vineyardModeBtn').classList.remove('active');
        document.getElementById('vineyardModeBtn').innerText = '⬡ Place Center';
        document.getElementById('drawLoopBtn').classList.remove('active');
        draw();
    }
});

// ========== Button Handlers ==========
document.getElementById('undoBtn').addEventListener('click', performUndo);

document.getElementById('newCurveBtn').addEventListener('click', () => {
    saveState();
    curves.push([]);
    curveOpen.push(false);
    activeCurveIdx = curves.length - 1;
    selectedInfo = null;
    updateUI();
    updateStatus(`New Curve ${activeCurveIdx}`);
    draw();
});

document.getElementById('clearBtn').addEventListener('click', () => {
    saveState();
    curves = [[]];
    curveOpen = [false];
    activeCurveIdx = 0;
    selectedInfo = null;
    cachedCurveData = null;
    cachedSSData = null; sortedSSIndices = null;
    cachedFocalData = null;
    vineyardCenter = null;
    vineyardData = null;
    vineyardCenters = [];
    fixedCurveMode = false;
    
    // Clear custom loop
    customLoopPoints = [];
    customLoopMode = false;
    document.getElementById('drawLoopBtn').classList.remove('active');
    updateLoopStatus();
    
    stopVineyardAnim();
    document.getElementById('vineyardPanel').classList.remove('active');
    document.getElementById('persistencePanel').classList.remove('active');
    updateModeIndicator();
    updateUI();
    updateStatus('Cleared');
    draw();
});

document.getElementById('resetViewBtn').addEventListener('click', () => {
    view.x = 0; view.y = 0; view.scale = 50;
    draw();
});

document.getElementById('computeBtn').addEventListener('click', () => triggerComputation(true));

document.getElementById('presetSelect').addEventListener('change', e => {
    if (e.target.value) loadPreset(e.target.value);
});

