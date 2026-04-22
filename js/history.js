// ========== History ==========
function saveState() {
    const state = {
        curves: curves.map(curve => curve.map(p => new Point(p.x, p.y))),
        activeCurveIdx: activeCurveIdx,
        vineyardCenter: vineyardCenter ? new Point(vineyardCenter.x, vineyardCenter.y) : null
    };
    historyStack.push(state);
    if (historyStack.length > MAX_HISTORY) historyStack.shift();
}

function performUndo() {
    if (historyStack.length === 0) return;
    const state = historyStack.pop();
    curves = state.curves;
    activeCurveIdx = state.activeCurveIdx;
    vineyardCenter = state.vineyardCenter;
    
    if (activeCurveIdx >= curves.length) activeCurveIdx = curves.length - 1;
    if (activeCurveIdx < 0) { activeCurveIdx = 0; curves = [[]]; }
    
    selectedInfo = null;
    updateStatus();
    updateUI();
    triggerComputation(true);
}

