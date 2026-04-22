// ========== Presets ==========
const PRESET_FILES = {
    ellipse: 'assets/ellipse.sspl',
    astroid: 'assets/astroid.sspl',
    bean: 'assets/kidney_bean.sspl',
    cardioid: 'assets/cardoid.sspl',
    offsetSpiral: 'assets/offset_spiral_scaled.sspl',
    bishopStaff: 'assets/bishop_staff_scaled.sspl',
    deformedEllipse: 'assets/deformed_ellipse.sspl'
};

async function loadPreset(type) {
    const url = PRESET_FILES[type];
    if (!url) return;
    try {
        const res = await fetch(url);
        applyFigureData(await res.json());
    } catch (err) {
        updateStatus('Failed to load preset');
    }
    document.getElementById('presetSelect').value = "";
}

function applyFigureData(data) {
    if (data.format !== 'symmetry-set-spline' || !Array.isArray(data.curves)) {
        updateStatus('Invalid figure file');
        return;
    }
    saveState();
    curves = data.curves.map(c => c.map(p => new Point(p.x, p.y)));
    if (curves.length === 0) curves = [[]];
    activeCurveIdx = Math.min(data.activeCurveIdx ?? 0, curves.length - 1);
    if (data.view) {
        view.x = data.view.x;
        view.y = data.view.y;
        view.scale = data.view.scale;
    }
    fixedCurveMode = false;
    cachedCurveData = null;
    cachedSSData = null; sortedSSIndices = null;
    selectedInfo = null;
    vineyardCenter = null;
    vineyardData = null;
    vineyardCenters = [];
    customLoopPoints = [];
    updateLoopStatus();
    updateUI();
    triggerComputation(true);
    updateStatus(`Loaded ${curves.reduce((s, c) => s + c.length, 0)} points`);
}

