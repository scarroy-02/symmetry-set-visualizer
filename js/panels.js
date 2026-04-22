// ========== Collapsible Sections ==========
document.querySelectorAll('.section-header').forEach(header => {
    header.addEventListener('click', () => {
        header.parentElement.classList.toggle('collapsed');
    });
});

// ========== Close buttons for floating panels ==========
document.getElementById('closeVineyardBtn').addEventListener('click', () => {
    document.getElementById('vineyardPanel').classList.remove('active');
    document.getElementById('showVineyardPlot').checked = false;
});

document.getElementById('closePDBtn').addEventListener('click', () => {
    document.getElementById('persistencePanel').classList.remove('active');
    document.getElementById('showPD').checked = false;
});

// ========== Panel Dragging ==========
function makePanelDraggable(panelId) {
    const panel = document.getElementById(panelId);
    const header = panel.querySelector('.float-panel-header');
    let isDragging = false;
    let startX, startY, startLeft, startTop;
    
    header.addEventListener('mousedown', (e) => {
        // Don't drag if clicking buttons
        if (e.target.tagName === 'BUTTON') return;
        
        isDragging = true;
        const rect = panel.getBoundingClientRect();
        startX = e.clientX;
        startY = e.clientY;
        startLeft = rect.left;
        startTop = rect.top;
        
        // Switch to fixed positioning if not already
        panel.style.position = 'fixed';
        panel.style.left = startLeft + 'px';
        panel.style.top = startTop + 'px';
        panel.style.right = 'auto';
        panel.style.bottom = 'auto';
        
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        
        let newLeft = startLeft + dx;
        let newTop = startTop + dy;
        
        // Keep panel within viewport bounds
        newLeft = Math.max(0, Math.min(window.innerWidth - panel.offsetWidth, newLeft));
        newTop = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, newTop));
        
        panel.style.left = newLeft + 'px';
        panel.style.top = newTop + 'px';
    });
    
    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
}

makePanelDraggable('vineyardPanel');
makePanelDraggable('persistencePanel');

// ========== Panel Resizing ==========
function makePanelResizable(panelId, plotId) {
    const panel = document.getElementById(panelId);
    const MIN_W = 220, MIN_H = 180;

    panel.querySelectorAll('.resize-handle').forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();

            const dir = handle.dataset.dir;
            const rect = panel.getBoundingClientRect();
            const startX = e.clientX, startY = e.clientY;
            const startW = rect.width, startH = rect.height;
            const startLeft = rect.left, startTop = rect.top;

            panel.style.position = 'fixed';
            panel.style.left = startLeft + 'px';
            panel.style.top = startTop + 'px';
            panel.style.right = 'auto';
            panel.style.bottom = 'auto';
            panel.classList.add('resizing');

            let rafId = null;
            const resizePlot = () => {
                if (typeof Plotly === 'undefined') return;
                const plotEl = document.getElementById(plotId);
                if (plotEl && plotEl.data) {
                    try { Plotly.Plots.resize(plotEl); } catch (_) {}
                }
            };

            const onMove = (ev) => {
                const dx = ev.clientX - startX;
                const dy = ev.clientY - startY;

                let newW = startW, newH = startH;
                let newLeft = startLeft, newTop = startTop;

                if (dir.includes('e')) newW = Math.max(MIN_W, startW + dx);
                if (dir.includes('s')) newH = Math.max(MIN_H, startH + dy);
                if (dir.includes('w')) {
                    newW = Math.max(MIN_W, startW - dx);
                    newLeft = startLeft + (startW - newW);
                }
                if (dir.includes('n')) {
                    newH = Math.max(MIN_H, startH - dy);
                    newTop = startTop + (startH - newH);
                }

                newLeft = Math.max(0, Math.min(window.innerWidth - newW, newLeft));
                newTop = Math.max(0, Math.min(window.innerHeight - newH, newTop));

                panel.style.width = newW + 'px';
                panel.style.height = newH + 'px';
                panel.style.left = newLeft + 'px';
                panel.style.top = newTop + 'px';

                if (rafId === null) {
                    rafId = requestAnimationFrame(() => {
                        rafId = null;
                        resizePlot();
                    });
                }
            };

            const onUp = () => {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                panel.classList.remove('resizing');
                resizePlot();
            };

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    });
}

makePanelResizable('vineyardPanel', 'vineyardPlot');
makePanelResizable('persistencePanel', 'persistencePlot');

