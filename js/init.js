// ========== Resize ==========
function resize() {
    const container = document.querySelector('.canvas-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    draw();
}
window.addEventListener('resize', resize);
resize();

updateStatus('Click to add points or load a preset');
