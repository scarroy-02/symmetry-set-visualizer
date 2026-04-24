// ========== Math Classes ==========
class Point {
    constructor(x, y) { this.x = x; this.y = y; }
    add(o) { return new Point(this.x + o.x, this.y + o.y); }
    sub(o) { return new Point(this.x - o.x, this.y - o.y); }
    mult(s) { return new Point(this.x * s, this.y * s); }
    dot(o) { return this.x * o.x + this.y * o.y; }
    norm() { return Math.sqrt(this.x * this.x + this.y * this.y); }
    normalize() { const n = this.norm(); return n < 1e-15 ? new Point(0,0) : this.mult(1/n); }
}

// ========== API Configuration ==========
// For GitHub Pages: Set this to your deployed Python server URL
// For local development: Use 'http://localhost:5000'
// For hosting: 'https://scarroy.eu.pythonanywhere.com/' 'https://scarroy-persistence-endpoint.hf.space/'
const PERSISTENCE_API_URL = 'https://scarroy-persistence-endpoint.hf.space/';
// const PERSISTENCE_API_URL = 'http://localhost:5000';

// ========== Global State ==========
const SAMPLING_DENSITY = 20000;
const LAMBDA_MAX = 10000.0;
const R_MAX = 100000.0;

let curves = [[]];
let activeCurveIdx = 0;

let cachedCurveData = null;
let cachedSSData = null;
let cachedFocalData = null;
let ssStepSize = 5;

let dragInfo = null;
let hoverInfo = null;
let selectedInfo = null;
let view = { x: 0, y: 0, scale: 50 };

let isInteractionActive = false;
let interactionType = null;
let mouseStart = { x: 0, y: 0 };
let viewStart = { x: 0, y: 0 };

const MAX_HISTORY = 50;
let historyStack = [];

let vineyardMode = false;
let vineyardCenter = null;
let vineyardRadius = 1.0;
let vineyardSamples = 64;
let vineyardCenters = [];
let vineyardData = null;
let vineyardAnimIdx = 0;
let vineyardAnimPlaying = false;
let currentPersistence = null;
let vineyardMaxVal = 1;
let vineyardDragging = false;

// Custom loop for vineyard
let vineyardLoopType = 'circular'; // 'circular' or 'custom'
let customLoopMode = false;
let customLoopPoints = []; // Control points for the custom loop spline
let customLoopDragIdx = -1;
let customLoopHoverIdx = -1;

let focalAnimPlaying = false;
let focalAnimIdx = 0;

let ssAnimPlaying = false;
let ssAnimIdx = 0;

let radiusSweepData = null;
let radiusSweepPlaying = false;
let radiusSweepIdx = 0;
let radiusStart = 0.5;
let radiusEnd = 5.0;
let radiusSteps = 30;

let fixedCurveMode = false;

const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');

