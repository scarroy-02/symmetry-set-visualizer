// ========== B-Spline ==========
function solveControlPoints(points) {
    const n = points.length;
    if (n < 3) return points;

    const rhsX = new Float64Array(n);
    const rhsY = new Float64Array(n);
    
    for(let i=0; i<n; i++) {
        rhsX[i] = 6.0 * points[i].x;
        rhsY[i] = 6.0 * points[i].y;
    }

    const solX = new Float64Array(n);
    const solY = new Float64Array(n);
    
    for(let i=0; i<n; i++) { solX[i] = points[i].x; solY[i] = points[i].y; }

    for(let iter=0; iter<20; iter++) {
        for(let i=0; i<n; i++) {
            const prev = (i === 0) ? n - 1 : i - 1;
            const next = (i === n - 1) ? 0 : i + 1;
            solX[i] = (rhsX[i] - solX[prev] - solX[next]) * 0.25;
            solY[i] = (rhsY[i] - solY[prev] - solY[next]) * 0.25;
        }
    }

    return Array.from({length: n}, (_, i) => new Point(solX[i], solY[i]));
}

// Natural cubic B-spline interpolation for an open curve.
// Endpoints pinned: C[0] = P[0], C[n-1] = P[n-1].
// Interior knots: C[i-1] + 4*C[i] + C[i+1] = 6*P[i]   (i = 1..n-2)
// Phantom controls used at draw time: C[-1] = 2*C[0]-C[1], C[n] = 2*C[n-1]-C[n-2].
function solveControlPointsOpen(points) {
    const n = points.length;
    if (n < 3) return points;

    const solX = new Float64Array(n);
    const solY = new Float64Array(n);
    for (let i = 0; i < n; i++) { solX[i] = points[i].x; solY[i] = points[i].y; }
    for (let iter = 0; iter < 30; iter++) {
        for (let i = 1; i < n - 1; i++) {
            solX[i] = (6.0 * points[i].x - solX[i - 1] - solX[i + 1]) * 0.25;
            solY[i] = (6.0 * points[i].y - solY[i - 1] - solY[i + 1]) * 0.25;
        }
    }
    return Array.from({ length: n }, (_, i) => new Point(solX[i], solY[i]));
}

function bSplineEval(p0, p1, p2, p3, t) {
    const t2 = t*t, t3 = t2*t;
    const b0 = (-t3 + 3*t2 - 3*t + 1)/6;
    const b1 = (3*t3 - 6*t2 + 4)/6;
    const b2 = (-3*t3 + 3*t2 + 3*t + 1)/6;
    const b3 = t3/6;
    return new Point(p0.x*b0 + p1.x*b1 + p2.x*b2 + p3.x*b3, p0.y*b0 + p1.y*b1 + p2.y*b2 + p3.y*b3);
}

function bSplineDeriv1(p0, p1, p2, p3, t) {
    const t2 = t*t;
    const d0 = (-3*t2 + 6*t - 3)/6;
    const d1 = (9*t2 - 12*t)/6;
    const d2 = (-9*t2 + 6*t + 3)/6;
    const d3 = (3*t2)/6;
    return new Point(p0.x*d0 + p1.x*d1 + p2.x*d2 + p3.x*d3, p0.y*d0 + p1.y*d1 + p2.y*d2 + p3.y*d3);
}

function bSplineDeriv2(p0, p1, p2, p3, t) {
    const dd0 = (-6*t + 6)/6;
    const dd1 = (18*t - 12)/6;
    const dd2 = (-18*t + 6)/6;
    const dd3 = (6*t)/6;
    return new Point(p0.x*dd0 + p1.x*dd1 + p2.x*dd2 + p3.x*dd3, p0.y*dd0 + p1.y*dd1 + p2.y*dd2 + p3.y*dd3);
}

function computeSplineData() {
    const allData = [];
    
    for (let cIdx = 0; cIdx < curves.length; cIdx++) {
        const pts = curves[cIdx];
        if (pts.length < 3) continue;

        const isOpen = !!curveOpen[cIdx];
        const C = isOpen ? solveControlPointsOpen(pts) : solveControlPoints(pts);
        const n = C.length;
        // Closed: wrap C[n-1], C[0], C[1] for cyclic continuity.
        // Open: phantom endpoints from natural-spline boundary (C''=0).
        const drawC = isOpen
            ? [
                new Point(2 * C[0].x - C[1].x, 2 * C[0].y - C[1].y),
                ...C,
                new Point(2 * C[n - 1].x - C[n - 2].x, 2 * C[n - 1].y - C[n - 2].y)
              ]
            : [C[n-1], ...C, C[0], C[1]];

        const numSegments = isOpen ? (pts.length - 1) : pts.length;
        const budget = Math.floor(SAMPLING_DENSITY / Math.max(1, curves.length));
        const pointsPerSeg = Math.floor(budget / Math.max(1, numSegments));

        for (let i = 0; i < numSegments; i++) {
            const p0 = drawC[i], p1 = drawC[i+1], p2 = drawC[i+2], p3 = drawC[i+3];

            for (let j = 0; j < pointsPerSeg; j++) {
                const t = j / pointsPerSeg;
                const P = bSplineEval(p0, p1, p2, p3, t);
                const d1 = bSplineDeriv1(p0, p1, p2, p3, t);
                const d2 = bSplineDeriv2(p0, p1, p2, p3, t);
                
                const velSq = d1.dot(d1);
                if (velSq < 1e-15) continue;

                const T = d1.normalize();
                const N = new Point(-T.y, T.x);
                const k = (d1.x * d2.y - d1.y * d2.x) / Math.pow(velSq, 1.5);

                allData.push({ p: P, T: T, N: N, curvature: k, curveId: cIdx });
            }
        }

        // For open curves, add the t=1 endpoint of the final segment
        // so the sample list reaches the last data point.
        if (isOpen) {
            const i = numSegments - 1;
            const p0 = drawC[i], p1 = drawC[i+1], p2 = drawC[i+2], p3 = drawC[i+3];
            const t = 1.0;
            const P = bSplineEval(p0, p1, p2, p3, t);
            const d1 = bSplineDeriv1(p0, p1, p2, p3, t);
            const d2 = bSplineDeriv2(p0, p1, p2, p3, t);
            const velSq = d1.dot(d1);
            if (velSq >= 1e-15) {
                const T = d1.normalize();
                const N = new Point(-T.y, T.x);
                const k = (d1.x * d2.y - d1.y * d2.x) / Math.pow(velSq, 1.5);
                allData.push({ p: P, T: T, N: N, curvature: k, curveId: cIdx });
            }
        }
    }
    return allData;
}

// ========== Symmetry Set ========== (new version)
function computeSymmetrySet(data, stepSize) {
    const out = [];
    out.branches = [];
    if (!data || data.length === 0) return out;

    const step = Math.max(1, stepSize);
    // Bucket active indices by curveId
    const groups = new Map();
    for (let i = 0; i < data.length; i += step) {
        const cId = data[i].curveId || 0;
        if (!groups.has(cId)) groups.set(cId, []);
        groups.get(cId).push(i);
    }

    const openOf = (cId) => !!(typeof curveOpen !== 'undefined' && curveOpen[cId]);

    // Self-symmetry of each curve
    for (const [cId, indices] of groups.entries()) {
        for (const br of _ssTraceSelf(data, indices, openOf(cId))) {
            out.branches.push(br);
            for (const p of br) out.push(p);
        }
    }

    // Cross-symmetry between pairs of distinct curves
    const curveIds = [...groups.keys()];
    for (let i = 0; i < curveIds.length; i++) {
        for (let j = i + 1; j < curveIds.length; j++) {
            const cA = curveIds[i], cB = curveIds[j];
            for (const br of _ssTraceCross(data, groups.get(cA), groups.get(cB), openOf(cA), openOf(cB))) {
                out.branches.push(br);
                for (const p of br) out.push(p);
            }
        }
    }
    return out;
}

// --- helpers --------------------------------------------------------------

function _ssEvalF(data, ia, ib) {
    const pa = data[ia].p, pb = data[ib].p, Ta = data[ia].T, Tb = data[ib].T;
    const dx = pa.x - pb.x, dy = pa.y - pb.y;
    return {
        fp: dx * (Ta.x + Tb.x) + dy * (Ta.y + Tb.y),
        fm: dx * (Ta.x - Tb.x) + dy * (Ta.y - Tb.y)
    };
}

function _ssDotT(data, i, j) {
    return data[i].T.x * data[j].T.x + data[i].T.y * data[j].T.y;
}

// Marching-squares 16-case lookup, with saddle-point resolution for cases 5/10.
// Returns up to two segments, each a pair of {edge, alpha}.
function _ssMarch(f00, f10, f11, f01) {
    const b = ((f00 > 0) ? 1 : 0) | ((f10 > 0) ? 2 : 0) | ((f11 > 0) ? 4 : 0) | ((f01 > 0) ? 8 : 0);
    const ia = (u, v) => u / (u - v);
    const a0 = () => ia(f00, f10), a1 = () => ia(f10, f11),
          a2 = () => ia(f11, f01), a3 = () => ia(f01, f00);
    switch (b) {
        case 0: case 15: return [];
        case 1: case 14: return [[{edge:0,alpha:a0()},{edge:3,alpha:a3()}]];
        case 2: case 13: return [[{edge:0,alpha:a0()},{edge:1,alpha:a1()}]];
        case 3: case 12: return [[{edge:1,alpha:a1()},{edge:3,alpha:a3()}]];
        case 4: case 11: return [[{edge:1,alpha:a1()},{edge:2,alpha:a2()}]];
        case 6: case 9:  return [[{edge:0,alpha:a0()},{edge:2,alpha:a2()}]];
        case 7: case 8:  return [[{edge:2,alpha:a2()},{edge:3,alpha:a3()}]];
        case 5: {
            const c = (f00 + f10 + f11 + f01) * 0.25;
            return c > 0
                ? [[{edge:0,alpha:a0()},{edge:1,alpha:a1()}],[{edge:2,alpha:a2()},{edge:3,alpha:a3()}]]
                : [[{edge:0,alpha:a0()},{edge:3,alpha:a3()}],[{edge:1,alpha:a1()},{edge:2,alpha:a2()}]];
        }
        case 10: {
            const c = (f00 + f10 + f11 + f01) * 0.25;
            return c > 0
                ? [[{edge:0,alpha:a0()},{edge:3,alpha:a3()}],[{edge:1,alpha:a1()},{edge:2,alpha:a2()}]]
                : [[{edge:0,alpha:a0()},{edge:1,alpha:a1()}],[{edge:2,alpha:a2()},{edge:3,alpha:a3()}]];
        }
    }
    return [];
}

// Lift an edge zero-crossing to a symmetry-set center. Linearly interpolates
// position, renormalises tangent, derives normal, then solves for lambda via
//   sigma = +1 (F_+ = 0):  lambda (N_a + N_b) = P_b - P_a   (opposite normals)
//   sigma = -1 (F_- = 0):  lambda (N_a - N_b) = P_b - P_a   (same-sign normals)
function _ssLift(data, idxA, idxB, nA, nB, a, b, edge, alpha, sigma) {
    const a1 = (a + 1) % nA, b1 = (b + 1) % nB;
    let ia0, ia1, ib0, ib1, ta, tb;
    switch (edge) {
        case 0: ia0 = idxA[a];  ib0 = idxB[b];  ia1 = idxA[a1]; ib1 = idxB[b];  ta = alpha; tb = 0;     break;
        case 1: ia0 = idxA[a1]; ib0 = idxB[b];  ia1 = idxA[a1]; ib1 = idxB[b1]; ta = 0;     tb = alpha; break;
        case 2: ia0 = idxA[a1]; ib0 = idxB[b1]; ia1 = idxA[a];  ib1 = idxB[b1]; ta = alpha; tb = 0;     break;
        case 3: ia0 = idxA[a];  ib0 = idxB[b1]; ia1 = idxA[a];  ib1 = idxB[b];  ta = 0;     tb = alpha; break;
    }
    const Pa = { x: data[ia0].p.x * (1 - ta) + data[ia1].p.x * ta,
                 y: data[ia0].p.y * (1 - ta) + data[ia1].p.y * ta };
    const Pb = { x: data[ib0].p.x * (1 - tb) + data[ib1].p.x * tb,
                 y: data[ib0].p.y * (1 - tb) + data[ib1].p.y * tb };
    const Tax = data[ia0].T.x * (1 - ta) + data[ia1].T.x * ta,
          Tay = data[ia0].T.y * (1 - ta) + data[ia1].T.y * ta;
    const Tbx = data[ib0].T.x * (1 - tb) + data[ib1].T.x * tb,
          Tby = data[ib0].T.y * (1 - tb) + data[ib1].T.y * tb;
    const maN = Math.sqrt(Tax * Tax + Tay * Tay) || 1;
    const mbN = Math.sqrt(Tbx * Tbx + Tby * Tby) || 1;
    const Nax = -Tay / maN, Nay = Tax / maN;
    const Nbx = -Tby / mbN, Nby = Tbx / mbN;

    // A = N_a + sigma * N_b  (note the +: this is the correct correlation)
    const Ax = Nax + sigma * Nbx, Ay = Nay + sigma * Nby;
    const denA = Ax * Ax + Ay * Ay;
    if (denA < 1e-12) return null;
    const lam = ((Pb.x - Pa.x) * Ax + (Pb.y - Pa.y) * Ay) / denA;
    if (!isFinite(lam) || Math.abs(lam) > LAMBDA_MAX) return null;

    const cx = Pa.x + lam * Nax, cy = Pa.y + lam * Nay, r = Math.abs(lam);

    // Physical sanity: the bitangent circle must span the chord (2r >= |Pa - Pb|).
    // Reject the numerical sliver where lambda collapses to zero near the diagonal.
    const chordSq = (Pa.x - Pb.x) ** 2 + (Pa.y - Pb.y) ** 2;
    if (4 * r * r < chordSq * 0.98) return null;

    // Canonical key for the shared edge (stitching neighbouring cells).
    let edgeKey;
    switch (edge) {
        case 0: edgeKey = `H${a}_${b}_${sigma}`;  break;
        case 1: edgeKey = `V${a1}_${b}_${sigma}`; break;
        case 2: edgeKey = `H${a}_${b1}_${sigma}`; break;
        case 3: edgeKey = `V${a}_${b}_${sigma}`;  break;
    }
    return { x: cx, y: cy, r, i1: ia0, i2: ib0, edgeKey };
}

// Walk segments sharing edge keys into polylines.
function _ssStitch(segments) {
    if (segments.length === 0) return [];
    const edgeMap = new Map();
    for (let s = 0; s < segments.length; s++) {
        for (let e = 0; e < 2; e++) {
            const k = segments[s][e].edgeKey;
            if (!edgeMap.has(k)) edgeMap.set(k, []);
            edgeMap.get(k).push({ seg: s, end: e });
        }
    }
    const visited = new Array(segments.length).fill(false);
    const branches = [];
    const neighbour = (s, e) => {
        const list = edgeMap.get(segments[s][e].edgeKey);
        for (const n of list) if (n.seg !== s && !visited[n.seg]) return n;
        return null;
    };
    for (let start = 0; start < segments.length; start++) {
        if (visited[start]) continue;
        visited[start] = true;
        const path = [segments[start][0], segments[start][1]];
        // extend from tail
        let cur = start, end = 1;
        while (true) {
            const n = neighbour(cur, end);
            if (!n) break;
            visited[n.seg] = true;
            const other = 1 - n.end;
            path.push(segments[n.seg][other]);
            cur = n.seg; end = other;
        }
        // extend from head
        cur = start; end = 0;
        while (true) {
            const n = neighbour(cur, end);
            if (!n) break;
            visited[n.seg] = true;
            const other = 1 - n.end;
            path.unshift(segments[n.seg][other]);
            cur = n.seg; end = other;
        }
        branches.push(path);
    }
    return branches;
}

function _ssTraceSelf(data, idx, isOpen) {
    const n = idx.length;
    if (n < 10) return [];
    const MIN_ARC = Math.max(3, Math.floor(n * 0.02));
    const halfN = Math.floor(n / 2);
    const DOTT_GUARD = 0.9999;
    const segsP = [], segsM = [];

    const aHi = isOpen ? n - 1 : n;
    const bHi = isOpen ? n - 1 : n;

    for (let a = 0; a < aHi; a++) {
        for (let b = 0; b < bHi; b++) {
            const a1 = isOpen ? a + 1 : (a + 1) % n;
            const b1 = isOpen ? b + 1 : (b + 1) % n;
            // Fundamental domain.
            // Closed: MIN_ARC < (a - b) mod n <= n/2 at every corner (half-strip).
            // Open  : (b - a) > MIN_ARC at every corner (lower triangle, no wrap).
            let ok = true;
            for (const [ca, cb] of [[a,b],[a1,b],[a1,b1],[a,b1]]) {
                if (isOpen) {
                    const d = cb - ca;
                    if (d <= MIN_ARC) { ok = false; break; }
                } else {
                    const fwd = ((ca - cb) % n + n) % n;
                    if (fwd <= MIN_ARC || fwd > halfN) { ok = false; break; }
                }
            }
            if (!ok) continue;

            // Tangent-parallelism guards (skip cells where F is ill-conditioned):
            //   F_+ misbehaves when T_a ~ -T_b,  F_- when T_a ~ +T_b.
            const d00 = _ssDotT(data, idx[a],  idx[b]);
            const d10 = _ssDotT(data, idx[a1], idx[b]);
            const d11 = _ssDotT(data, idx[a1], idx[b1]);
            const d01 = _ssDotT(data, idx[a],  idx[b1]);
            const safeP = d00 > -DOTT_GUARD && d10 > -DOTT_GUARD && d11 > -DOTT_GUARD && d01 > -DOTT_GUARD;
            const safeM = d00 <  DOTT_GUARD && d10 <  DOTT_GUARD && d11 <  DOTT_GUARD && d01 <  DOTT_GUARD;
            if (!safeP && !safeM) continue;

            const e00 = _ssEvalF(data, idx[a],  idx[b]);
            const e10 = _ssEvalF(data, idx[a1], idx[b]);
            const e11 = _ssEvalF(data, idx[a1], idx[b1]);
            const e01 = _ssEvalF(data, idx[a],  idx[b1]);

            if (safeP) {
                for (const seg of _ssMarch(e00.fp, e10.fp, e11.fp, e01.fp)) {
                    const p0 = _ssLift(data, idx, idx, n, n, a, b, seg[0].edge, seg[0].alpha, +1);
                    const p1 = _ssLift(data, idx, idx, n, n, a, b, seg[1].edge, seg[1].alpha, +1);
                    if (p0 && p1) segsP.push([p0, p1]);
                }
            }
            if (safeM) {
                for (const seg of _ssMarch(e00.fm, e10.fm, e11.fm, e01.fm)) {
                    const p0 = _ssLift(data, idx, idx, n, n, a, b, seg[0].edge, seg[0].alpha, -1);
                    const p1 = _ssLift(data, idx, idx, n, n, a, b, seg[1].edge, seg[1].alpha, -1);
                    if (p0 && p1) segsM.push([p0, p1]);
                }
            }
        }
    }
    return [..._ssStitch(segsP), ..._ssStitch(segsM)];
}

function _ssTraceCross(data, idxA, idxB, openA, openB) {
    const nA = idxA.length, nB = idxB.length;
    if (nA < 4 || nB < 4) return [];
    const DOTT_GUARD = 0.9999;
    const segsP = [], segsM = [];
    const aHi = openA ? nA - 1 : nA;
    const bHi = openB ? nB - 1 : nB;
    for (let a = 0; a < aHi; a++) {
        for (let b = 0; b < bHi; b++) {
            const a1 = openA ? a + 1 : (a + 1) % nA;
            const b1 = openB ? b + 1 : (b + 1) % nB;
            const d00 = _ssDotT(data, idxA[a],  idxB[b]);
            const d10 = _ssDotT(data, idxA[a1], idxB[b]);
            const d11 = _ssDotT(data, idxA[a1], idxB[b1]);
            const d01 = _ssDotT(data, idxA[a],  idxB[b1]);
            const safeP = d00 > -DOTT_GUARD && d10 > -DOTT_GUARD && d11 > -DOTT_GUARD && d01 > -DOTT_GUARD;
            const safeM = d00 <  DOTT_GUARD && d10 <  DOTT_GUARD && d11 <  DOTT_GUARD && d01 <  DOTT_GUARD;
            if (!safeP && !safeM) continue;

            const e00 = _ssEvalF(data, idxA[a],  idxB[b]);
            const e10 = _ssEvalF(data, idxA[a1], idxB[b]);
            const e11 = _ssEvalF(data, idxA[a1], idxB[b1]);
            const e01 = _ssEvalF(data, idxA[a],  idxB[b1]);

            if (safeP) {
                for (const seg of _ssMarch(e00.fp, e10.fp, e11.fp, e01.fp)) {
                    const p0 = _ssLift(data, idxA, idxB, nA, nB, a, b, seg[0].edge, seg[0].alpha, +1);
                    const p1 = _ssLift(data, idxA, idxB, nA, nB, a, b, seg[1].edge, seg[1].alpha, +1);
                    if (p0 && p1) segsP.push([p0, p1]);
                }
            }
            if (safeM) {
                for (const seg of _ssMarch(e00.fm, e10.fm, e11.fm, e01.fm)) {
                    const p0 = _ssLift(data, idxA, idxB, nA, nB, a, b, seg[0].edge, seg[0].alpha, -1);
                    const p1 = _ssLift(data, idxA, idxB, nA, nB, a, b, seg[1].edge, seg[1].alpha, -1);
                    if (p0 && p1) segsM.push([p0, p1]);
                }
            }
        }
    }
    return [..._ssStitch(segsP), ..._ssStitch(segsM)];
}

// ========== Focal Set ==========
// The evolute (focal set) of a plane curve gamma is the map
//   gamma(t) → gamma(t) + N(t)/kappa(t).
// Since it is parametrised by t alone, we do not need marching squares — we
// just walk curve samples, split at inflections (kappa sign change) and at
// near-flat regions (|kappa| tiny), and stitch the rest into polylines.
//
// Return shape is identical to computeSymmetrySet: a flat array of
// {x, y, i, r, k} with a `.branches` property (array of polylines). Each
// polyline that fully closes carries `.closed = true`.
function computeFocalSet(data) {
    const out = [];
    out.branches = [];
    if (!data || data.length === 0) return out;

    // Bucket by curveId, preserving sample order
    const groups = new Map();
    for (let i = 0; i < data.length; i++) {
        const cId = data[i].curveId || 0;
        if (!groups.has(cId)) groups.set(cId, []);
        groups.get(cId).push(i);
    }

    // Reference scale: any focal-point jump larger than the curve bounding-box
    // diagonal is almost certainly a near-inflection artefact and should cut
    // the branch.
    let minX =  Infinity, maxX = -Infinity, minY =  Infinity, maxY = -Infinity;
    for (const d of data) {
        if (d.p.x < minX) minX = d.p.x;
        if (d.p.x > maxX) maxX = d.p.x;
        if (d.p.y < minY) minY = d.p.y;
        if (d.p.y > maxY) maxY = d.p.y;
    }
    const diag = Math.max(1e-6, Math.hypot(maxX - minX, maxY - minY));
    const MAX_JUMP_SQ = diag * diag;
    const MIN_K = 1.0 / R_MAX;

    for (const [cId, indices] of groups.entries()) {
        const n = indices.length;
        if (n < 3) continue;
        const isOpen = !!(typeof curveOpen !== 'undefined' && curveOpen[cId]);

        // Per-sample focal point (or null if ill-defined)
        const foc = new Array(n);
        for (let k = 0; k < n; k++) {
            const pt = data[indices[k]];
            const kv = pt.curvature;
            if (!isFinite(kv) || Math.abs(kv) < MIN_K) { foc[k] = null; continue; }
            const R = 1.0 / kv;
            foc[k] = {
                x: pt.p.x + pt.N.x * R,
                y: pt.p.y + pt.N.y * R,
                i: indices[k],
                r: Math.abs(R),
                k: kv
            };
        }

        // Mark breaks between k and k+1 (cyclic): null, sign change, or large jump
        const brk = new Array(n).fill(false);
        for (let k = 0; k < n; k++) {
            const kn = (k + 1) % n;
            const a = foc[k], b = foc[kn];
            if (!a || !b)        { brk[k] = true; continue; }
            if (a.k * b.k < 0)   { brk[k] = true; continue; }  // inflection
            const dx = b.x - a.x, dy = b.y - a.y;
            if (dx*dx + dy*dy > MAX_JUMP_SQ) { brk[k] = true; }
        }
        // For open curves, the wraparound from k=n-1 to k=0 is not real geometry.
        if (isOpen) brk[n - 1] = true;

        const closed = !brk.some(b => b);

        // Start just after some break, or 0 if fully closed
        let start = 0;
        if (!closed) {
            for (let k = 0; k < n; k++) if (brk[k]) { start = (k + 1) % n; break; }
        }

        // Walk cyclically from start, cutting at each break
        let current = null;
        const flush = () => {
            if (current && current.length >= 2) {
                out.branches.push(current);
                for (const p of current) out.push(p);
            } else if (current && current.length === 1) {
                out.push(current[0]);
            }
            current = null;
        };

        for (let step = 0; step < n; step++) {
            const k = (start + step) % n;
            if (foc[k]) {
                if (!current) current = [];
                current.push(foc[k]);
            } else {
                flush();
            }
            if (brk[k]) flush();
        }
        if (current && closed) current.closed = true;
        flush();
    }

    return out;
}

