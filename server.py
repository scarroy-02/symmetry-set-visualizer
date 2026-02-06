"""
Extended Persistence Server using GUDHI
Flask server that computes extended persistence for radial filtration on closed curves.

To run locally:
    pip install flask flask-cors gudhi numpy
    python persistence_server.py

The server will run on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import gudhi as gd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'gudhi_version': gd.__version__})

@app.route('/persistence', methods=['POST'])
def compute_persistence():
    """
    Compute extended persistence for a radial filtration.
    
    Request JSON:
    {
        "center": {"x": float, "y": float},
        "points": [{"x": float, "y": float, "curveId": int}, ...],
        "use_squared_distance": bool (default: true)
    }
    
    Response JSON:
    {
        "ord0": [[birth, death], ...],
        "rel1": [[birth, death], ...],
        "ext0": [[birth, death], ...],
        "ext1": [[birth, death], ...],
        "r_min": float,
        "r_max": float
    }
    """
    try:
        data = request.get_json()
        
        center = data['center']
        points = data['points']
        use_squared = data.get('use_squared_distance', True)
        
        cx, cy = center['x'], center['y']
        n = len(points)
        
        if n < 3:
            return jsonify({'error': 'Need at least 3 points'}), 400
        
        # Group points by curve ID
        curve_groups = {}
        for i, pt in enumerate(points):
            cid = pt.get('curveId', 0)
            if cid not in curve_groups:
                curve_groups[cid] = []
            curve_groups[cid].append(i)
        
        # Compute distances
        coords = np.array([[p['x'], p['y']] for p in points])
        center_arr = np.array([cx, cy])
        
        if use_squared:
            distances = np.sum((coords - center_arr)**2, axis=1)
        else:
            distances = np.linalg.norm(coords - center_arr, axis=1)
        
        # Build simplex tree
        st = gd.SimplexTree()
        
        # Insert vertices
        for i in range(n):
            st.insert([i], filtration=float(distances[i]))
        
        # Insert edges (closed loops for each curve)
        for cid, indices in curve_groups.items():
            m = len(indices)
            for j in range(m):
                v1 = indices[j]
                v2 = indices[(j + 1) % m]
                f_val = max(distances[v1], distances[v2])
                st.insert([v1, v2], filtration=float(f_val))
        
        # Compute extended persistence
        st.extend_filtration()
        dgms = st.extended_persistence()
        
        # dgms[0] -> Ordinary (dimension 0)
        # dgms[1] -> Relative (dimension 0) - usually empty for simple curves
        # dgms[2] -> Extended+ (dimension 0)
        # dgms[3] -> Extended- (dimension 1)
        
        r_min = float(np.min(distances))
        r_max = float(np.max(distances))
        
        def format_dgm(dgm):
            """Convert diagram to list of [birth, death] pairs"""
            result = []
            for dim, (birth, death) in dgm:
                # Skip infinite deaths for now, or cap them
                if np.isinf(death):
                    death = r_max * 1.5
                if np.isinf(birth):
                    birth = r_min
                result.append([float(birth), float(death)])
            return result
        
        # Extract H0 (dimension 0) and H1 (dimension 1) pairs
        ord0 = []  # Ordinary H0
        ord1 = []  # Ordinary H1
        rel0 = []  # Relative H0
        rel1 = []  # Relative H1
        ext0 = []  # Extended H0
        ext1 = []  # Extended H1
        
        # Ordinary pairs (dgms[0])
        for dim, (birth, death) in dgms[0]:
            pair = [float(birth), float(death) if not np.isinf(death) else r_max * 1.5]
            if dim == 0:
                ord0.append(pair)
            else:
                ord1.append(pair)
        
        # Relative pairs (dgms[1])
        for dim, (birth, death) in dgms[1]:
            pair = [float(birth), float(death) if not np.isinf(death) else r_max * 1.5]
            if dim == 0:
                rel0.append(pair)
            else:
                rel1.append(pair)
        
        # Extended+ pairs (dgms[2])
        for dim, (birth, death) in dgms[2]:
            pair = [float(birth), float(death) if not np.isinf(death) else r_max * 1.5]
            if dim == 0:
                ext0.append(pair)
            else:
                ext1.append(pair)
        
        # Extended- pairs (dgms[3])
        for dim, (birth, death) in dgms[3]:
            pair = [float(birth), float(death) if not np.isinf(death) else r_max * 1.5]
            if dim == 0:
                ext0.append(pair)  # Usually not present
            else:
                ext1.append(pair)
        
        return jsonify({
            'ord0': ord0,
            'ord1': ord1,
            'rel0': rel0,
            'rel1': rel1,
            'ext0': ext0,
            'ext1': ext1,
            'r_min': r_min,
            'r_max': r_max
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/vineyard', methods=['POST'])
def compute_vineyard():
    """
    Compute vineyard (persistence over multiple centers).
    
    Request JSON:
    {
        "centers": [{"x": float, "y": float}, ...],
        "points": [{"x": float, "y": float, "curveId": int}, ...],
        "use_squared_distance": bool (default: true)
    }
    
    Response JSON:
    {
        "ord0": [{"birth": float, "death": float, "centerIdx": int}, ...],
        "rel1": [...],
        "ext0": [...],
        "ext1": [...],
        "infinityY": float
    }
    """
    try:
        data = request.get_json()
        
        centers = data['centers']
        points = data['points']
        use_squared = data.get('use_squared_distance', True)
        
        n = len(points)
        num_centers = len(centers)
        
        if n < 3:
            return jsonify({'error': 'Need at least 3 points'}), 400
        
        # Group points by curve ID
        curve_groups = {}
        for i, pt in enumerate(points):
            cid = pt.get('curveId', 0)
            if cid not in curve_groups:
                curve_groups[cid] = []
            curve_groups[cid].append(i)
        
        coords = np.array([[p['x'], p['y']] for p in points])
        
        # Compute max distance for infinityY
        max_dist_global = 0.0
        for c in centers:
            center_arr = np.array([c['x'], c['y']])
            if use_squared:
                dists = np.sum((coords - center_arr)**2, axis=1)
            else:
                dists = np.linalg.norm(coords - center_arr, axis=1)
            max_dist_global = max(max_dist_global, float(np.max(dists)))
        
        infinityY = float(max_dist_global * 1.15)
        
        # Results
        ord0_all = []
        ord1_all = []
        rel0_all = []
        rel1_all = []
        ext0_all = []
        ext1_all = []
        
        for ci, center in enumerate(centers):
            cx, cy = center['x'], center['y']
            center_arr = np.array([cx, cy])
            
            if use_squared:
                distances = np.sum((coords - center_arr)**2, axis=1)
            else:
                distances = np.linalg.norm(coords - center_arr, axis=1)
            
            # Build simplex tree
            st = gd.SimplexTree()
            
            for i in range(n):
                st.insert([i], filtration=float(distances[i]))
            
            for cid, indices in curve_groups.items():
                m = len(indices)
                for j in range(m):
                    v1 = indices[j]
                    v2 = indices[(j + 1) % m]
                    f_val = max(distances[v1], distances[v2])
                    st.insert([v1, v2], filtration=float(f_val))
            
            st.extend_filtration()
            dgms = st.extended_persistence()
            
            r_max = float(np.max(distances))
            
            # Process diagrams
            for dim, (birth, death) in dgms[0]:  # Ordinary
                is_inf = bool(np.isinf(death))
                d = float(death) if not is_inf else infinityY
                entry = {'birth': float(birth), 'death': d, 'centerIdx': ci, 
                         'isInfinite': is_inf, 'type': 'ord'}
                if dim == 0:
                    ord0_all.append(entry)
                else:
                    ord1_all.append(entry)
            
            for dim, (birth, death) in dgms[1]:  # Relative
                is_inf = bool(np.isinf(death))
                d = float(death) if not is_inf else infinityY
                entry = {'birth': float(birth), 'death': d, 'centerIdx': ci,
                         'isInfinite': is_inf, 'type': 'rel'}
                if dim == 0:
                    rel0_all.append(entry)
                else:
                    rel1_all.append(entry)
            
            for dim, (birth, death) in dgms[2]:  # Extended+
                is_inf = bool(np.isinf(death))
                d = float(death) if not is_inf else infinityY
                entry = {'birth': float(birth), 'death': d, 'centerIdx': ci,
                         'isInfinite': is_inf, 'type': 'ext'}
                if dim == 0:
                    ext0_all.append(entry)
                else:
                    ext1_all.append(entry)
            
            for dim, (birth, death) in dgms[3]:  # Extended-
                is_inf = bool(np.isinf(death))
                d = float(death) if not is_inf else infinityY
                entry = {'birth': float(birth), 'death': d, 'centerIdx': ci,
                         'isInfinite': is_inf, 'type': 'ext'}
                if dim == 0:
                    ext0_all.append(entry)
                else:
                    ext1_all.append(entry)
        
        return jsonify({
            'ord0': ord0_all,
            'ord1': ord1_all,
            'rel0': rel0_all,
            'rel1': rel1_all,
            'ext0': ext0_all,
            'ext1': ext1_all,
            'infinityY': infinityY
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("Starting Extended Persistence Server...")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /persistence - Single center persistence")
    print("  POST /vineyard    - Vineyard computation")
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)