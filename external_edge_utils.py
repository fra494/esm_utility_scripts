import pydiffvg
import torch
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from common_utils import *

def build_scene(svg_strokes, canvas_w, canvas_h):
    shape_groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0., 0., 0., 1.])
        )
        for i in range(len(svg_strokes))
    ]
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_w, canvas_h, svg_strokes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply
    raster = render(canvas_w, canvas_h, 2, 2, 0, None, *scene_args)
    gray = 1.0 - raster[..., :3].mean(dim=-1, keepdim=True)
    return gray.permute(2, 0, 1).unsqueeze(0)

def build_attraction_field(edge_map, canvas_w, canvas_h, sigma=8.0):
    """
    Costruisce un campo vettoriale che punta verso il bordo più vicino.
    Ogni pixel (x,y) contiene il vettore (dx,dy) verso il bordo più vicino.
    
    Restituisce:
        field_x, field_y: array [H, W] con componenti del vettore attrazione
        dist_map: distanza dal bordo più vicino per ogni pixel
    """
    # Mappa binaria dei bordi
    edge_binary = (edge_map > 0.1).astype(np.uint8)
    
    # Distance transform: per ogni pixel, distanza dal bordo più vicino
    # e coordinate del pixel di bordo più vicino
    dist_map, nearest_idx = distance_transform_edt(
        1 - edge_binary, return_indices=True
    )
    
    # nearest_idx[0] = riga del bordo più vicino
    # nearest_idx[1] = colonna del bordo più vicino
    rows = np.arange(canvas_h)[:, None] * np.ones((1, canvas_w))
    cols = np.ones((canvas_h, 1)) * np.arange(canvas_w)[None, :]
    
    # Vettore verso il bordo più vicino
    field_y = nearest_idx[0] - rows  # dy
    field_x = nearest_idx[1] - cols  # dx
    
    # Normalizza (evita divisione per zero dove il pixel è già sul bordo)
    magnitude = np.sqrt(field_x**2 + field_y**2) + 1e-8
    field_x = field_x / magnitude
    field_y = field_y / magnitude
    
    # Scala per distanza: attrazione forte vicino, debole lontano
    weight = np.exp(-dist_map / sigma)
    field_x *= weight
    field_y *= weight
    
    print(f"[attraction field] dist media dal bordo: {dist_map.mean():.2f}px, "
          f"max: {dist_map.max():.2f}px")
    
    return field_x, field_y, dist_map

def snap_strokes_to_edges(svg_strokes, field_x, field_y, dist_map,
                           canvas_w, canvas_h,
                           max_displacement=15.0,
                           n_iter=50,
                           only_far_threshold=5.0):
    """
    Sposta ogni punto di controllo verso il bordo più vicino
    usando il campo di attrazione.
    
    Args:
        max_displacement: spostamento massimo per iterazione (px)
        only_far_threshold: sposta solo i punti più lontani di N px dal bordo
    """
    field_x_t = torch.from_numpy(field_x).float()
    field_y_t = torch.from_numpy(field_y).float()
    dist_t    = torch.from_numpy(dist_map).float()
    
    total_moved = 0
    total_points = 0
    
    for iteration in range(n_iter):
        moved_this_iter = 0
        
        for stroke in svg_strokes:
            pts = stroke.points.data  # [N, 2] — x, y in pixel
            
            for pi in range(pts.shape[0]):
                px = pts[pi, 0].long().clamp(0, canvas_w - 1).item()
                py = pts[pi, 1].long().clamp(0, canvas_h - 1).item()
                
                # Distanza dal bordo per questo punto
                d = dist_t[py, px].item()
                
                # Salta punti già vicini al bordo
                if d < only_far_threshold:
                    continue
                
                # Vettore attrazione al pixel corrente
                ax = field_x_t[py, px].item()
                ay = field_y_t[py, px].item()
                
                # Spostamento proporzionale alla distanza, capped a max_displacement
                step = min(d * 0.4, max_displacement)
                dx = ax * step
                dy = ay * step
                
                pts[pi, 0] = (pts[pi, 0] + dx).clamp(0, canvas_w)
                pts[pi, 1] = (pts[pi, 1] + dy).clamp(0, canvas_h)
                moved_this_iter += 1
        
        if iteration == 0:
            total_points = sum(s.points.shape[0] for s in svg_strokes)
            print(f"[snap] punti totali: {total_points}")
        
        if iteration % 10 == 0:
            # Calcola distanza media attuale
            all_dists = []
            for stroke in svg_strokes:
                pts = stroke.points.data
                for pi in range(pts.shape[0]):
                    px = pts[pi, 0].long().clamp(0, canvas_w - 1).item()
                    py = pts[pi, 1].long().clamp(0, canvas_h - 1).item()
                    all_dists.append(dist_t[py, px].item())
            mean_dist = np.mean(all_dists)
            print(f"[snap] iter {iteration:3d} | dist media dal bordo: {mean_dist:.2f}px "
                  f"| punti mossi: {moved_this_iter}")
        
        total_moved += moved_this_iter
        
        # Early stop: se nessun punto si muove più
        if moved_this_iter == 0:
            print(f"[snap] convergenza a iter {iteration}")
            break
    
    return svg_strokes