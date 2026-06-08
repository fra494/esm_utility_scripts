import pydiffvg
import torch
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def classify_strokes(svg_strokes, dist_map, canvas_w, canvas_h,
                     contour_threshold=15.0):
    """
    Classifica ogni stroke come 'contorno' o 'dettaglio interno'.
    
    Uno stroke è di contorno se almeno uno dei suoi punti di controllo
    è entro contour_threshold pixel dal bordo della maschera.
    
    Restituisce:
        contour_strokes: lista di stroke da spostare verso i bordi
        detail_strokes:  lista di stroke da lasciare invariati
        contour_idx:     indici originali degli stroke di contorno
        detail_idx:      indici originali degli stroke di dettaglio
    """
    dist_t = torch.from_numpy(dist_map).float()
    
    contour_strokes, detail_strokes = [], []
    contour_idx, detail_idx = [], []
    
    for i, stroke in enumerate(svg_strokes):
        pts = stroke.points.data
        
        # Distanza minima dal bordo tra tutti i punti di controllo
        min_dist = float('inf')
        for pi in range(pts.shape[0]):
            px = pts[pi, 0].long().clamp(0, canvas_w - 1).item()
            py = pts[pi, 1].long().clamp(0, canvas_h - 1).item()
            d = dist_t[py, px].item()
            if d < min_dist:
                min_dist = d
        
        if min_dist <= contour_threshold:
            contour_strokes.append(stroke)
            contour_idx.append(i)
        else:
            detail_strokes.append(stroke)
            detail_idx.append(i)
    
    print(f"[classify] contorno: {len(contour_strokes)} stroke "
          f"(min_dist ≤ {contour_threshold}px)")
    print(f"[classify] dettaglio interno: {len(detail_strokes)} stroke "
          f"(min_dist > {contour_threshold}px)")
    
    return contour_strokes, detail_strokes, contour_idx, detail_idx

def filter_strokes_by_mask(svg_strokes, mask, canvas_w, canvas_h,
                            keep_inside=True):
    """
    Opzionale: rimuove gli stroke i cui punti di controllo sono
    tutti fuori dalla maschera oggetto.
    """
    mask_t = torch.from_numpy((mask > 127).astype(np.float32))
    
    kept = []
    removed = 0
    
    for stroke in svg_strokes:
        pts = stroke.points.data
        inside_count = 0
        
        for pi in range(pts.shape[0]):
            px = pts[pi, 0].long().clamp(0, canvas_w - 1).item()
            py = pts[pi, 1].long().clamp(0, canvas_h - 1).item()
            if mask_t[py, px] > 0.5:
                inside_count += 1
        
        ratio_inside = inside_count / pts.shape[0]
        
        if keep_inside and ratio_inside > 0.3:
            kept.append(stroke)
        elif not keep_inside:
            kept.append(stroke)
        else:
            removed += 1
    
    print(f"[filter] stroke mantenuti: {len(kept)}/{len(svg_strokes)} "
          f"(rimossi {removed} fuori dalla maschera)")
    
    return kept