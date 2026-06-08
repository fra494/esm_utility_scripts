import pydiffvg
import torch
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def build_internal_edge_map_enhanced(orig_img_path, mask, canvas_w, canvas_h,
                                      canny_low=20, canny_high=80,
                                      use_clahe=True,
                                      clahe_clip=2.0,        # era 4.0, abbassa
                                      min_edge_length=10,    # filtra bordi corti
                                      blur_before=3):        # sfuma prima di Canny
    orig = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (canvas_w, canvas_h))

    mask_norm = mask.astype(np.float32) / 255.0
    orig_masked = (orig.astype(np.float32) * mask_norm).astype(np.uint8)

    # Blur prima di CLAHE: riduce rumore ad alta frequenza
    if blur_before > 0:
        orig_masked = cv2.GaussianBlur(orig_masked, (blur_before, blur_before), 0)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        orig_enhanced = clahe.apply(orig_masked)
    else:
        orig_enhanced = orig_masked

    edges_orig = cv2.Canny(orig_enhanced, canny_low, canny_high)

    # Filtra i bordi corti (rumore) — tieni solo contorni lunghi
    if min_edge_length > 0:
        contours, _ = cv2.findContours(edges_orig, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_NONE)
        edges_filtered = np.zeros_like(edges_orig)
        kept, removed = 0, 0
        for cnt in contours:
            if len(cnt) >= min_edge_length:
                cv2.drawContours(edges_filtered, [cnt], -1, 255, 1)
                kept += 1
            else:
                removed += 1
        print(f"[edge filter] contorni tenuti: {kept}, rimossi (corti): {removed}")
        edges_orig = edges_filtered

    # Erodi maschera per escludere il contorno esterno
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=3)
    edges_internal = cv2.bitwise_and(edges_orig, edges_orig, mask=mask_eroded)

    print(f"[internal edges] pixel bordi interni: {np.count_nonzero(edges_internal)}")

    cv2.imwrite("debug_orig_enhanced.png", orig_enhanced)
    cv2.imwrite("debug_edges_internal.png", edges_internal)

    return edges_internal, orig_enhanced


def build_nearest_edge_index(edges_internal):
    """
    Per ogni pixel restituisce le coordinate (row, col) del pixel di bordo
    più vicino — indipendentemente dalla distanza.
    """
    edge_binary = (edges_internal > 0).astype(np.uint8)
    dist_map, nearest_idx = distance_transform_edt(
        1 - edge_binary, return_indices=True
    )
    # nearest_idx[0] = riga più vicina, nearest_idx[1] = colonna più vicina
    return dist_map, nearest_idx



def snap_detail_strokes_nearest(detail_strokes, edges_internal,
                                 canvas_w, canvas_h,
                                 max_displacement_per_iter=3.0,
                                 n_iter=80,
                                 snap_threshold=30.0,
                                 only_far_threshold=2.0,
                                 stretch_along_edge=True):
    """
    Versione robusta: ogni punto conosce esattamente dove si trova
    il bordo più vicino, indipendentemente dalla distanza.
    Non usa sigma — nessun decay esponenziale.

    snap_threshold: sposta solo punti entro questa distanza dal bordo
                    (evita di spostare stroke lontanissimi che non
                    appartengono a nessun dettaglio visibile)
    max_displacement_per_iter: spostamento massimo per iterazione in px
                    (piccolo = movimento graduale e stabile)
    """
    dist_map, nearest_idx = build_nearest_edge_index(edges_internal)
    dist_t = torch.from_numpy(dist_map.astype(np.float32))

    # Tangenti al bordo per stretch_along_edge
    gy, gx = np.gradient(dist_map)
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    tangent_x = (-gy / mag).astype(np.float32)
    tangent_y = ( gx / mag).astype(np.float32)
    tang_x_t = torch.from_numpy(tangent_x)
    tang_y_t = torch.from_numpy(tangent_y)

    for iteration in range(n_iter):
        moved = 0

        for stroke in detail_strokes:
            pts = stroke.points.data  # [N, 2]

            for pi in range(pts.shape[0]):
                px = int(pts[pi, 0].clamp(0, canvas_w - 1).item())
                py = int(pts[pi, 1].clamp(0, canvas_h - 1).item())

                d = dist_t[py, px].item()

                # Troppo lontano: nessun bordo interno rilevante vicino
                if d > snap_threshold:
                    continue
                # Già sul bordo
                if d < only_far_threshold:
                    continue

                # Coordinate del bordo più vicino
                target_row = nearest_idx[0][py, px]
                target_col = nearest_idx[1][py, px]

                # Vettore diretto verso il bordo (non normalizzato dal campo)
                dx_to_edge = float(target_col) - pts[pi, 0].item()
                dy_to_edge = float(target_row) - pts[pi, 1].item()

                # Normalizza e scala al passo massimo
                dist_to_edge = (dx_to_edge**2 + dy_to_edge**2)**0.5 + 1e-8
                dx = (dx_to_edge / dist_to_edge) * min(d, max_displacement_per_iter)
                dy = (dy_to_edge / dist_to_edge) * min(d, max_displacement_per_iter)

                # Componente tangenziale opzionale
                if stretch_along_edge and d < 10.0:
                    tx = tang_x_t[py, px].item()
                    ty = tang_y_t[py, px].item()
                    next_pi = (pi + 1) % pts.shape[0]
                    npx = pts[next_pi, 0].item() - pts[pi, 0].item()
                    npy = pts[next_pi, 1].item() - pts[pi, 1].item()
                    nmag = (npx**2 + npy**2)**0.5 + 1e-8
                    sign = 1.0 if (tx * npx/nmag + ty * npy/nmag) > 0 else -1.0
                    dx += sign * 0.4 * tx * min(d, max_displacement_per_iter)
                    dy += sign * 0.4 * ty * min(d, max_displacement_per_iter)

                pts[pi, 0] = (pts[pi, 0] + dx).clamp(0, canvas_w)
                pts[pi, 1] = (pts[pi, 1] + dy).clamp(0, canvas_h)
                moved += 1

        if iteration % 10 == 0:
            all_dists = []
            for stroke in detail_strokes:
                pts = stroke.points.data
                for pi in range(pts.shape[0]):
                    px = int(pts[pi, 0].clamp(0, canvas_w - 1).item())
                    py = int(pts[pi, 1].clamp(0, canvas_h - 1).item())
                    all_dists.append(dist_t[py, px].item())
                    
                mean_d = np.mean(all_dists)
                
            #print(f"[detail nearest] iter {iteration:3d} | "
                  #f"dist media: {mean_d:.2f}px | mossi: {moved}")

        if moved == 0:
            print(f"[detail nearest] convergenza a iter {iteration}")
            break

    return detail_strokes