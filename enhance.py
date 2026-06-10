import pydiffvg
import torch
import cv2
import numpy as np
import sys

from details_utils import *
from external_edge_utils import *
from common_utils import *

def improve_strokes(image_name, refine_details=False, cut_mask=False) :


    # ── 1. Carica lo sketch ──────────────────────────────────────────
    canvas_w, canvas_h, svg_strokes, shape_groups = pydiffvg.svg_to_scene(
        "/kaggle/working/test_post/" + image_name + ".svg"
    )

    print(f"canvas: {canvas_w}x{canvas_h}, stroke: {len(svg_strokes)}")


    # ── 2. Prepara maschera e bordi ───────────────────────────────────
    mask = cv2.imread("/kaggle/working/test_post/mask_" + image_name + ".png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (canvas_w, canvas_h))

    # Bordi del contorno oggetto dalla maschera
    edges_mask = cv2.Canny(mask, 30, 100)

    # Dilata i bordi per creare una zona di attrazione più ampia
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges_mask, kernel, iterations=2)

    edge_map = cv2.GaussianBlur(edges_dilated.astype(np.float32), (5, 5), 1.5)
    edge_map = edge_map / (edge_map.max() + 1e-8)

    cv2.imwrite("/kaggle/working/test_post/edges_dilated_" + image_name + ".png", edges_dilated)

    print(f"edge_map: pixel nonzero={np.count_nonzero(edge_map > 0.1)}")


    # ── 3. Costruisci il campo di attrazione ─────────────────────────
    field_x, field_y, dist_map = build_attraction_field(
        (edges_mask > 0).astype(np.float32),
        canvas_w, canvas_h,
        sigma=12.0  # raggio di influenza in pixel — aumenta se i bordi sono lontani
    )

    if cut_mask:
        svg_strokes = filter_strokes_by_mask(
            svg_strokes, mask, canvas_w, canvas_h, keep_inside=True
        )


    # ── 4. Separa tratti di contorno e di dettaglio ────────────────── 
    contour_strokes, detail_strokes, contour_idx, detail_idx = classify_strokes(
        svg_strokes,
        dist_map,
        canvas_w, canvas_h,
        contour_threshold=15.0  # abbassa (es. 8) per spostare meno stroke,
                                # alza (es. 25) per spostarne di più
    )


    # ── 5. Snap stroke di contorno  ─────────────────────────────────
    refined_contour = snap_strokes_to_edges(
        contour_strokes, field_x, field_y, dist_map,
        canvas_w, canvas_h,
        max_displacement=12.0,
        n_iter=50,
        only_far_threshold=1.0
    )
    
# ── 8. Riassembla e salva ───────────────────────────────────────────
    all_strokes_ordered = [None] * len(svg_strokes)
    for i, stroke in zip(contour_idx, refined_contour):
        all_strokes_ordered[i] = stroke

    if refine_details:
        for i, stroke in zip(detail_idx, refined_detail):
            all_strokes_ordered[i] = stroke
    else:
        # Mantieni i detail strokes originali invariati
        for i, stroke in zip(detail_idx, detail_strokes):
            all_strokes_ordered[i] = stroke

    shape_groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0., 0., 0., 1.])
        )
        for i in range(len(all_strokes_ordered))
    ]

    pydiffvg.save_svg(
        "/kaggle/working/test_post/" + image_name + "_refined.svg",
        canvas_w, canvas_h,
        all_strokes_ordered, shape_groups
    )


image_name = sys.argv[1]
refine_details = sys.argv[2].lower() == "true"
cut_mask = sys.argv[3].lower() == "true"
improve_strokes(image_name, refine_details, cut_mask)
