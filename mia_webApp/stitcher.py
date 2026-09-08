"""
stitcher.py
-----------
Modulo per l'unione (stitching) automatica di profili metallici e disegni CAD.
Utilizza il Consenso Geometrico dei Contorni (Contour Centroid Voting) per 
riconoscere e sovrapporre con precisione millimetrica fori, asole e cerchi.
"""

import os
import cv2
import numpy as np


class ProfileStitcher:
    """
    Riconosce i gruppi di forme geometriche sovrapposte tra fotogrammi consecutivi,
    calcola lo spostamento 2D esatto al pixel ed unisce i profili senza vuoti né duplicati.
    """

    def __init__(self):
        pass

    def _extract_geometric_features(self, img):
        """Estrae i baricentri e le dimensioni dei contorni (fori/asole/cerchi)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        
        # Inversione: sfondo bianco -> 0, linee/fori neri -> 255
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        max_area = (h * w) * 0.7  # Scarta il rettangolo di bordo esterno

        features = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 10 < area < max_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    features.append({
                        'cx': cx, 'cy': cy, 'w': bw, 'h': bh, 'area': area
                    })

        return features, gray

    def _find_exact_shift(self, img1, img2):
        feats1, gray1 = self._extract_geometric_features(img1)
        feats2, gray2 = self._extract_geometric_features(img2)
        h1, w1 = gray1.shape[:2]

        # 1. Voting System sui vettori di spostamento (dx, dy) delle forme coincidenti
        votes = {}
        for f1 in feats1:
            for f2 in feats2:
                # Verifica se le due forme hanno dimensioni compatibili (tolleranza 4px)
                if abs(f1['w'] - f2['w']) <= 4 and abs(f1['h'] - f2['h']) <= 4:
                    dx = int(round(f1['cx'] - f2['cx']))
                    dy = int(round(f1['cy'] - f2['cy']))

                    # Frame 2 si trova a destra di Frame 1 (dx > 0)
                    if 10 <= dx < w1 and abs(dy) < 40:
                        bin_key = (dx // 2 * 2, dy // 2 * 2)
                        votes[bin_key] = votes.get(bin_key, 0) + 1

        # Se il voto sulle forme dà una corrispondenza chiara, usa quel vettore
        if votes:
            best_bin = max(votes, key=votes.get)
            if votes[best_bin] >= 2:
                return best_bin[0], best_bin[1]

        # 2. Fallback: Template Matching su gradiente verticale se ci sono poche forme
        sobel1 = np.abs(cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3))
        sobel2 = np.abs(cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3))

        tpl_w = int(w1 * 0.35)
        template = sobel1[:, w1 - tpl_w:]
        res = cv2.matchTemplate(sobel2, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        match_x, match_y = max_loc
        dx = (w1 - tpl_w) - match_x
        dy = match_y

        if dx <= 0 or dx >= w1:
            dx = int(w1 * 0.5)
            dy = 0

        return dx, dy

    def _stitch_pair(self, panorama, reference, next_img):
        w_pano = panorama.shape[1]
        w_ref = reference.shape[1]
        h_ref = reference.shape[0]

        h_next, w_next = next_img.shape[:2]

        # Mantiene uniforme l'altezza delle immagini
        if h_next != h_ref:
            scale = h_ref / float(h_next)
            w_next = max(1, int(round(w_next * scale)))
            next_img = cv2.resize(next_img, (w_next, h_ref))

        dx, dy = self._find_exact_shift(reference, next_img)

        # Correzione dello scostamento verticale con sfondo bianco di riempimento
        if dy != 0:
            M = np.float32([[1, 0, 0], [0, 1, -dy]])
            next_img = cv2.warpAffine(
                next_img, M, (w_next, h_ref),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

        # Calcolo del punto esatto di giunzione nel panorama complessivo
        pano_cut_x = (w_pano - w_ref) + dx

        if 0 < pano_cut_x < w_pano:
            left_part = panorama[:, :pano_cut_x]
            new_panorama = np.hstack([left_part, next_img])
        else:
            new_panorama = np.hstack([panorama, next_img])

        return new_panorama, next_img

    def stitch_sequence(self, image_paths):
        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)

        if not images:
            return None
        if len(images) == 1:
            return images[0]

        panorama = images[0]
        reference = images[0]
        for next_img in images[1:]:
            panorama, reference = self._stitch_pair(panorama, reference, next_img)

        return panorama


def stitch_captured_frames(output_dir, batch_size=None):
    files = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if len(files) < 2:
        return []

    parent_dir = os.path.dirname(output_dir)
    stitcher = ProfileStitcher()
    saved_panoramas = []

    batches = [files] if batch_size is None else [
        files[i:i + batch_size] for i in range(0, len(files), batch_size)
    ]

    for idx, batch in enumerate(batches, start=1):
        if len(batch) < 2:
            continue
        panorama = stitcher.stitch_sequence(batch)
        if panorama is None:
            continue
        output_path = os.path.join(parent_dir, f"profilo_finale_{idx}.jpg")
        cv2.imwrite(output_path, panorama, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_panoramas.append(f"profilo_finale_{idx}.jpg")

    return saved_panoramas