import os
import cv2
import numpy as np


class ProfileAnalyzer:
    def __init__(self, references_dir, mm_per_pixel=None, hole_match_tolerance_px=30):
        self.references_dir = references_dir
        self.references = {}
        self.mm_per_pixel = mm_per_pixel
        self.hole_match_tolerance_px = hole_match_tolerance_px
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._load_references()

    def _prepare_image(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return gray

    def _draw_banner(self, img, text, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        h, w = img.shape[:2]
        banner_height = 24
        cv2.rectangle(img, (0, 0), (w, banner_height), bg_color, -1)
        cv2.putText(img, text, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    def _fmt_len(self, px):
        if self.mm_per_pixel:
            return f"{px:.1f}px ({px * self.mm_per_pixel:.2f}mm)"
        return f"{px:.1f}px"

    def _direction_label(self, dx, dy, min_component=2.0):
        parts = []
        if abs(dy) >= min_component:
            parts.append("verso il basso" if dy > 0 else "verso l'alto")
        if abs(dx) >= min_component:
            parts.append("verso destra" if dx > 0 else "verso sinistra")
        if not parts:
            return "leggermente fuori posizione"
        return " e ".join(parts)

    def _cluster_points(self, raw_points, dist_thresh=8):
        clusters = []
        used = [False] * len(raw_points)
        for i, h in enumerate(raw_points):
            if used[i]:
                continue
            group = [h]
            used[i] = True
            for j in range(i + 1, len(raw_points)):
                if used[j]:
                    continue
                h2 = raw_points[j]
                if abs(h['cx'] - h2['cx']) < dist_thresh and abs(h['cy'] - h2['cy']) < dist_thresh:
                    group.append(h2)
                    used[j] = True
            best = max(group, key=lambda d: d['area'])
            clusters.append(best)
        return clusters

    def _filter_hole_contours(self, contours, img_shape, min_area, max_area_frac, min_circularity):
        h_img, w_img = img_shape[:2]
        raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if w > w_img * 0.4 or h > h_img * 0.6:
                continue
            if area < min_area or area > (w_img * h_img * max_area_frac):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < min_circularity:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            raw.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area, 'circ': circularity})
        return self._cluster_points(raw)

    def _detect_holes_cad(self, gray, min_area=20, max_area_frac=0.03, min_circularity=0.3):
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = self._filter_hole_contours(contours, gray.shape, min_area, max_area_frac, min_circularity)
        for hle in holes:
            hle['shape'] = self._hole_shape(hle['w'], hle['h'])
        return holes

    def _detect_holes_photo(self, gray, mask=None, min_area=20, max_area_frac=0.03, min_circularity=0.3, max_mean_intensity=90):
        work = gray.copy()
        if mask is not None:
            work[mask == 0] = 255
        
        blurred = cv2.GaussianBlur(work, (5, 5), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 31, 5)
        binary = cv2.bitwise_and(thresh1, thresh2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = gray.shape[:2]
        raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if w > w_img * 0.4 or h > h_img * 0.6:
                continue
            if area < min_area or area > (w_img * h_img * max_area_frac):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < min_circularity:
                continue
            
            c_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], 0, 255, -1)
            mean_val = cv2.mean(work, mask=c_mask)[0]
            if mean_val > max_mean_intensity:
                continue
                
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            raw.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area, 'circ': circularity})
            
        holes = self._cluster_points(raw)
        for hle in holes:
            hle['shape'] = self._hole_shape(hle['w'], hle['h'])
        return holes

    def _is_plausible_transform(self, M, num_inliers, max_rotation_deg=25, scale_range=(0.4, 2.0), min_inliers=6):
        if M is None or num_inliers < min_inliers:
            return False
        a, b = M[0, 0], M[1, 0]
        scale = np.hypot(a, b)
        if not (scale_range[0] <= scale <= scale_range[1]):
            return False
        rotation_deg = abs(np.degrees(np.arctan2(b, a)))
        rotation_deg = min(rotation_deg, 360 - rotation_deg)
        if rotation_deg > max_rotation_deg:
            return False
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if det < 0:
            return False
        return True

    def _hole_shape(self, w, h):
        ratio = w / max(1, h)
        if 0.7 <= ratio <= 1.4:
            return "circolare"
        return "asola/ovale"

    def _match_holes(self, cad_holes, real_holes):
        pairs = []
        for i, c in enumerate(cad_holes):
            for j, r in enumerate(real_holes):
                d = float(np.hypot(c['cx'] - r['cx'], c['cy'] - r['cy']))
                if d <= self.hole_match_tolerance_px:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda p: p[0])

        used_cad, used_real = set(), set()
        matched = []
        for d, i, j in pairs:
            if i in used_cad or j in used_real:
                continue
            used_cad.add(i)
            used_real.add(j)
            matched.append((i, j, d))

        missing = [i for i in range(len(cad_holes)) if i not in used_cad]
        extra = [j for j in range(len(real_holes)) if j not in used_real]
        return matched, missing, extra

    def _analyze_holes(self, cad_holes, real_holes, position_tolerance_px=6.0, size_tolerance_ratio=0.3):
        matched, missing, extra = self._match_holes(cad_holes, real_holes)
        defects = []
        covered_regions = []

        for i, j, d in matched:
            c = cad_holes[i]
            r = real_holes[j]
            radius = max(c['w'], c['h']) / 2 + 6
            covered_regions.append((c['cx'], c['cy'], radius + d))

            dx = r['cx'] - c['cx']
            dy = r['cy'] - c['cy']

            if d > position_tolerance_px:
                direction = self._direction_label(dx, dy)
                defects.append({
                    "type": "foro_spostato",
                    "x": int(r['cx']), "y": int(r['cy']),
                    "width": r['w'], "height": r['h'], "area": r['area'],
                    "description": (
                        f"Foro spostato di {self._fmt_len(d)} {direction} rispetto alla posizione "
                        f"teorica (atteso in x={c['cx']:.0f}, y={c['cy']:.0f})"
                    )
                })

            expected_diam = (c['w'] + c['h']) / 2
            detected_diam = (r['w'] + r['h']) / 2
            if expected_diam > 0:
                size_diff_ratio = abs(detected_diam - expected_diam) / expected_diam
                if size_diff_ratio > size_tolerance_ratio:
                    bigger_smaller = "più grande" if detected_diam > expected_diam else "più piccolo"
                    defects.append({
                        "type": "foro_fuori_tolleranza",
                        "x": int(r['cx']), "y": int(r['cy']),
                        "width": r['w'], "height": r['h'], "area": r['area'],
                        "description": (
                            f"Foro {bigger_smaller} rispetto al disegno: atteso Ø~{self._fmt_len(expected_diam)}, "
                            f"rilevato Ø~{self._fmt_len(detected_diam)}"
                        )
                    })

        for i in missing:
            c = cad_holes[i]
            radius = max(c['w'], c['h']) / 2 + 6
            covered_regions.append((c['cx'], c['cy'], radius))
            defects.append({
                "type": "foro_mancante",
                "x": int(c['cx']), "y": int(c['cy']),
                "width": c['w'], "height": c['h'], "area": c['area'],
                "description": f"Foro {c['shape']} mancante in x={c['cx']:.0f}, y={c['cy']:.0f}"
            })

        for j in extra:
            r = real_holes[j]
            radius = max(r['w'], r['h']) / 2 + 6
            covered_regions.append((r['cx'], r['cy'], radius))
            defects.append({
                "type": "foro_extra",
                "x": int(r['cx']), "y": int(r['cy']),
                "width": r['w'], "height": r['h'], "area": r['area'],
                "description": f"Foro non previsto dal disegno in x={r['cx']:.0f}, y={r['cy']:.0f}"
            })

        return defects, covered_regions

    def _classify_surface_defect(self, x, y, w_box, h_box, area, cad_w, cad_h):
        if area > (cad_w * cad_h * 0.25):
            return "Disallineamento globale o scostamento importante del profilo"
        aspect_ratio = float(w_box) / max(1, h_box)
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            return "Possibile sbavatura lineare / taglio incompleto lungo il bordo"
        return "Irregolarità superficiale / tolleranza non conforme (graffio o ammaccatura)"

    def _remove_covered_regions(self, thresh, covered_regions):
        if not covered_regions:
            return thresh
        mask = np.zeros_like(thresh)
        for cx, cy, radius in covered_regions:
            cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
        return cv2.bitwise_and(thresh, cv2.bitwise_not(mask))

    def _load_references(self):
        if not os.path.exists(self.references_dir):
            os.makedirs(self.references_dir)
            return

        self.references.clear()
        for filename in os.listdir(self.references_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(self.references_dir, filename)
                raw_img = cv2.imread(path)
                if raw_img is not None:
                    gray_cad = self._prepare_image(raw_img)
                    if np.mean(gray_cad) < 127:
                        gray_cad = cv2.bitwise_not(gray_cad)

                    edges = cv2.Canny(gray_cad, 50, 150)
                    kp, des = self.orb.detectAndCompute(edges, None)
                    holes = self._detect_holes_cad(gray_cad)

                    self.references[filename] = {
                        'raw': raw_img,
                        'cad_white': gray_cad,
                        'edges': edges,
                        'keypoints': kp,
                        'descriptors': des,
                        'holes': holes
                    }
        print(f"[ANALYZER] Caricati {len(self.references)} disegni di riferimento.")

    def analyze_image(self, real_image_path, output_dir, overlay_mode='multi_panel'):
        real_img_color = cv2.imread(real_image_path)
        if real_img_color is None:
            return {"error": "Immagine reale non trovata."}

        h_real, w_real = real_img_color.shape[:2]
        real_gray = cv2.cvtColor(real_img_color, cv2.COLOR_BGR2GRAY)
        real_edges = cv2.Canny(real_gray, 50, 150)
        kp_real, des_real = self.orb.detectAndCompute(real_edges, None)

        if des_real is None or len(self.references) == 0:
            return {"error": "Impossibile estrarre feature dalla foto o nessun CAD presente."}

        best_match_name = None
        max_inliers = 0
        best_matrix = None

        for name, ref_data in self.references.items():
            if ref_data['descriptors'] is None or len(ref_data['descriptors']) < 4:
                continue

            matches = self.matcher.match(des_real, ref_data['descriptors'])
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:int(len(matches) * 0.4)]

            if len(good_matches) >= 4:
                src_pts = np.float32([kp_real[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref_data['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

                if M is not None and inliers is not None:
                    num_inliers = int(np.sum(inliers))
                    if self._is_plausible_transform(M, num_inliers) and num_inliers > max_inliers:
                        max_inliers = num_inliers
                        best_match_name = name
                        best_matrix = M

        if best_match_name is None:
            best_match_name = list(self.references.keys())[0]

        ref_data = self.references[best_match_name]
        cad_white = ref_data['cad_white']
        h_cad, w_cad = cad_white.shape[:2]

        mask_real = np.ones((h_real, w_real), dtype=np.uint8) * 255

        if best_matrix is not None:
            aligned_photo = cv2.warpAffine(real_img_color, best_matrix, (w_cad, h_cad), borderValue=(255, 255, 255))
            aligned_mask = cv2.warpAffine(mask_real, best_matrix, (w_cad, h_cad), borderValue=0)
        else:
            scale_w = w_cad / w_real
            new_w = w_cad
            new_h = int(h_real * scale_w)
            resized_photo = cv2.resize(real_img_color, (new_w, new_h))
            
            aligned_photo = np.ones((h_cad, w_cad, 3), dtype=np.uint8) * 255
            aligned_mask = np.zeros((h_cad, w_cad), dtype=np.uint8)
            
            y_off = (h_cad - new_h) // 2
            if new_h <= h_cad:
                aligned_photo[y_off:y_off+new_h, 0:w_cad] = resized_photo
                aligned_mask[y_off:y_off+new_h, 0:w_cad] = 255
            else:
                cropped = resized_photo[(new_h - h_cad)//2:(new_h - h_cad)//2 + h_cad, 0:w_cad]
                aligned_photo[:, :] = cropped
                aligned_mask[:, :] = 255

        cad_bgr = cv2.cvtColor(cad_white, cv2.COLOR_GRAY2BGR)
        cad_patched = cad_bgr.copy()
        idx_valid = aligned_mask > 0
        cad_patched[idx_valid] = aligned_photo[idx_valid]

        aligned_gray = cv2.cvtColor(aligned_photo, cv2.COLOR_BGR2GRAY)

        valid_cad_holes = []
        for c in ref_data['holes']:
            cx, cy = int(c['cx']), int(c['cy'])
            if 0 <= cy < h_cad and 0 <= cx < w_cad:
                if aligned_mask[cy, cx] > 0:
                    valid_cad_holes.append(c)

        real_holes = self._detect_holes_photo(aligned_gray, mask=aligned_mask)
        hole_defects, covered_regions = self._analyze_holes(valid_cad_holes, real_holes)

        diff = cv2.absdiff(cad_white, aligned_gray)
        _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=aligned_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = self._remove_covered_regions(thresh, covered_regions)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        surface_defects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 120:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if area > (w_cad * h_cad * 0.8):
                    continue

                description = self._classify_surface_defect(x, y, w_box, h_box, area, w_cad, h_cad)
                surface_defects.append({
                    "type": "superficie",
                    "x": x, "y": y, "width": w_box, "height": h_box,
                    "area": area, "description": description
                })

        defects = hole_defects + surface_defects

        defect_layer = cad_patched.copy()
        type_colors = {
            "foro_mancante": (0, 0, 255),
            "foro_extra": (255, 0, 255),
            "foro_spostato": (0, 140, 255),
            "foro_fuori_tolleranza": (0, 255, 255),
            "superficie": (0, 0, 200),
        }
        type_labels = {
            "foro_mancante": "MANCANTE",
            "foro_extra": "EXTRA",
            "foro_spostato": "SPOSTATO",
            "foro_fuori_tolleranza": "FUORI TOLL.",
            "superficie": "DIFETTO",
        }

        for d in defects:
            color = type_colors.get(d["type"], (0, 0, 255))
            label = type_labels.get(d["type"], "DIFETTO")
            x, y, w_box, h_box = d["x"], d["y"], d["width"], d["height"]
            if d["type"] in ("foro_mancante", "foro_extra", "foro_spostato", "foro_fuori_tolleranza"):
                cx, cy = int(x), int(y)
                axis_w = int(w_box / 2) + 4
                axis_h = int(h_box / 2) + 4
                cv2.ellipse(defect_layer, (cx, cy), (axis_w, axis_h), 0, 0, 360, color, 2)
                cv2.putText(defect_layer, label, (cx - axis_w, max(15, cy - axis_h - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(defect_layer, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(defect_layer, label, (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        if overlay_mode == 'cad_patch':
            final_visual = defect_layer
        else:
            p1 = cad_bgr.copy()
            self._draw_banner(p1, "1. DISEGNO TEORICO (CAD COMPLETO)", bg_color=(50, 50, 50))

            p2 = cad_patched.copy()
            self._draw_banner(p2, "2. FOTO REALE INCOLLATA SUL CAD", bg_color=(0, 120, 0))

            p3 = defect_layer.copy()
            self._draw_banner(p3, f"3. ISPEZIONE E DIFETTI TROVATI ({len(defects)})", bg_color=(0, 0, 180))

            final_visual = np.vstack([p1, p2, p3])

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(real_image_path)
        output_filepath = os.path.join(output_dir, f"analyzed_{base_name}")
        cv2.imwrite(output_filepath, final_visual)

        return {
            "success": True,
            "best_match": best_match_name,
            "defects_found": len(defects),
            "defects_list": defects,
            "output_image": output_filepath
        }