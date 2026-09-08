import os
import cv2
import numpy as np

def stitch_captured_frames(output_dir):
    """Unisce i frame affiancandoli direttamente in sequenza orizzontale"""
    files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])
    if len(files) < 2:
        print("[STITCHING] Servono almeno 2 frame per l'unione.")
        return None

    parent_dir = os.path.dirname(output_dir)
    saved_panoramas = []

    # Scorre i file a blocchi di 4
    for i in range(0, len(files), 4):
        batch = files[i:i+4]
        if len(batch) < 2:
            continue

        images = [cv2.imread(f) for f in batch if cv2.imread(f) is not None]
        if len(images) < 2:
            continue

        print(f"[STITCHING] Unione lineare di {len(images)} frame per il blocco {i//4}...")
        
        # Uniforma l'altezza di tutte le immagini del blocco
        h_target = images[0].shape[0]
        processed_images = []
        for img in images:
            if img.shape[0] != h_target:
                img = cv2.resize(img, (int(img.shape[1] * h_target / img.shape[0]), h_target))
            processed_images.append(img)

        # Affiancamento orizzontale diretto dei frame
        pano = np.hstack(processed_images)

        block_idx = (i // 4) + 1
        output_path = os.path.join(parent_dir, f"profilo_lineare_{block_idx}.jpg")
        cv2.imwrite(output_path, pano, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"[STITCHING] Blocco lineare {block_idx} salvato in: {output_path}")
        saved_panoramas.append(output_path)

    return saved_panoramas