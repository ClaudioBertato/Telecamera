"""
test_stitching_real.py
-----------------------
Script standalone per testare l'algoritmo di stitching su foto REALI del
profilo metallico, senza dover avviare Flask ne' collegare la telecamera.

Basta avere una cartella con qualche foto (es. quelle gia' salvate da una
precedente acquisizione in "captured_frames/", oppure delle foto scattate
a mano muovendoti lungo il pezzo).

USO:
    python test_stitching_real.py cartella_con_le_foto
    python test_stitching_real.py cartella_con_le_foto --output risultato.jpg
    python test_stitching_real.py cartella_con_le_foto --detector sift
    python test_stitching_real.py cartella_con_le_foto --show

Richiede che stitcher.py sia nella stessa cartella di questo script
(o comunque nel PYTHONPATH).
"""

import argparse
import os
import sys
import cv2

from stitcher import ProfileStitcher


def find_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def main():
    parser = argparse.ArgumentParser(description="Test dello stitching su foto reali")
    parser.add_argument("folder", help="Cartella con le foto da unire (ordinate alfabeticamente)")
    parser.add_argument("--output", default="test_risultato.jpg", help="Percorso del file di output")
    parser.add_argument("--detector", default="orb", choices=["orb", "sift"], help="Feature detector da usare")
    parser.add_argument("--show", action="store_true",
                         help="Mostra il risultato in una finestra (richiede un ambiente grafico)")
    args = parser.parse_args()

    images = find_images(args.folder)
    if len(images) < 2:
        print(f"Servono almeno 2 immagini in '{args.folder}', trovate: {len(images)}")
        sys.exit(1)

    print(f"Trovate {len(images)} immagini:")
    for p in images:
        print("  -", os.path.basename(p))

    stitcher = ProfileStitcher(detector=args.detector)
    panorama = stitcher.stitch_sequence(images)

    if panorama is None:
        print("Stitching fallito: nessuna immagine valida trovata.")
        sys.exit(1)

    cv2.imwrite(args.output, panorama, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Salvato il profilo unito in: {args.output}  "
          f"(dimensioni: {panorama.shape[1]}x{panorama.shape[0]})")

    if args.show:
        cv2.imshow("Risultato stitching", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
