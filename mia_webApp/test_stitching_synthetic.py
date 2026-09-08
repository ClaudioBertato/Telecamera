"""
test_stitching_synthetic.py (Modificato per immagini reali)
---------------------------------------------------------
Legge una sequenza di immagini inserite dall'utente in una cartella,
le unisce usando il ProfileStitcher e salva il risultato finale per
consentire una verifica visiva immediata.

USO:
    1. Crea una cartella chiamata 'immagini_test_reali' nella stessa directory dello script.
    2. Inserisci i fotogrammi sequenziali (es. frame_01.jpg, frame_02.jpg, ecc.).
    3. Esegui: python test_stitching_synthetic.py
"""

import os
import cv2
import numpy as np
from stitcher import ProfileStitcher

def main():
    # Nome della cartella in cui metterai le tue immagini
    input_folder = "immagini_test_reali"
    
    # Controlla se la cartella esiste, altrimenti la crea e avvisa l'utente
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"[ATTENZIONE] La cartella '{input_folder}' non esisteva ed è stata creata.")
        print(f"Metti le tue immagini dentro la cartella '{input_folder}' e riavvia lo script.")
        return

    # Cerca tutti i file immagine supportati all'interno della cartella
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ])

    if len(image_files) < 2:
        print(f"[ERRORE] Trovate solo {len(image_files)} immagini in '{input_folder}'.")
        print("Servono almeno 2 fotogrammi consecutivi sovrapposti per effettuare lo stitching.")
        return

    print(f"[INFO] Trovate {len(image_files)} immagini nella cartella '{input_folder}'. Avvio stitching...")

    # Inizializza il tuo stitcher (usa ORB di default, puoi passare detector="sift" se preferisci)
    stitcher = ProfileStitcher()
    
    # Esegue l'unione della sequenza
    panorama = stitcher.stitch_sequence(image_files)

    if panorama is None:
        print("[ERRORE] Lo stitching è fallito. Nessun panorama generato.")
        return

    # Salva il risultato finale nella cartella principale
    out_path = "risultato_stitching_reale.png"
    cv2.imwrite(out_path, panorama)
    
    print(f"[SUCCESSO] Panorama completato!")
    print(f"Dimensioni finali immagine cucita: {panorama.shape[1]}x{panorama.shape[0]} pixel")
    print(f"Risultato salvato con successo in: {out_path}")

if __name__ == "__main__":
    main()