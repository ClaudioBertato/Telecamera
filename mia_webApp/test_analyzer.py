import os
import cv2
from analyzer import ProfileAnalyzer

# ==============================================================================
# CONFIGURAZIONE CARTELLE
# ==============================================================================
# Cartella contenente i disegni teorici / CAD (.jpg, .png) inseriti da te
REF_DIR = os.path.join(os.path.dirname(__file__), 'reference_drawings')

# Cartella contenente le foto del profilo da analizzare (.jpg, .png) inserite da te
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

# Cartella dove verranno salvati i risultati con la sovrapposizione e i difetti cerchiati
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'analyzed_results')

# Se conosci il fattore di conversione pixel -> millimetri (es. misurando a mano una
# quota nota sia sul disegno CAD che sulla foto reale, in pixel, e dividendo i mm per
# quel numero di pixel) impostalo qui: le distanze verranno mostrate anche in mm.
# Lascialo a None per vedere le misure solo in pixel.
MM_PER_PIXEL = None

# Crea le cartelle se non esistono ancora
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("==================================================")
    print("       TEST ANALYZER - ISPEZIONE PROFILI          ")
    print("==================================================")

    # 1. Carica i disegni teorici dalla cartella reference_drawings
    print(f"\n[1] Caricamento disegni teorici da: '{REF_DIR}'...")
    analyzer = ProfileAnalyzer(REF_DIR, mm_per_pixel=MM_PER_PIXEL)

    if not analyzer.references:
        print(f"\n[ERRORE] Nessun disegno teorico trovato in '{REF_DIR}'.")
        print("-> Inserisci i tuoi file (.jpg o .png) dei disegni teorici in quella cartella e riavvia lo script.")
        return

    print(f"-> Caricati con successo {len(analyzer.references)} disegni teorici:")
    for ref_name in analyzer.references.keys():
        print(f"   - {ref_name}")

    # 2. Cerca le foto reali presenti nella cartella test_images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    photo_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)])

    print(f"\n[2] Ricerca foto da analizzare in: '{INPUT_DIR}'...")
    if not photo_files:
        print(f"\n[AVVISO] Nessuna foto trovata nella cartella '{INPUT_DIR}'.")
        print("-> Inserisci le foto del profilo da analizzare in quella cartella e riavvia lo script.")
        return

    print(f"-> Trovate {len(photo_files)} foto da analizzare.")

    # 3. Processa ogni foto confrontandola con i disegni teorici
    print("\n[3] Inizio analisi ed elaborazione...")
    print("-" * 50)

    for idx, photo_name in enumerate(photo_files, 1):
        photo_path = os.path.join(INPUT_DIR, photo_name)
        print(f"\n[{idx}/{len(photo_files)}] Analisi foto: {photo_name}")
        
        # Esegue il matching, l'allineamento e il calcolo dei difetti
        result = analyzer.analyze_image(photo_path, OUTPUT_DIR)

        if result.get("success"):
            print(f"  ✓ Disegno Teorico Più Simile  : {result['best_match']}")
            print(f"  ✓ Numero Difetti Rilevati      : {result['defects_found']}")
            
            if result['defects_found'] > 0:
                print("  ! Dettagli Difetti Trovati:")
                for d_idx, defect in enumerate(result['defects_list'], 1):
                    tipo = defect.get('type', 'sconosciuto').upper()
                    print(f"      - Difetto #{d_idx} [{tipo}]: {defect.get('description', 'Anomalia generica')}")
                    print(f"          Coordinate (x={defect['x']}, y={defect['y']}), "
                          f"Dimensione ({defect['width']}x{defect['height']}px), Area={defect['area']}px²")
            else:
                print("  ✓ Il pezzo è PERFETTO: nessuna differenza rispetto al disegno teorico.")

            print(f"  ✓ Immagine di output salvata in: {result['output_image']}")
        else:
            print(f"  ✗ Errore durante l'analisi: {result.get('error')}")

    print("\n==================================================")
    print("Analisi completata!")
    print(f"Tutti i risultati con i difetti evidenziati si trovano in: '{OUTPUT_DIR}'")
    print("==================================================")

if __name__ == '__main__':
    main()