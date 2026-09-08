import os
import sys

# Aggiunge il percorso dell'SDK
sdk_path = os.path.join(os.path.dirname(__file__), 'u3v-sdk-2.3.2-python')
sys.path.append(sdk_path)

import u3v_cam

cams = u3v_cam.list_cameras()
if cams:
    cam_info = cams[0]
    cam_index = cam_info['index'] if isinstance(cam_info, dict) else 0
    
    # Inizializza e avvia la telecamera
    cam = u3v_cam.Camera(cam_index)
    cam.start()
    
    # Legge un fotogramma
    frame = cam.read_frame()
    if frame is not None:
        print("\n[SUCCESS] Fotogramma acquisito con successo!")
        if hasattr(frame, 'shape'):
            print(f"Risoluzione e canali: {frame.shape}")
    else:
        print("\n[WARNING] Nessun fotogramma ricevuto.")
        
    # Arresta e chiude il dispositivo
    cam.stop()
    cam.close()