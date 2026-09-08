import os
import sys
import cv2
import numpy as np

sdk_path = os.path.join(os.path.dirname(__file__), 'u3v-sdk-2.3.2-python')
sys.path.append(sdk_path)
import u3v_cam

cams = u3v_cam.list_cameras()
print(f"\n1. Telecamere trovate: {len(cams)}")

if cams:
    cam_index = cams[0]['index'] if isinstance(cams[0], dict) else 0
    try:
        camera = u3v_cam.Camera(cam_index)
        camera.start()
        print("2. Telecamera avviata con successo!")
        
        for i in range(5):
            frame = camera.read_frame()
            if frame is None:
                print(f"   Frame {i}: NONE")
            else:
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                print(f"   Frame {i}: OK | Type: {frame.dtype} | Shape: {frame.shape}")
        
        camera.stop()
        camera.close()
        print("3. Telecamera chiusa e risorsa rilasciata.\n")
    except Exception as e:
        print(f"[ECCEZIONE]: {e}")