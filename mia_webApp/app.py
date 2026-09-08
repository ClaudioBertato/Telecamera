import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
from flask import Flask, render_template, request, jsonify, send_from_directory

# Importa la funzione di stitching dal modulo esterno stitcher.py
from stitcher import stitch_captured_frames

sdk_path = os.path.join(os.path.dirname(__file__), 'u3v-sdk-2.3.2-python')
sys.path.append(sdk_path)

import u3v_cam

app = Flask(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'captured_frames')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coda thread-safe per la scrittura asincrona su disco
disk_queue = queue.Queue()

capture_job = {
    'active': False,
    'interval': 1.0,
    'target_total': 0,
    'count': 0,
    'last_capture_time': 0,
    'duration': 2
}
lock = threading.Lock()

def disk_writer_worker():
    """Thread dedicato esclusivamente a scrivere le immagini su disco senza bloccare la telecamera"""
    while True:
        item = disk_queue.get()
        if item is None:
            break
        filepath, frame = item
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        disk_queue.task_done()

# Avvia il worker di scrittura su disco in background
writer_thread = threading.Thread(target=disk_writer_worker, daemon=True)
writer_thread.start()

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_burst', methods=['POST'])
def start_burst():
    fps = float(request.form.get('fps', 1))
    duration = float(request.form.get('duration', 5))
    target_total = int(fps * duration)
    
    # Pulisce i file vecchi
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.jpg'):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except:
                pass

    with lock:
        capture_job['interval'] = 1.0 / fps
        capture_job['target_total'] = target_total
        capture_job['count'] = 0
        capture_job['duration'] = duration
        capture_job['last_capture_time'] = 0  # Azzerato per far partire subito il primo scatto
        capture_job['active'] = True

    return jsonify({'success': True, 'target_total': target_total})

@app.route('/get_status')
def get_status():
    with lock:
        is_active = capture_job['active']
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    return jsonify({'active': is_active, 'files': files})

@app.route('/frames/<filename>')
def get_frame_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[FLASK] Server attivo su http://127.0.0.1:5000")

    cams = u3v_cam.list_cameras()
    if not cams:
        print("[ERRORE] Nessuna telecamera trovata.")
        sys.exit(1)

    camera = u3v_cam.Camera(cams[0]['index'] if isinstance(cams[0], dict) else 0)
    camera.start()
    print("[SDK] Telecamera pronta per lo streaming ad alte prestazioni.")

    try:
        while True:
            try:
                frame = camera.read_frame()
            except Exception:
                time.sleep(0.001)
                continue

            if frame is None:
                continue

            with lock:
                is_active = capture_job['active']
                interval = capture_job['interval']
                last_time = capture_job['last_capture_time']
                target_total = capture_job['target_total']

            if is_active:
                current_time = time.time()
                if current_time - last_time >= interval:
                    if not isinstance(frame, np.ndarray):
                        frame = np.array(frame)
                    
                    if frame.size > 0:
                        if frame.dtype != np.uint8:
                            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        
                        with lock:
                            capture_job['count'] += 1
                            idx = capture_job['count']
                            capture_job['last_capture_time'] = current_time
                            
                            # Controllo di arresto immediato al raggiungimento del target
                            if idx >= target_total:
                                capture_job['active'] = False
                                print(f"[ACQUISIZIONE] Completata! Raggiunti {idx} frame.")
                                
                                # Avvia lo stitching in background al termine della sequenza
                                threading.Thread(target=stitch_captured_frames, args=(OUTPUT_DIR,), daemon=True).start()

                        filename = f"frame_{idx:04d}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        disk_queue.put((filepath, frame.copy()))
                        print(f"[HIGH-SPEED] Accodato frame {idx}/{target_total}")

    except KeyboardInterrupt:
        camera.stop()