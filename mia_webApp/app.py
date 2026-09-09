import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory

# Importa i moduli esterni (rimosso lo stitcher)
from analyzer import ProfileAnalyzer

# Setup percorsi SDK fotocamera
sdk_path = os.path.join(os.path.dirname(__file__), 'u3v-sdk-2.3.2-python')
sys.path.append(sdk_path)
import u3v_cam

app = Flask(__name__)

# Setup Cartelle di lavoro (Aggiornato a test_images)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_images')
REF_DIR = os.path.join(os.path.dirname(__file__), 'reference_drawings')
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), 'analyzed_results')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MM_PER_PIXEL = None

# Inizializza l'analizzatore
analyzer = ProfileAnalyzer(REF_DIR, mm_per_pixel=MM_PER_PIXEL)

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
    while True:
        item = disk_queue.get()
        if item is None:
            break
        filepath, frame = item
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        disk_queue.task_done()

writer_thread = threading.Thread(target=disk_writer_worker, daemon=True)
writer_thread.start()

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stop_burst', methods=['POST'])
def stop_burst():
    with lock:
        capture_job['active'] = False
    return jsonify({'success': True, 'message': 'Acquisizione interrotta.'})

@app.route('/start_burst', methods=['POST'])
def start_burst():
    fps = float(request.form.get('fps', 1))
    duration = float(request.form.get('duration', 5))
    target_total = int(fps * duration)
    
    # Pulisce i file vecchi in test_images
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
        capture_job['last_capture_time'] = 0 
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
    camera = None
    if not cams:
        print("[ERRORE] Nessuna telecamera trovata. L'app web rimane attiva.")
    else:
        camera = u3v_cam.Camera(cams[0]['index'] if isinstance(cams[0], dict) else 0)
        camera.start()
        print("[SDK] Telecamera pronta per lo streaming ad alte prestazioni.")

    try:
        while True:
            if camera is None:
                time.sleep(1)
                continue

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
                            
                            filename = f"frame_{idx:04d}.jpg"
                            filepath = os.path.join(OUTPUT_DIR, filename)
                            
                            disk_queue.put((filepath, frame.copy()))
                            print(f"[HIGH-SPEED] Accodato frame {idx}/{target_total} in test_images")
                            
                            if idx >= target_total:
                                capture_job['active'] = False
                                print(f"[ACQUISIZIONE] Completata! Raggiunti {idx} frame.")
                                
                                def run_analysis(image_path):
                                    time.sleep(0.5) 
                                    print(f"\n[ANALISI] Avvio ispezione su {image_path}...")
                                    
                                    analyzer._load_references() 
                                    result = analyzer.analyze_image(image_path, ANALYSIS_DIR)
                                    
                                    if result.get("success"):
                                        print("="*60)
                                        print(" ANALISI COMPLETATA CON SUCCESSO!")
                                        print("="*60)
                                        print(f"✓ Disegno Teorico Più Simile : {result['best_match']}")
                                        print(f"✓ Numero Difetti Rilevati   : {result['defects_found']}")
                                        print("! Dettagli Difetti Trovati:")
                                        
                                        if result['defects_list']:
                                            for i, d in enumerate(result["defects_list"], 1):
                                                tipo = d.get('type', 'sconosciuto').upper()
                                                desc = d.get('description', 'Anomalia generica')
                                                print(f"  - Difetto #{i} [{tipo}]: {desc} "
                                                      f"[Coord: x={d['x']}, y={d['y']} | Dim: {d['width']}x{d['height']}px | Area: {d['area']}px²]")
                                        else:
                                            print("  - Nessun difetto rilevato (pezzo conforme).")
                                            
                                        print(f"\n✓ Immagine di output salvata in: {os.path.abspath(result['output_image'])}\n")
                                        print("="*60)

                                        # Invio dei dati e dell'immagine al server di monitoraggio
                                        try:
                                            server_url = "http://IL_TUO_SERVER_IP:PORTA/api/upload_report"
                                            payload = {
                                                "status": "success",
                                                "best_match": result['best_match'],
                                                "defects_count": result['defects_found'],
                                                "defects_list": result['defects_list']
                                            }
                                            output_img_path = result['output_image']
                                            with open(output_img_path, 'rb') as img_file:
                                                files = {'image': (os.path.basename(output_img_path), img_file, 'image/jpeg')}
                                                response = requests.post(server_url, data={'report_data': str(payload)}, files=files, timeout=5)
                                                if response.status_code == 200:
                                                    print("[SERVER] Report e immagine inviati con successo al server di monitoraggio.")
                                                else:
                                                    print(f"[SERVER] Risposta anomala dal server: {response.status_code}")
                                        except Exception as e:
                                            print(f"[SERVER] Errore di connessione al server di monitoraggio: {e}")

                                    else:
                                        print(f"[ERRORE ANALISI]: {result.get('error')}\n")

                                threading.Thread(target=run_analysis, args=(filepath,), daemon=True).start()

    except KeyboardInterrupt:
        if camera:
            camera.stop()
        print("\n[CHIUSURA] Telecamera fermata correttamente.")