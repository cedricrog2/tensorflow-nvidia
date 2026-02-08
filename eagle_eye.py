import os
# Silences technical noise but keeps critical ERRORS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import sys
import argparse
import time
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import exifread

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
OUTPUT_FOLDER = "output_eagle_eye" 

def check_hardware():
    print("="*50)
    print("      🦅 EAGLE EYE FORENSICS PRO 🦅")
    print("="*50)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[*] SUCCESS: GPU Detected! \n[*] Hardware: {gpus[0]}")
            print("[*] Mode: HIGH SPEED (CUDA Enabled)")
        except RuntimeError as e:
            print(e)
    else:
        print("[!] WARNING: No GPU detected. Mode: SLOW (CPU Only)")
    print("="*50)

def get_metadata(image_path):
    """Extracts forensic metadata before processing"""
    print(f"\n[--- METADATA ANALYSIS ---]")
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            print(f"[*] Camera:    {tags.get('Image Model', 'Unknown')}")
            print(f"[*] Software:  {tags.get('Image Software', 'Original')}")
            print(f"[*] DateTime:  {tags.get('Image DateTime', 'Unknown')}")
            
            lat = tags.get('GPS GPSLatitude')
            lon = tags.get('GPS GPSLongitude')
            if lat and lon:
                print(f"[*] GPS:       {lat}, {lon}")
            else:
                print("[*] GPS:       No Geotag found.")
    except Exception as e:
        print(f"[!] Metadata Error: {e}")
    print("-" * 50 + "\n")

def load_model():
    print(f"[*] Loading Forensic Model... (Wait for initialization)")
    return hub.load(MODEL_URL)

def preprocess_image(image):
    hr_image = tf.convert_to_tensor(image)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def enhance_frame(model, frame):
    """AI Upscaling + Forensic Sharpening"""
    # Step 1: AI Upscale
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pre_data = preprocess_image(rgb_frame)
    fake_image = model(pre_data)
    
    enhanced_img = tf.clip_by_value(fake_image, 0, 255)
    enhanced_img = tf.cast(enhanced_img, tf.uint8)
    enhanced_img = tf.squeeze(enhanced_img).numpy()
    
    # Correct BGR conversion for OpenCV
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    # Step 2: Forensic Sharpening (CLAHE)
    lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Step 3: Sharpness Kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    final = cv2.filter2D(final, -1, kernel)
    return final

def main():
    parser = argparse.ArgumentParser(description="EagleEye: Forensic Enhancer")
    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    args = parser.parse_args()

    check_hardware()

    if not os.path.exists(args.input):
        print(f"[!] Error: File {args.input} not found.")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    file_ext = os.path.splitext(args.input)[1].lower()
    if file_ext in ['.jpg', '.jpeg', '.png', '.tiff']:
        get_metadata(args.input)

    model = load_model()
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    
    if file_ext in video_exts:
        print(f"[*] Processing VIDEO: {args.input}")
        cap = cv2.VideoCapture(args.input)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        save_path = os.path.join(OUTPUT_FOLDER, "enhanced_" + os.path.basename(args.input))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*4, height*4))
        
        frame_count, start_time = 0, time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed = enhance_frame(model, frame)
            out.write(processed)
            frame_count += 1
            
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_proc = frame_count / elapsed
                sys.stdout.write(f"\r[*] Frame: {frame_count}/{total_frames} | Speed: {fps_proc:.2f} fps")
                sys.stdout.flush()
            
        cap.release()
        out.release()
    else:
        print(f"[*] Processing IMAGE: {args.input}")
        img = cv2.imread(args.input)
        enhanced_img = enhance_frame(model, img)
        save_path = os.path.join(OUTPUT_FOLDER, "forensic_" + os.path.basename(args.input) + ".png")
        cv2.imwrite(save_path, enhanced_img)

    print(f"\n[*] Done! Saved to: {save_path}")
    
    # Automatically open the output folder (Linux/Ubuntu command)
    subprocess.run(['xdg-open', OUTPUT_FOLDER])

if __name__ == "__main__":
    main()
