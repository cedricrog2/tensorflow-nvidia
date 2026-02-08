import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import exifread

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def check_hardware():
    print("="*50)
    print("      🦅 EAGLE EYE FORENSICS: PRO EDITION 🦅")
    print("="*50)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[*] GPU: Detected ({gpus[0].name})")
        except RuntimeError as e:
            print(e)
    else:
        print("[!] WARNING: No GPU detected. Running on CPU.")

def get_metadata(image_path):
    """Extracts and prints Forensic Metadata (EXIF)"""
    print(f"\n[--- METADATA ANALYSIS: {os.path.basename(image_path)} ---]")
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Key Forensic Fields
            camera = tags.get('Image Model', 'Unknown')
            software = tags.get('Image Software', 'Original/Unknown')
            datetime = tags.get('Image DateTime', 'Unknown')
            
            print(f"[*] Camera Model: {camera}")
            print(f"[*] Software:     {software}")
            print(f"[*] Timestamp:    {datetime}")

            # GPS Extraction
            lat = tags.get('GPS GPSLatitude')
            lon = tags.get('GPS GPSLongitude')
            if lat and lon:
                print(f"[*] GPS COORDS:   {lat}, {lon}")
                print(f"[*] Location found. Cross-reference in Maps.")
            else:
                print("[*] GPS COORDS:   No Geotag available.")
    except Exception as e:
        print(f"[!] Metadata Error: {e}")
    print("-" * 50 + "\n")

def preprocess_image(image):
    hr_image = tf.convert_to_tensor(image)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def enhance_frame(model, frame):
    # 1. AI Upscaling (TensorFlow)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pre_data = preprocess_image(rgb_frame)
    fake_image = model(pre_data)
    
    enhanced_img = tf.clip_by_value(fake_image, 0, 255)
    enhanced_img = tf.cast(enhanced_img, tf.uint8)
    enhanced_img = tf.squeeze(enhanced_img).numpy()
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    # 2. Forensic Post-Processing (OpenCV)
    lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    final = cv2.filter2D(final, -1, kernel)
    return final

def main():
    parser = argparse.ArgumentParser(description="EagleEye: Forensic Enhancer Pro")
    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    args = parser.parse_args()

    check_hardware()

    if not os.path.exists(args.input):
        print(f"[!] Error: File {args.input} not found.")
        return

    # Run Metadata Analysis first if it's an image
    file_ext = os.path.splitext(args.input)[1].lower()
    if file_ext in ['.jpg', '.jpeg', '.tiff', '.png']:
        get_metadata(args.input)

    model = hub.load(MODEL_URL)
    
    if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(args.input)
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter("enhanced_vid.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w*4, h*4))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            out.write(enhance_frame(model, frame))
        cap.release()
        out.release()
    else:
        img = cv2.imread(args.input)
        if img is not None:
            cv2.imwrite("forensic_result.png", enhance_frame(model, img))
            print(f"[*] Enhancement complete. Output: forensic_result.png")

if __name__ == "__main__":
    main()
