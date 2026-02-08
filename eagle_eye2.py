import os
import sys
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def check_hardware():
    print("="*40)
    print("      🦅 EAGLE EYE FORENSICS PRO 🦅")
    print("="*40)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Optimize memory usage for your GTX 1650
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[*] SUCCESS: GPU Detected! \n[*] Hardware: {gpus[0]}")
            print("[*] Mode: HIGH SPEED (CUDA Enabled)")
        except RuntimeError as e:
            print(e)
    else:
        print("[!] WARNING: No GPU detected. Mode: SLOW (CPU Only)")
    print("="*40)

def load_model():
    print(f"[*] Initializing AI Brain (ESRGAN)...")
    return hub.load(MODEL_URL)

def preprocess_image(image):
    """Prepares image for the model (removes alpha, crops to x4)"""
    hr_image = tf.convert_to_tensor(image)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def enhance_frame(model, frame):
    """Integrated 2-Step Enhancement Pipeline"""
    # --- STEP 1: AI Upscaling (TensorFlow) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pre_data = preprocess_image(rgb_frame)
    fake_image = model(pre_data)
    
    # Convert tensor back to numpy (OpenCV format)
    enhanced_img = tf.clip_by_value(fake_image, 0, 255)
    enhanced_img = tf.cast(enhanced_img, tf.uint8)
    enhanced_img = tf.squeeze(enhanced_img).numpy()
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    # --- STEP 2: Forensic Post-Processing (OpenCV) ---
    # A. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # B. Laplacian Edge Sharpening
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]])
    final = cv2.filter2D(final, -1, kernel)

    return final

def main():
    parser = argparse.ArgumentParser(description="EagleEye: Forensic Enhancer Pro")
    parser.add_argument("-i", "--input", required=True, help="Path to input image/video")
    args = parser.parse_args()

    check_hardware()

    if not os.path.exists(args.input):
        print(f"[!] Error: File {args.input} not found.")
        return

    model = load_model()
    
    file_ext = os.path.splitext(args.input)[1].lower()
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    
    if file_ext in video_exts:
        print(f"[*] Target: VIDEO | Processing: {args.input}")
        cap = cv2.VideoCapture(args.input)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        save_path = "forensic_enhanced_" + os.path.basename(args.input)
        # Note: ESRGAN increases size by 4x
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*4, height*4))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            enhanced_frame_data = enhance_frame(model, frame)
            out.write(enhanced_frame_data)
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_proc = frame_count / elapsed
                sys.stdout.write(f"\r[*] Frame: {count}/{total_frames} | {fps_proc:.2f} FPS")
                sys.stdout.flush()
            
        cap.release()
        out.release()
        print(f"\n[*] Evidence Exported: {save_path}")

    else:
        print(f"[*] Target: IMAGE | Processing: {args.input}")
        img = cv2.imread(args.input)
        if img is None:
            print("[!] Error reading image.")
            return

        final_output = enhance_frame(model, img)
        
        save_path = "forensic_enhanced_" + os.path.basename(args.input) + ".png"
        cv2.imwrite(save_path, final_output)
        print(f"[*] Evidence Exported: {save_path}")

if __name__ == "__main__":
    main()
