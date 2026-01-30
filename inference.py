import torch
import cv2
import numpy as np
import sqlite3
import datetime
import os
from torchvision import models, transforms, datasets
from PIL import Image
from grad_cam import GradCAM

# --- CONFIG ---
MODEL_PATH = 'plant_doctor_model_v2.pth'
DATA_DIR = 'dataset'  # Used to read class names automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    temp_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = temp_dataset.classes
    print(f"Loaded {len(class_names)} classes.")

    model = models.resnet34(weights=None)  # No internet weights needed, we have our own
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    target_layer = model.layer4[2].conv2
    cam = GradCAM(model, target_layer)

    conn = sqlite3.connect('plant_disease.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL
        )
    ''')
    conn.commit()

    while True:
        img_path = input("\nEnter image path (or 'q' to quit): ").strip().strip('"')
        if img_path.lower() == 'q':
            break
        
        if not os.path.exists(img_path):
            print("Error: File not found.")
            continue

        try:
            original_image = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

            heatmap, output = cam(input_tensor)
            
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            predicted_label = class_names[pred_idx.item()]
            conf_score = confidence.item()

            print(f"Result: {predicted_label} ({conf_score*100:.2f}%)")

            probs = torch.nn.functional.softmax(output, dim=1)
            
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            print(f"\n--- Analysis for {os.path.basename(img_path)} ---")
            for i in range(3):
                class_name = class_names[top3_idx[0][i].item()]
                probability = top3_prob[0][i].item()
                print(f"{i+1}. {class_name}: {probability*100:.2f}%")
            print("-----------------------------\n")

            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            img_cv = cv2.resize(img_cv, (224, 224))
            
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
            
            save_name = f"result_{os.path.basename(img_path)}"
            cv2.imwrite(save_name, superimposed)
            print(f"Visual explanation saved to: {save_name}")

            timestamp = datetime.datetime.now().isoformat()
            cursor.execute("INSERT INTO inference_logs (timestamp, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
                           (timestamp, os.path.basename(img_path), predicted_label, conf_score))
            conn.commit()
            print("Logged to Database.")

        except Exception as e:
            print(f"Error processing image: {e}")

    conn.close()

if __name__ == '__main__':
    main()