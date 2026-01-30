import torch
import cv2
import sqlite3
import numpy as np
from torchvision import models, transforms
from PIL import Image
from grad_cam import GradCAM
import datetime

conn = sqlite3.connect('plant_disease.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS inference_log
             (id INTEGER PRIMARY KEY, timestamp TEXT, file TEXT, prediction TEXT, confidence REAL)''')
conn.commit()

classes = ['Early_Blight', 'Healthy', 'Late_Blight']
model = models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('plant_doctor_model.pth'))
model.eval()

cam = GradCAM(model, model.layer4[2].conv2)

def predict_and_log(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)
    label = classes[predicted.item()]
    
    heatmap = cam(input_tensor)
    
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_cv * 0.6
    cv2.imwrite(f"gradcam_{label}.jpg", superimposed_img)
    
    timestamp = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO inference_log (timestamp, file, prediction, confidence) VALUES (?, ?, ?, ?)",
              (timestamp, image_path, label, confidence.item()))
    conn.commit()
    
    print(f"Processed: {label} ({confidence.item()*100:.2f}%)")
    print(f"Log saved to DB. Visualization saved as gradcam_{label}.jpg")

# Run it
predict_and_log('test_leaf.jpg')