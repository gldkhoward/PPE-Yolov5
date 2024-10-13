import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
def load_model(model_path):
    return torch.load(model_path, map_location=torch.device('cpu'))  # Load model from local path

# Function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    inter_area = max(0, x2_min - x1_max + 1) * max(0, y2_min - y1_max + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# Function to check if a person is wearing the required PPE (simplified version)
def check_ppe(detections, required_ppe):
    people = detections[detections['name'] == 'PERSON']
    ppe_items = detections[detections['name'].isin(required_ppe)]
    
    compliance = {}
    
    for index, person in people.iterrows():
        person_id = index
        person_bbox = [person['xmin'], person['ymin'], person['xmax'], person['ymax']]
        compliance[person_id] = {"compliant": True, "missing_ppe": []}
        
        for ppe in required_ppe:
            ppe_worn = False
            for _, item in ppe_items.iterrows():
                item_bbox = [item['xmin'], item['ymin'], item['xmax'], item['ymax']]
                
                # Check if the PPE's bounding box overlaps with the person’s bounding box
                if iou(person_bbox, item_bbox) > 0.3:  # Simple IoU threshold for overlap
                    ppe_worn = True
                    break
            
            if not ppe_worn:
                compliance[person_id]["compliant"] = False
                compliance[person_id]["missing_ppe"].append(ppe)
    
    return compliance

# Function to display the detection results
def display_results(image_path, detections, compliance):
    img = cv2.imread(image_path)
    
    people = detections[detections['name'] == 'PERSON']
    
    for index, person in people.iterrows():
        person_id = index
        person_bbox = [int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])]
        color = (0, 255, 0) if compliance[person_id]["compliant"] else (0, 0, 255)
        
        # Draw the person's bounding box
        cv2.rectangle(img, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), color, 2)
        
        # Add text indicating compliance status
        status_text = "Compliant" if compliance[person_id]["compliant"] else f"Missing: {', '.join(compliance[person_id]['missing_ppe'])}"
        cv2.putText(img, status_text, (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

