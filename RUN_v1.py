import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data = r'C:\Users\a.kuipers\PycharmProjects\Q12324\CholecAHK.yaml' , epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model.predict(source = r'\\clin-storage\Group Ruers\Students personal folder\Alexander Kuipers (M2)_TUDelft_2023\Data\Cholec80AHK\images_original_test\video01.mp4')

# Export the model to ONNX format
success = model.export(format='onnx')