from PIL import Image
from ultralytics import YOLO
import os
# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# Run inference on 'bus.jpg'
# path = '/Users/prabhnoorsingh/Downloads/Object-Detection-main/images/'
path = './images'

images = os.listdir(path)
for i in images :
    results = model(path+i)  # results list

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
