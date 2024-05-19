from typing import Annotated

from fastapi import FastAPI, File, UploadFile
import numpy
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keras.models import load_model
import numpy as np
from PIL import Image
from torchvision import transforms
from PIL import Image
import cv2
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount a directory containing images
app.mount("/cropped_image", StaticFiles(directory="cropped_image"), name="cropped_image")

@app.post("/uploadfile/")
async def create_upload_file(uniqueID:str,file: UploadFile):
    # try:

        # Define the directory paths
        images_dir = f"./images/{uniqueID}"
        path=f"cropped_image/{uniqueID}"
        cropped_image_dir = f"./{path}"

        # Ensure the directories exist, if not create them
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(cropped_image_dir, exist_ok=True)

        # Now you can proceed with your FastAPI application

        image_path = f"{images_dir}/{file.filename}"
        with open(image_path,'wb') as f:
            contents=file.file.read()
            f.write(contents)


        # Apply the transformation to the image
        # image_tensor = transform(image)
        weight='./model/weight/best.pt'
        # Create yolo model 
        model=YOLO(weight)
        result=model.predict(source=image_path,conf=0.25)

        img = cv2.imread(f'{images_dir}/{file.filename}')

        # Extract bounding boxes
        boxes = result[0].boxes.xyxy.tolist()

        # Iterate through the bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            # Save the cropped object as an image
            cv2.imwrite(f'{cropped_image_dir}/{file.filename}', ultralytics_crop_object)


        # Load the model
        keras_model = load_model('./model/image_classifier_model1.h5')



    

        # Load the image using PIL
        pil_image = Image.open(f'{cropped_image_dir}/{file.filename}')

        # Ensure the image has 4 channels
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGBA')

            # Split the channels
            r, g, b, a = pil_image.split()

            # Concatenate RGB channels
            rgb_image = Image.merge('RGB', (r, g, b))

        else:
            rgb_image=pil_image

        # Apply transformations

        transform=transforms.Compose(
            [
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ]
        )
        trans_image=transform(rgb_image).permute(1,2,0)
        trans_image=trans_image.unsqueeze(dim=0)


        # Make predictions
        predictions = keras_model.predict(trans_image)

        # Get the predicted class
        predicted_class = np.argmax(predictions)


        # Print the predicted class


        return {"predicted_class":str(predicted_class),"cropped_image":FileResponse(f'{path}/{file.filename}', media_type="image/jpeg")}
    # except Exception:
    #     return {"error":"Face not detected"}