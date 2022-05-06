import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from application.components import classifier_inference
from application.utils import validate_input_file, transform_image_cv2, transform_cv2_image
import io
import numpy as np
import cv2

app = FastAPI()
classifier = classifier_inference()


@app.get("/")
async def root():
    return {"message": "Hello World"}
    
    
@app.post("/api/predict")
async def category_classifier(file: UploadFile = File(...)):
    fileExtension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
        
    image_stream = io.BytesIO(file.file.read())
    image = transform_image_cv2(image_stream)
    image = classifier.preprocess (image)
    predictions = classifier.predict(image)
    print (predictions)
    return predictions
    
    
if __name__=="__main__":
    uvicorn.run(app)