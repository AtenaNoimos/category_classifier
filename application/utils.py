from fastapi import HTTPException
import numpy as np
import cv2

def validate_input_file(file):
    fileExtension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
   
def transform_image_cv2(stream):       
    # Start the stream from the beginning (position zero)
    stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def transform_cv2_image(image):       
    return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
