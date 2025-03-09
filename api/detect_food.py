from fastapi import FastAPI, File, UploadFile
import os
from ultralytics import YOLO
from PIL import Image

model_yolo = YOLO('best.pt')
app = FastAPI()
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Mở ảnh và chạy YOLO để phát hiện vật thể
        image = Image.open(file_path)
        results = model_yolo(image)

        # Lấy danh sách các vật thể phát hiện được
        detected_objects = []
        for r in results:
            for box in r.boxes:
                detected_objects.append({
                    "class": model_yolo.names[int(box.cls)],  # Tên lớp
                    "confidence": float(box.conf)  # Độ tin cậy
                })
        
        return {"file_path": file_path, "detections": detected_objects}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
