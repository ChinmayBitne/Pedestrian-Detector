import os
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Load the YOLOv8 model for pedestrian detection
chosen_model = YOLO("Model/Person.pt")

@st.cache_data()
def predict(_chosen_model, img, classes=[], conf=0.5):
    # Resize the image to 640x480
    img = cv2.resize(img, (640, 480))
    if classes:
        results = _chosen_model.predict(img, classes=classes, conf=conf, save_txt=False)
    else:
        results = _chosen_model.predict(img, conf=conf, save_txt=False)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    img = cv2.resize(img, (640, 480))
    results = predict(chosen_model, img, classes, conf=conf)

    pedestrians = 0
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] == "person":
                pedestrians += 1
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                cv2.putText(img,  f"Pedestrian", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img, results, pedestrians

def process_frame(frame):
    result_frame, _, pedestrians = predict_and_detect(chosen_model, frame)
    return result_frame, pedestrians

def main():
    st.title("Pedestrian Detection")

    option = st.selectbox("Select Option", ["Upload Video", "Upload Image", "Live Camera"])

    if option == "Live Camera":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        pedestrians = 0
        stop_button = st.button("Stop Live Camera")
        pedestrian_text = st.empty()
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame, pedestrians = process_frame(frame_rgb)
            stframe.image(result_frame)
            pedestrian_text.text(f"Pedestrians Detected: {pedestrians}")
        cap.release()

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            result_frame, pedestrians = process_frame(img_array)
            st.image(result_frame)
            st.write(f"Pedestrians Detected: {pedestrians}")

    elif option == "Upload Video":
      uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
      if uploaded_file is not None:
          with tempfile.NamedTemporaryFile(delete=False) as temp_file:
              temp_file.write(uploaded_file.read())
              temp_file_path = temp_file.name

          cap = cv2.VideoCapture(temp_file_path)
          stframe = st.empty()
          pedestrians = 0
          pedestrian_text = st.empty()
          while cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                  break
              frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              result_frame, pedestrians = process_frame(frame_rgb)
              stframe.image(result_frame)
              pedestrian_text.text(f"Pedestrians Detected: {pedestrians}")
          cap.release()

          os.unlink(temp_file_path)

if __name__ == '__main__':
    main()
