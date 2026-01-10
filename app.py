
import streamlit as st
from ultralytics import YOLO
import cv2, tempfile

st.title("YOLOv8 Face Detection")

model = YOLO("runs/detect/train/weights/best.onnx")

mode = st.radio("select mode", ['Image','Video','Webcam'])
if mode == 'Image':
    file = st.file_uploader("Upload image", type=["jpg","png"])
    if file:
        suffix = file.name.split(".")[-1]
        print(suffix)
        t = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
        t.write(file.read())
        img = cv2.imread(t.name)
        res = model(t.name)
        for r in res:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),6)
        st.image(img, channels="BGR")
        

elif mode == 'Video':
    file = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])
    if file:
        suffix1 = file.name.split(".")[-1]
        t_file = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix1)
        t_file.write(file.read())
        video = cv2.VideoCapture(t_file.name)
        start = st.button("Start video")
        stop = st.button("Stop video")

        if "run_video" not in st.session_state:
            st.session_state.run_video = False

        if start:
            st.session_state.run_video = True
        if stop:
            st.session_state.run_video = False

        frame_placeholder = st.empty()
        while st.session_state.run_video and video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = model(frame, conf=0.5)

            for r in results:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            frame_placeholder.image(frame, channels="BGR")

        video.release()
else:
    st.warning("Press Start to open webcam")

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    frame_placeholder = st.empty()

    if st.session_state.run:
        cap = cv2.VideoCapture(0)

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.5)

            for r in results:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()