from flask import Flask, render_template, Response
import cv2
import uuid 

app = Flask(__name__)
video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces

def save_faces(faces):
    saved_faces = []
    enlarge_factor = 10  # Adjust this value to control enlargement
    
    for (x, y, w, h) in faces:
        
        
        # Enlarged coordinates
        enlarged_x = x - enlarge_factor
        enlarged_y = y - enlarge_factor
        enlarged_w = w + (2 * enlarge_factor)
        enlarged_h = h + (2 * enlarge_factor)
        
        frame = video.read()[1]
        face_frame = frame[enlarged_y:enlarged_y+enlarged_h, enlarged_x:enlarged_x+enlarged_w]
        
        filename = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(filename, face_frame)
        saved_faces.append(filename)
    return saved_faces


def generate_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

        faces = detect_faces(frame)
        saved_faces = save_faces(faces)

        for (x, y, w, h) in faces:
            resize_factor = 1.5  # Adjust this value to control resizing

            # Calculate the resized rectangle coordinates
            resized_x = int(x - (w * resize_factor / 2))
            resized_y = int(y - (h * resize_factor / 2))
            resized_w = int(w * resize_factor)
            resized_h = int(h * resize_factor)

            cv2.rectangle(frame, (resized_x, resized_y), (resized_x + resized_w, resized_y + resized_h), (100, 255, 100), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    frame = video.read()[1]
    faces = detect_faces(frame)
    num_faces = len(faces)
    return render_template('index.html', num_faces=num_faces, faces=faces)

@app.route('/face_image/<int:face_id>/<int:x>/<int:y>/<int:width>/<int:height>')
def face_image(face_id, x, y, width, height):
    frame = video.read()[1]
    roi = frame[y:y+height, x:x+width]
    ret, buffer = cv2.imencode('.jpg', roi)
    roi_bytes = buffer.tobytes()
    return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + roi_bytes + b'\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


