import os
import cv2
import torch
import joblib
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive Agg backend for plotting
from PIL import Image
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import InceptionResnetV1, MTCNN
from flask import Flask, Response, request, render_template, redirect, url_for
# Add the missing import at the top of your script
from sklearn.calibration import CalibratedClassifierCV


# Initialize Flask app
app = Flask(__name__)

# Set up folder for image uploads (inside the static folder)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize FaceNet and MTCNN models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=100, thresholds=[0.7, 0.8, 0.9])
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the pre-trained SVM model and label encoder from the same directory
model_filename = 'svm_model.pkl'
label_encoder_filename = 'label_encoder.pkl'

svm_model = joblib.load(model_filename)  # Load the pre-trained SVM model
label_encoder = joblib.load(label_encoder_filename)  # Load the label encoder

# Route for home page
@app.route('/')
def home():
    return render_template('upload.html')

# Route for uploading and processing an image
@app.route('/upload', methods=['POST'])
def upload_image():
    labels = joblib.load('labels.pkl')  # Load saved labels
    print(labels)
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file in the 'static/uploads' directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Define the output image path with bounding boxes and labels
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + file.filename)
        
        # Process the image to detect faces and draw bounding boxes
        predictions, confidences = predict_and_draw_boxes(filepath, output_image_path)
        
        # Display the result on a new page
        return render_template(
            'result.html', 
            result="Predictions made", 
            image_file='output_' + file.filename,
            predictions_with_confidences=zip(predictions, confidences)
        )

# Function to process image, detect faces, and draw bounding boxes with labels
def predict_and_draw_boxes(image_path, output_path):
    # Load the image
    image = Image.open(image_path)
    
    # Detect faces in the image
    boxes, probs = mtcnn.detect(image)  # Get face detection probabilities
    
    predictions = []  # Store the predicted class labels
    confidences = []  # Store the corresponding confidence scores

    if boxes is not None and len(boxes) > 0:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        # Iterate over all detected faces
        for i, box in enumerate(boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            # Filter out low-confidence and small bounding boxes
            if probs[i] >= 0.7 and width > 50 and height > 50:  # Filter on confidence and box size
                face = image.crop(box)
                face = face.resize((160, 160))
                face_tensor = torch.unsqueeze(torch.Tensor(np.array(face).transpose(2, 0, 1) / 255), 0).to(device)

                # Get face embedding from FaceNet model
                with torch.no_grad():
                    embedding = facenet_model(face_tensor).cpu().numpy()

                # Predict using the pre-trained SVM classifier
                pred_label = svm_model.predict(embedding)
                pred_proba = svm_model.predict_proba(embedding).max()  # Get the highest confidence score

                # Check confidence and mark as "Unknown" if below threshold
                if pred_proba < 0.60:
                    pred_class = f"Unknown but looks like {label_encoder.inverse_transform(pred_label)[0]}"
                else:
                    pred_class = label_encoder.inverse_transform(pred_label)[0]

                # Append prediction and confidence to lists
                predictions.append(pred_class)
                confidences.append(f"{pred_proba * 100:.2f}")  # Convert to percentage format
                
                # Draw the bounding box
                rect = patches.Rectangle(
                    (box[0], box[1]), width, height, 
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)

                # Annotate with predicted class and confidence (using percentage format)
                ax.text(box[0], box[1] - 10, f"{pred_class}: {pred_proba * 100:.2f}%", color='red', fontsize=12)

        plt.axis('off')  # Hide axes
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return predictions, confidences

# Route to serve real-time video
def gen_frames():
    cap = cv2.VideoCapture(0)  # Use webcam
    while True:
        success, frame = cap.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Convert the frame to RGB (OpenCV uses BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            boxes, probs = mtcnn.detect(rgb_frame)

            if boxes is not None:
                # Iterate over detected faces and make predictions
                for i, box in enumerate(boxes):
                    if probs[i] >= 0.7:  # Only process confident detections
                        # Ensure that the face region is valid before cropping
                        x1, y1, x2, y2 = map(int, box)
                        if x2 - x1 > 0 and y2 - y1 > 0:  # Ensure valid box dimensions
                            # Crop the face region
                            face = rgb_frame[y1:y2, x1:x2]
                            
                            # Ensure the face is non-empty before resizing
                            if face.size > 0:
                                face = cv2.resize(face, (160, 160))  # Resize for FaceNet
                                face_tensor = torch.unsqueeze(torch.Tensor(np.array(face).transpose(2, 0, 1) / 255), 0).to(device)

                                # Get face embedding from FaceNet model
                                with torch.no_grad():
                                    embedding = facenet_model(face_tensor).cpu().numpy()

                                # Predict using the pre-trained SVM classifier
                                pred_label = svm_model.predict(embedding)
                                pred_proba = svm_model.predict_proba(embedding).max()
                                pred_class = label_encoder.inverse_transform(pred_label)[0]

                                # If confidence is below a certain threshold, mark it as 'Unknown'
                                if pred_proba < 0.60:
                                    pred_class = "Unknown"

                                # Annotate the image with the predicted class and confidence
                                text = f"{pred_class}: {pred_proba * 100:.2f}%"
                                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in a format that Flask can stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method == 'POST':
        # Get the uploaded files and the new person's name
        files = request.files.getlist("file[]")
        person_name = request.form['name']

        # Ensure exactly 8 files are uploaded
        if len(files) != 8:
            return "Please upload exactly 8 photos."

        # Directory where new face images will be stored
        new_face_dir = os.path.join('static', 'faces', person_name)
        os.makedirs(new_face_dir, exist_ok=True)

        # Save the uploaded images
        for file in files:
            file_path = os.path.join(new_face_dir, file.filename)
            file.save(file_path)
        
        # Generate embeddings for the new face and update the model
        add_new_face_to_model(new_face_dir, person_name)

        return f"Added new face for {person_name} and updated the model."
    
    return render_template('add_face.html')

def add_new_face_to_model(face_dir, person_name):
    # Load existing embeddings and labels from the SVM model and label encoder
    embeddings = joblib.load('embeddings.pkl')  # Load saved embeddings
    labels = joblib.load('labels.pkl')  # Load saved labels
    
    new_face_embeddings = []
    new_labels = []
    
    for image_file in os.listdir(face_dir):
        image_path = os.path.join(face_dir, image_file)
        image = Image.open(image_path)

        # Detect face and get embeddings
        boxes, _ = mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            face = image.crop(boxes[0])
            face = face.resize((160, 160))
            face_tensor = torch.unsqueeze(torch.Tensor(np.array(face).transpose(2, 0, 1) / 255), 0).to(device)

            # Generate face embeddings using FaceNet
            with torch.no_grad():
                embedding = facenet_model(face_tensor).cpu().numpy()

            # Append new embeddings and labels
            new_face_embeddings.append(embedding)
            new_labels.append(person_name)
        else:
            print(f"No face detected in {image_file}")
    
    # If embeddings are generated, proceed to update the model
    if len(new_face_embeddings) > 0:
        # Combine new embeddings with existing ones
        embeddings = np.vstack([embeddings, np.array(new_face_embeddings).squeeze()])
        labels.extend(new_labels)

        # Encode the updated labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Retrain the SVM model
        classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')
        classifier.fit(embeddings, encoded_labels)

        # Calibrate the classifier
        calibrated_classifier = CalibratedClassifierCV(classifier, cv='prefit')
        calibrated_classifier.fit(embeddings, encoded_labels)

        # Save the updated model, label encoder, and embeddings
        joblib.dump(calibrated_classifier, 'svm_model.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        joblib.dump(embeddings, 'embeddings.pkl')
        joblib.dump(labels, 'labels.pkl')

        print(f"Added new face for {person_name} and updated the model.")
    else:
        print(f"Error: No embeddings generated for {person_name}.")

def remove_face_from_model(person_name):
    # Load existing embeddings and labels from the SVM model and label encoder
    embeddings = joblib.load('embeddings.pkl')  # Load saved embeddings
    labels = joblib.load('labels.pkl')  # Load saved labels

    # Check if the person exists in the labels
    if person_name not in labels:
        print(f"Error: {person_name} not found in the model.")
        return

    # Find the indices of the embeddings and labels corresponding to the person to be removed
    indices_to_remove = [i for i, label in enumerate(labels) if label == person_name]

    # Remove the corresponding embeddings and labels
    embeddings = np.delete(embeddings, indices_to_remove, axis=0)
    labels = [label for i, label in enumerate(labels) if i not in indices_to_remove]

    # Ensure there's at least one person left in the dataset
    if len(labels) == 0:
        print(f"Error: No embeddings left in the model after removal of {person_name}.")
        return

    # Encode the updated labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Retrain the SVM model with remaining data
    classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')
    classifier.fit(embeddings, encoded_labels)

    # Calibrate the classifier
    calibrated_classifier = CalibratedClassifierCV(classifier, cv='prefit')
    calibrated_classifier.fit(embeddings, encoded_labels)

    # Save the updated model, label encoder, and embeddings
    joblib.dump(calibrated_classifier, 'svm_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(embeddings, 'embeddings.pkl')
    joblib.dump(labels, 'labels.pkl')

    print(f"Removed {person_name} from the model and updated the classifier.")

if __name__ == '__main__':
    # Run the Flask app, accessible on the local network
    app.run(host='0.0.0.0', port=5099, debug=True)
