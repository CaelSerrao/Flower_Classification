from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and define class labels
model = load_model('flower_classifier.h5')
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Update as needed

# Image preprocessing function
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'flower_image' not in request.files:
            return "No file part"
        file = request.files['flower_image']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Make prediction
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            predicted_label = class_labels[class_index]

            return render_template('upload.html', filename=filename, label=predicted_label)
    return render_template('upload.html', filename=None, label=None)

if __name__ == '__main__':
    app.run(debug=True)
