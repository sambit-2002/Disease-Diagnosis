from flask import Flask, flash, request, redirect, url_for, render_template
#pip install flask
import urllib.request
import os
from werkzeug.utils import secure_filename

from PIL import Image 
#pip install PIL
import tensorflow as tf
#pip install tensorflow
import pandas as pd
#pip install pandas

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])\
    
import tensorflow as tf

model1 = tf.keras.models.load_model("mymodel.h5")
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate(path):
    # Step 1: Read the image
    image = Image.open(path)

    # Step 2: Create DataFrame with 5 columns (assuming binary classes)
    test_df = pd.DataFrame({
        'path': [path],
        'Atelectasis': [0],   # replace these values based on your use case
        'Cardiomegaly': [1],
        'Consolidation': [0],
        'Edema': [1],
        'Pleural Effusion': [0]
    })

    # Step 3: Define class names and target size
    class_names =  ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    IMG_SIZE = (224, 224)  # Adjust to your image size

    # Step 4: Set up ImageDataGenerator and create batches
    base_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_X, test_Y = next(base_gen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,  # set to None if paths are absolute
        x_col='path',
        y_col=class_names,
        class_mode='raw',
        target_size=IMG_SIZE,
        shuffle=True,
        batch_size=1
    ))
    # return test_X
    predictions = model1.predict(test_X)
    binary_predictions = (predictions >= 0.5).astype(int)
    return binary_predictions   
 
@app.route('/')
def home():
    return render_template('index.html' , len = 0 , labels = [])
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('Image successfully uploaded and displayed below')
    
        path = f"static/uploads/{filename}"
        x = validate(path=path)
        print(x.shape)
        class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        x = x[0]
        l = []
        for i in range(len(x)):
            if(x[i] == 1):
                l.append(class_names[i])
        print(l)
            
        return render_template('index.html', filename=filename , len = len(l) , labels = l)
    
    
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()