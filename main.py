import os
from flask import Flask, request, jsonify

print("Loading...")

import os


# CV and Image

from keras import backend as K
K.image_data_format()
import os
from datetime import datetime
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.models import load_model
import urllib.request
import wolframalpha
K.image_data_format()
print("Done")

from flask import Flask, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads/files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set maximum file size to 16MB (adjust as needed)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # image = open(image_path, 'rb').read()
        # image_data = BytesIO(image)
        equation = process_image(image_path)
        print(equation)
        return equation

    return "No file provided."

@app.route('/calculate', methods=['GET'])
def calculate_equation():
    equation_str = request.args.get('equation')  # Get the equation from the request parameters
    if 'x' not in equation_str and 'y' not in equation_str and '=' not in equation_str:
        print(eval(equation_str))
        return str(eval(equation_str))
    app_id = '57UAP2-8Y42L28TQQ'
    client = wolframalpha.Client(app_id)

    # Replace '=' with '==' for the Wolfram Alpha API
    query = equation_str.replace('=', '==')

    # Make a query to the API
    res = client.query(query)

    # Extract the solution from the 'Solution' pod in the API response
    for pod in res.pods:
        if pod.title == 'Solution':
            return pod.text.split('=')[-1].strip()

    # If the API response doesn't contain a 'Solution' pod, return an error message
    return "Unable to solve equation"

def process_image(image_path):
    url = "https://github.com/Dris7/overview/raw/main/eq_solver.h5"
    urllib.request.urlretrieve(url, "eq_solver.h5")

    # Load the model
    model = load_model("eq_solver.h5")
    #url_img = 'https://raw.githubusercontent.com/Dris7/overview/main/testttt.jpeg'


    # Read the image file as a binary stream
    with open(image_path, 'rb') as file:
        img_data = file.read()

    # Convert the binary stream to a NumPy array
    img_array = np.frombuffer(img_data, dtype=np.uint8)

    # Decode the image using cv2.imdecode()
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Display the image using matplotlib
    plt.imshow(img, cmap='gray')
    plt.show()

    if img is not None:
        img = ~img
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w = int(28)
        h = int(28)
        train_data = []
        print(len(cnt))
        rects = []
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, w, h]
            rects.append(rect)
        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (
                            rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)
        dump_rect = []
        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if (area1 == min(area1, area2)):
                        dump_rect.append(rects[i])
        print(len(dump_rect))
        final_rect = [i for i in rects if i not in dump_rect]
        print(final_rect)
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            im_resize = np.reshape(im_resize, (28, 28, 1))
            train_data.append(im_resize)
            plt.imshow(im_resize, cmap='gray')
            plt.show()

    equation = ''

    for i in range(len(train_data)):

        train_data[i] = np.array(train_data[i])
        train_data[i] = train_data[i].reshape(1, 28, 28, 1)
        result = np.argmax(model.predict(train_data[i]), axis=-1)

        for j in range(10):
            if result[0] == j:
                equation = equation + str(j)

        if result[0] == 10:
            equation = equation + "+"
        if result[0] == 11:
            equation = equation + "-"
        if result[0] == 12:
            equation = equation + "*"
        if result[0] == 13:
            equation = equation + "/"
        if result[0] == 14:
            equation = equation + "="
        if result[0] == 15:
            equation = equation + "."
        if result[0] == 16:
            equation = equation + "x"
        if result[0] == 17:
            equation = equation + "y"
        if result[0] == 18:
            equation = equation + "z"
    print("Your Equation :", equation)

    return equation




if __name__ == '__main__':
    app.run(port=7777)
