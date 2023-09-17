import torchvision
import torch
import torchvision.models as models
from torchvision import datasets
import yaml
from yaml.loader import SafeLoader
from detect import run
from flask import Flask, request , jsonify ,send_file
import os 
from PIL import Image, ImageOps  # Install pillow instead of PIL

app = Flask(__name__)
@app.route('/' , methods = ['POST']) 
@app.route('/detectAD' , methods = ['POST']) 
def detectAD():
    if 'image' not in request.files :
        return jsonify({'error' : 'no image found'})
    file = request.files['image']
    file.save(r"C:\Users\Fouad\yolov5\x.jpg")
    image = Image.open(file).convert("RGB")
    x = run(weights= "best.pt", source= r"C:\Users\Fouad\yolov5\x.jpg" , conf_thres= 0.6 , imgsz= (800, 800))
    return send_file(x, mimetype='image/jpg')


    # file = request.files['image']
    # file.save(r"C:\Users\Fouad\yolov5\x.jpg")
    # # Perform object detection on the input image using the YOLOv5 model
    # # and save the output image to the local disk
    # # (replace this with your actual detection code)
    # output_image_path = x = run(weights= "best.pt", source= r"C:\Users\Fouad\yolov5\x.jpg" , conf_thres= 0.6 , imgsz= (640 ,640))
    # os.system('cp x.jpg output.jpg')
    #  Return the path or link of the output image in the HTTP response
    # output_image_link = f"http://{request.host}/{output_image_path}"
    # return jsonify({'output_image_link': output_image_link})

if __name__ == '__main__' :
    app.debug = True
    app.run(host='0.0.0.0', port=8000)

    