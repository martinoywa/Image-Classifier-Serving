import torch

from PIL import Image
from torchvision import transforms

import urllib


from flask import Flask, request


app = Flask(__name__)

# load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


def create_image():
    url, filename = (request.json.get('url'), 'image.jpg')
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        urllib.request.URLopener().retrieve(url, filename)


def preprocess(filename):
    # load  image
    input_image = Image.open(filename)
    # transform image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image)
    # add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

@app.route('/')
def home():
    return '''<h1>Welcome to the Image Classifier</h1>
    <p>Please use the following POST API endpoint to classify an image:</p>
    <p>/predict</p>
    <p>Request body example:</p>
    <p>
        { 
            "url": "https://s27870.pcdn.co/assets/flower-729512_1920-1024x678.jpg.optimal.jpg" 
        }
    </p>
    <p>Response body example:</p>
    <p>
        {
            "ant": 0.005777660757303238,
            "bee": 0.007221573032438755,
            "daisy": 0.9742687940597534,
            "fly": 0.001883076736703515,
            "ladybug": 0.001868181861937046
        }
    </p>
    <br>
    <p>Please use the following GET API endpoint to model metadata:</p>
    <p>/metadata</p>
    '''

@app.route('/metadata')
def metadata():
    return '''
    The MobileNet v2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck 
    layers opposite to traditional residual models which use expanded representations in the input. MobileNet v2 uses lightweight depthwise convolutions 
    to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers 
    were removed in order to maintain representational power.


    Model structure	Top-1 error	Top-5 error
    mobilenet_v2	    28.12	    9.71
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # download image
    create_image()

    # prediction
    with torch.no_grad():
        output = model(preprocess(filename="image.jpg"))

    probabilities = torch.nn.functional.softmax(output, dim=1)

    # load class names
    with open('imagenet_classes.txt', 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    
    # get top 5 predictions
    top_5_prob, top_5_class = torch.topk(probabilities, 5)
    final = dict()

    for i in range(top_5_class.size(1)):
        final[categories[top_5_class[0][i].item()]] = top_5_prob[0][i].item()

    return final


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)