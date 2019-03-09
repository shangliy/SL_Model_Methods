import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import io
import cStringIO as StringIO
import urllib
import json
import requests
import exifutil
import uuid


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = './tmp/img_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

api_address = "http://www.gofindapi.com:3000/searchapi"
graph_path = "gofind_58_v2.pb"
output_box = 'Softmax:0'
input_tensor = 'input_1:0'
output_fea = 'Relu_94:0'
INPUT_SIZE = 299


# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    global file_id
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        string_buf = StringIO.StringIO()
        image_pil = Image.fromarray((255 * image).astype('uint8'))
        image_pil = image_pil.resize((299, 299))
        image_pil.save(string_buf, format='png')
        img_str = string_buf.getvalue().encode('base64').replace('\n', '')
        req_body = {"img64":img_str}
        resp = requests.post(api_address,
                                json=req_body,
                                headers={'Content-Type':'application/json'})
        resp_content = json.loads(resp.content)
        output = (resp_content)
        result = [True]
        file_id = str(uuid.uuid4())
        with open(file_id, 'w') as outfile:
            json.dump(output["data"], outfile,indent=4)
        for image_info in output["data"]:
            img_link =  image_info["reference_image_links"]
            

            fd = urllib.urlopen(str(img_link[0]))
            image_file = io.BytesIO(fd.read())
            im = Image.open(image_file)
            image_pil = im.resize((256, 256))
            string_buf = StringIO.StringIO()
            image_pil.save(string_buf, format='png')
            data = string_buf.getvalue().encode('base64').replace('\n', '')
            result.append('data:image/png;base64,' + data)
        
    except Exception as err:
        print (err)
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    
    #result = app.clf.classify_image(image)
    
    print (result[0])
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

@app.route('/poll')
def poll():
    global file_id
    print "file_id",file_id
    vote = flask.request.args.get('field')[0]
    
    out =  open('vote.txt', 'a')
    out.write( file_id+":"+vote + '\n' )
    out.close() 
    return flask.render_template(
        'index.html', has_result=False
    )

if __name__ == '__main__':
    global file_id
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)

'''
<form action="/poll">
  {% for e in data.fields %}
     <input type="radio" name="field" value="{{ e }}"> {{ e }}<br>
  {% endfor %}
  <input type="submit" value="Vote" />
'''