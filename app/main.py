import numpy as np
import cv2
from flask import Flask, request
import werkzeug
from app.PostProcessing import postprocess
from app.PreProcessing import show_image

app = Flask(__name__)


@app.errorhandler(werkzeug.exceptions.BadRequest)
@app.route("/api", methods=["POST"])
def scan():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # # show_image(img)
    return postprocess(img)



if __name__ == "__main__":
    app.run()