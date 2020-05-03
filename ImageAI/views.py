from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
import numpy as np
from PIL import Image
import argparse, base64, io, cv2, requests
from ISR.models import RDN, RRDN
from .algorithms import super_resolution, colorize, deep_art, deblur, classify

from .Serializers import ProcessingSerializer
from .models import ImageAI, Method


class Processing(views.APIView):
    def post(self, request, *args, **kwargs):
        img=request.data.get("img")
        method = request.data.get("method")

        # If the POSTED image is a string
        if isinstance(img, str):
            # check if it is base64
            if 'base64' in img[10:30]:
                img_b64 = img.split(',', 1)[1]
                img = np.array(Image.open(io.BytesIO(base64.b64decode(img_b64))))
            # check if it is a url
            else:
                response = requests.get(img)
                img = np.array(Image.open(io.BytesIO(response.content)))

        # If POSTED image is a file
        else:
            img = np.array(Image.open(io.BytesIO(img.file.read())))

        # if image only has 1 channel convert it to 3 channels RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # In case of png with alpha channel
        if len(img.shape) > 2 and img.shape[2] == 4:
            #convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # In case image is black and white and has alpha channel
        if len(img.shape) == 2 and img.shape[2] == 4:
            # convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # run each algorithm based on method and return processed img
        if method == "SuperResolution":
            print("HERE")
            print(img.shape)
            print(np.prod(img.shape))
            im = super_resolution(img)

        elif method == "Colorize":
            im = colorize(img)

        elif method == "DeepArt":
            style = request.data.get("style")
            if style:
                im = deep_art(img, style)
                if isinstance(im, str):
                    return Response(im, status=status.HTTP_405_METHOD_NOT_ALLOWED)
            else:
                im = deep_art(img, "wave")
        elif method == "Deblur":
            im = deblur(img)
        elif method == "Classify":
            # in the case of classification, we return an object instead of an image
            obj = classify(img)
            return Response(obj, status=status.HTTP_201_CREATED)
        else:
            pass

        # The image is then encoded to base64 and returned to the request user
        buffered =  io.BytesIO()
        im.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue())
        return Response(encoded_img, status=status.HTTP_201_CREATED)
