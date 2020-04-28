from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
import numpy as np
from PIL import Image
import argparse, base64, io, cv2
from ISR.models import RDN, RRDN


from .Serializers import ProcessingSerializer
from .models import ImageAI
from .models import Method




class Processing(views.APIView):
    def post(self, request, *args, **kwargs):

        img=request.data.get("img")

        # test = base64.b64encode(img.getvalue())
        # return Response(test, status=status.HTTP_201_CREATED)
        img = (np.array(Image.open(io.BytesIO(img.file.read()))))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        method = request.data.get("method")

        
        if method == "SuperResolution":

            # First we initialize the RDN (Recurral neural network model)
            model = RDN(weights='noise-cancel')

            # Model is then used to 'predict' the higher resolution image
            processed_img_arr = model.predict(np.array(img))
            im = Image.fromarray(processed_img_arr.astype('uint8'), 'RGB')

            # The image is then encoded to base64 and returned to the request user
            buffered =  io.BytesIO()
            im.save(buffered, format="JPEG")
            encoded_img = base64.b64encode(buffered.getvalue())

            return Response(encoded_img, status=status.HTTP_201_CREATED)

        elif method == "Colorize":

            # Loading model and network weights from model file
            model = cv2.dnn.readNetFromCaffe('./bw-colorization/model/colorization_deploy_v2.prototxt', './bw-colorization/model/colorization_release_v2.caffemodel')
            pts = np.load('./bw-colorization/model/pts_in_hull.npy')
            
            # Adding the 1x1 convolutions to the model
            class8 = model.getLayerId("class8_ab")
            conv8 = model.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2, 313, 1, 1)
            model.getLayer(class8).blobs = [pts.astype("float32")]
            model.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

            # Load the image, normalize the pixels to [0, 1] range, convert colorspace to Lab color
            scaled = img.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)   

            # Resize image to 224x224 (this is the dimensions of the neural network input)
            # Extract the L from colorspace and center the mean
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            # The network expects black and white images, this is why we extract the L channel
            # We pass the L channel to the network, which predicts the a and b channel values (colors)
            # a = green/red, b = blue/yellow, L = black/white
            model.setInput(cv2.dnn.blobFromImage(L))
            ab = model.forward()[0, :, :, :].transpose((1, 2, 0))

            # Resize the output image into our input image
            ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

            # Grab L channel from original image and concatenate to the ab channels
            # Now we have all three channels
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

            # Convert the output image from the Lab color space to RGB, then
            # Clip any values that fall outside the range [0, 1]
            # Rescale back to [0, 255] then convert it to 8bit image
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8") 
            im = Image.fromarray(colorized.astype('uint8'), 'RGB')

            # The image is then encoded to base64 and returned to the request user
            buffered =  io.BytesIO()
            im.save(buffered, format="JPEG")
            encoded_img = base64.b64encode(buffered.getvalue())

            return Response(encoded_img, status=status.HTTP_201_CREATED)

        else:
            pass
