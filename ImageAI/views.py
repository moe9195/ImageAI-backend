from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
from rest_framework.generics import  CreateAPIView,RetrieveAPIView,UpdateAPIView
import numpy as np
from PIL import Image
import argparse, base64, io, cv2, requests
from ISR.models import RDN, RRDN
from .algorithms import super_resolution, colorize, deep_art, deblur, classify
from datetime import datetime, timedelta
from rest_framework_api_key.models import APIKey
from rest_framework_api_key.permissions import HasAPIKey
from .Serializers import UserCreateSerializer, UserSerializer, ProfileSerializer
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from django.contrib.auth.models import User
from .models import Profile

class RegisterView(CreateAPIView):
	serializer_class = UserCreateSerializer

class ProfileDetails(RetrieveAPIView):
    serializer_class = ProfileSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'user_id'
    lookup_url_kwarg = 'profile_id'

    def get_queryset(self):
        return Profile.objects.filter(user = self.request.user)

class ProfileUpdate(UpdateAPIView):
    serializer_class = ProfileSerializer
    def put(self, request, profile_id, format=None):
       profile = Profile.objects.get(user_id = profile_id)
       serializer = ProfileSerializer(profile, data=request.data)
       if serializer.is_valid():
    	   serializer.save()
    	   return Response(serializer.data)
       return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserDetails(RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'
    lookup_url_kwarg = 'user_id'

    def get_queryset(self):
        return User.objects.filter(id = self.request.user.id)

class GiveKey(views.APIView):

	permission_classes = [IsAuthenticated]

	def post(self, request, *args, **kwargs):
		name = request.data.get("name")
		api_key, key = APIKey.objects.create_key(name=name ,expiry_date =datetime.now()+timedelta(days=30) )
		profile = Profile.objects.get(user = self.request.user)
		profile.limit += 60
		profile.subscribed = True
		profile.key= key
		profile.save()
		return Response(key , status=status.HTTP_201_CREATED)


class Processing(views.APIView):

	permission_classes =[HasAPIKey]

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
	            # if style selected is invalid return the error string
	            if isinstance(im, str):
	                return Response(im, status=status.HTTP_405_METHOD_NOT_ALLOWED)
	        # if no style is selected, default to wave
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
