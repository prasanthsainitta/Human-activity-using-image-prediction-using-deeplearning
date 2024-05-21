from django.shortcuts import render
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

from . import forms
from .models import UserImageModel

# Create your views here.
def home(request):
    print("HI")
    if request.method == "POST":
        form = forms.UserImageForm(files=request.FILES)
        if form.is_valid():
            print('HIFORM')
            form.save()
        obj = form.instance
        #('obj',obj)

        result1 = UserImageModel.objects.latest('id')
        models = keras.models.load_model('C:/Users/SPIRO25/Desktop/problem/1.FINAL/Deploy/app/keras_model.h5')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open("C:/Users/SPIRO25/Desktop/problem/1.FINAL/Deploy/media/" + str(result1)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        classes = ['calling','clapping','cycling','dancing','drinking','eating','fighting','hugging','laughing','sitting']
        prediction = models.predict(data)
        idd = np.argmax(prediction)
        a = (classes[idd])
        return render(request, 'app/index.html',{'form':form,'obj':obj,'predict':a})
    else:
        form = forms.UserImageForm()
    return render(request, 'app/index.html',{'form':form})


# result = UserImageModel.objects.latest('id')
# result = result.image           
# models = keras.models.load_model('D:/BRAG/WORK/WORKING/DEEPLEARNING/HUMAN ACTIVITY - DL20/CODE/Deploy/app/LENET.h5')
# from tensorflow.keras.preprocessing import image
# test_image = image.load_img('D:/BRAG/WORK/WORKING/DEEPLEARNING/HUMAN ACTIVITY - DL20/CODE/Deploy/media/' + str(result),target_size=(225, 225))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = models.predict(test_image)
# prediction = result[0]
# prediction = list(prediction)
# classes = ['calling','clapping','cycling','dancing','drinking','eating','fighting','hugging','laughing','running','sitting']
# output = zip(classes, prediction)
# output = dict(output)
# if output['calling'] == 1.0:
#     a='calling'
# elif output['clapping'] == 1.0:
#     a='clapping'
# elif output['cycling'] == 1.0:
#     a="cycling"
# elif output['dancing'] == 1.0:
#     a="dancing"

# elif output['drinking'] == 1.0:
#     a='drinking'
# elif output['eating'] == 1.0:
#     a="eating"
# elif output['fighting'] == 1.0:
#     a="fighting"
# elif output['hugging'] == 1.0:
#     a='hugging'
# elif output['laughing'] == 1.0:
#     a="laughing"
# elif output['running'] == 1.0:
#     a="running"    
# elif output['sitting'] == 1.0:
#     a="sitting" 