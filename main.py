
import streamlit as st
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
from keras.models import load_model
import keras



 
code = st.text_input('Enter your code: ')
if code=='1234':
 
 img_size=224
 labels = {0: 'Normal', 1: 'Pathological'}


 model = load_model('Bioiatriki_project_binary.h5')

 def get_activation_map(image_path,my_model,labels):
         img_size =224        
         image_loaded = PIL.Image.open(image_path)
         image_loaded = image_loaded.resize((img_size, img_size))
         image_loaded = np.asarray(image_loaded)

         if len(image_loaded.shape) < 3:
           image_loaded = np.stack([image_loaded.copy()] * 3, axis=2)

         preprocessed_image = preprocess_input(image_loaded)
         preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

         class_weights = my_model.layers[-1].get_weights()[0]
         final_conv_layer = my_model.layers[-3]

         get_output = keras.backend.function([my_model.layers[0].input], 
                                                [final_conv_layer.output, my_model.layers[-1].output])

         [conv_outputs, predictions] = get_output([preprocessed_image])
         conv_outputs = conv_outputs[0, :, :, :]
         cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

         class_predicted = np.argmax(predictions[0])
         class_predicted_name = labels[class_predicted]                                  
         for index, weight in enumerate(class_weights[:, class_predicted]):
           cam += weight * conv_outputs[:, :, index]
         predictions1 = f'Class predicted: {class_predicted_name}'
         cam /= np.max(cam)
         cam = cv2.resize(cam, (img_size, img_size))
         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
         heatmap[np.where(cam < 0.2)] = 0
         heatmap = np.uint8(heatmap * 0.3 + image_loaded)

         font = cv2.FONT_HERSHEY_SIMPLEX
         org = (50, 50) 
         fontScale = 0.3
         color = (255, 255,255)
         thickness = 1

         # Using cv2.putText() method
         cv2.putText(heatmap, ( '{} with  {:.3f} confidence'.format(class_predicted_name,predictions[0][class_predicted])) , org, font, 
                           fontScale, color, thickness, cv2.LINE_AA)


         return image_loaded,heatmap,predictions1,class_predicted_name,class_predicted,predictions




 print(".......PLEASE WAIT.....")

 st.title("Prediction of Chest X-RAYs")
 with st.beta_container():
 #   bio_image= cv2.imread('bioiatriki.png')
   bio_image= cv2.imread('ISCA_Logo_small2.png')
   bio_image = cv2.cvtColor(bio_image, cv2.COLOR_BGR2RGB)
   st.image(bio_image)

 uploaded_file = st.file_uploader("Choose an XRAY image (not DICOM) ",type=['png', 'jpg','jpeg'])

 if uploaded_file is not None:
   image_loaded,heatmap,predictions,class_predicted_name,class_predicted,predictions = get_activation_map(uploaded_file,model,labels)

   col1,col2 = st.beta_columns(2)
   with col1:
     st.image(image_loaded)

   with col2:
     st.image(heatmap)

     st.write( '{} with  {:.3f} confidence'.format(class_predicted_name,predictions[0][class_predicted])) 

else:
 print("Wrong code")




