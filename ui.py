import streamlit as st
from tensorflow.keras.models import Sequential      
from tensorflow.keras.optimizers import Adam               
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  
from PIL import Image
import numpy as np
from keras.applications import DenseNet121

#Recall DenseNet121
conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(256,256,3),
    pooling='avg'
)
conv_base.trainable = False

# Reload the model
model_reload = Sequential()
model_reload.add(conv_base)
model_reload.add(BatchNormalization())
model_reload.add(Dense(256, activation='relu'))
model_reload.add(Dropout(0.35))
model_reload.add(BatchNormalization())
model_reload.add(Dense(120, activation='relu'))
model_reload.add(Dense(10, activation='softmax'))

model_reload.build(input_shape=(None, 256,256,3))
model_reload.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_reload.load_weights('model.weights.h5')

def img_to_np(image, target_size=(256, 256)):
    # Resize image and convert to numpy array
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if image has 4 channels (RGBA) and convert to RGB
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize the image to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array

labels=["Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites","Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",]

def predict(image):
    arr = img_to_np(image)
    arr = arr.reshape(1, 256, 256, 3)  # Reshape for the model
    prediction = model_reload.predict(arr)
    return labels[np.argmax(prediction)]

# Streamlit user interface
st.title('Tomato Leaf Disease Prediction')
uploaded_file = st.file_uploader("Choose an image...",type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    with col2:
        st.write("Classifying...")
        prediction = predict(image)
        st.write(f'Prediction: {prediction}')

# Run the app: streamlit run ui.py

