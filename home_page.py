import joblib
import numpy as np
from PIL import Image
import face_recognition
import streamlit as st

st.set_page_config("BMI Calculator App","👨‍💻")

st.title("BMI Calculator App")


height_model = joblib.load('height_model.joblib')
weight_model = joblib.load('weight_model.joblib')
bmi_model = joblib.load('bmi_model.joblib')

def get_face_encoding(image_path):
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        st.error("No face found")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


def predict_height_width_BMI(test_image,height_model,weight_model,bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)),axis=0)
    height = np.exp(height_model.predict(test_array))
    weight = np.exp(weight_model.predict(test_array))
    bmi = np.exp(bmi_model.predict(test_array))
    return {'height': str(height) + "  cm",
            'weight':str(weight) + "  kg",
            'bmi':str(bmi) + "  kg/cm2"}

input_image = st.file_uploader("Upload image")
submit = st.button("Submit")

if submit:
    input_image = Image.open(input_image)
    input_image.save("demo.png")
    st.image(input_image,width=500)
    test_image = r"demo.png"
    output = predict_height_width_BMI(test_image,height_model,weight_model,bmi_model)

    for i,j in output.items():
        c1,c2 = st.columns(2)
        with c1: st.success(i)
        with c2: st.info(j.replace("]","").replace("[",""))
        
