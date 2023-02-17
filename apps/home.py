import streamlit as st
import pandas as pd
import numpy as np
from data.create_data import create_table
from PIL import  Image

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# st.set_page_config(
#         page_title="Covid19 CT Scan Images Detector",
#         page_icon="clean_hands_open_hearts_covid19footerimage2-removebg-preview.png",
#         layout="centered",
#         initial_sidebar_state="auto",
#
#     )

st.set_option('deprecation.showfileUploaderEncoding', False)

def app():
    display = Image.open('clean_hands_open_hearts_covid19footerimage2.jpg')
    display = np.array(display)
    st.image(display, width=400)
    # st.title("Covid19 Chest Images Scans Detector")

    new_title = '<p style="text-align: center; font-weight: bold; font-family:sans-serif; color:Black; font-size: 62px;">Covid19 Chest Images Scans Detector</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.title('Xray')

    adjust_footer = """
        <style>
        footer:after {
        content: 'Copyright @ 2023 By Ramy Elsaraf';
        display: block;
        position: relative;
        }
        </style>
        """

    st.markdown(adjust_footer, unsafe_allow_html=True)

    # @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def import_and_predict(image_data, model):
        image = ImageOps.fit(image_data, (224, 224), Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        st.image(image, channels='RGB')
        image = (image.astype(np.float32) / 255.0)
        img_reshape = image[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    model = tf.keras.models.load_model('20211113-21011636837298-Covid19-XRayDetection-Model-Good-2 (1).h5')

    st.write("""
             # ***Covid19 Detector***
             """
             )

    st.write("This is a simple image classification web app to predict covid19 of chest images Xrays")

    # file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    # #
    # if file is None:
    #     st.text("You haven't uploaded an image file")
    # else:
    #     imageI = Image.open(file)
    #     st.image(imageI, use_column_width=True)
    #     # prediction = import_and_predict(image, model)
    #     #
    #     # if np.argmax(prediction) == 0:
    #     #     st.write("It is a paper!")
    #     # elif np.argmax(prediction) == 1:
    #     #     st.write("It is a rock!")
    #     # else:
    #     #     st.write("It is a scissor!")
    #     #
    #     # st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    #     # st.write(prediction)
    #
    #     prediction = import_and_predict(imageI, model)
    #     pred = prediction[0][0]
    #     if (pred > 0.5):
    #         st.write("""
    #                              ## **Prediction:** You eye is Healthy. Great!!
    #                              """
    #                  )
    #         st.balloons()
    #     else:
    #         st.write("""
    #                              ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
    #                              """
    #                  )

    uploaded_files = st.file_uploader("Upload Chest Xray Images",
                                      type=["png", "PNG", "jpg", "jpeg", "tiff", "gif", "jfif", "raw"],
                                      accept_multiple_files=True)
    if uploaded_files is not None:
        # TO See details
        for image_file in uploaded_files:
            file_details = {"filename": image_file.name, "filetype": image_file.type,
                            "filesize": str(image_file.size/1024) + " KB"}
            imageIM = Image.open(image_file)
            st.image(imageIM, use_column_width=True)
            st.write(file_details)
            # st.image(load_image(image_file), width=250)
            prediction = import_and_predict(imageIM, model)
            pred = prediction[0][0]
            print(prediction)
            print("pred only is: ", pred)
            if prediction < 0.5:
                # if (pred > 0.5):
                st.write("""
                                     ## **Prediction:** Covid19 Detected!
                                     """
                         )
                new_space = '<br><br><hr>'
                st.markdown(new_space, unsafe_allow_html=True)
            else:
                st.write("""
                                     ## **Prediction:** Normal and healthy chest
                                     """
                         )
                st.balloons()
                new_space = '<br><br><hr>'
                st.markdown(new_space, unsafe_allow_html=True)

    else:
        st.text("You haven't uploaded an image or multiple images")

    # extra
    # st.write("""
    #              # ***Glaucoma detector***
    #              """
    #          )
    #
    # st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")
    #
    # file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])
    #
    # if file is None:
    #     st.text("You haven't uploaded a jpg image file")
    # else:
    #     imageI = Image.open(file)
    #     prediction = import_and_predict(imageI, model)
    #     pred = prediction[0][0]
    #     if (pred > 0.5):
    #         st.write("""
    #                      ## **Prediction:** You eye is Healthy. Great!!
    #                      """
    #                  )
    #         st.balloons()
    #     else:
    #         st.write("""
    #                      ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
    #                      """
    #                  )
    #
    #
