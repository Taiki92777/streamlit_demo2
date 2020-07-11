import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from utils1 import load_model, get_prediction, get_heatmaps
import numpy as np
from keras import backend as K
plt.rcParams['figure.figsize'] = [14, 10]

def main():
    st.title("Histopathologic Cancer Detection")
    st.header("Task: Binary classification")
    st.write("Set up...")
    model, funcs, session = load_model() 
    st.write("Done!!")
        
    option = st.selectbox(
                            '',
                            ['Choose demo data or your data','Use Demo data','Use your data'])
    if option == 'Choose demo data or your data':
        st.write('')

    else:    
        if option == 'Use Demo data':
            # some pics
            image = Image.open('lymph_cancer.png')
            st.image(image, caption='Original histopathological scan (label = Cancer)', use_column_width=True)
                
            if model is not None:
                st.write("")
                st.write("Classifying ...")
                K.set_session(session)
                probas = get_prediction(image, model)
                probas = list(probas)
                st.write("Done!!")
                if probas is not None:
                    st.write("")
                    st.write("Making heatmap...")
                    heatmap = get_heatmaps(image, model, funcs)
                    st.write("Done!!")
                    st.write("")
                    plt.title("Cancer: " + f" {probas[0]:.3f}",fontsize=20)
                    plt.axis('off')
                    plt.imshow(ImageOps.fit(image, (96,96), Image.ANTIALIAS), cmap='gray')
                    plt.imshow(heatmap, cmap='magma', alpha=min(0.5, probas[0]))
                    st.pyplot()
            
        else:
            uploaded_file = st.file_uploader("Choose data to input (only JPG, JPEG or PNG)")

            if uploaded_file is not None:
                    # Upload image and confirm
                image = Image.open(uploaded_file)
                shape = np.asarray(image).shape
                print(shape)
                if len(shape) != 3:
                    st.write("Your image is gray-scale image, you need to input color image!!")
                else:
                    st.image(image, caption='Original histopathological scan', use_column_width=True)
                    
                    if model is not None:
                        K.set_session(session)
                        st.write("")
                        st.write("Classifying ...")
                        probas = get_prediction(image, model)
                        probas = list(probas)
                        st.write("Done!!")
                        if probas is not None:
                            st.write("")
                            st.write("Making heatmap...")
                            heatmap = get_heatmaps(image, model, funcs)
                            st.write("Done!!")
                            st.write("")
                            plt.title("Cancer: " + f" {probas[0]:.3f}",fontsize=20)
                            plt.axis('off')
                            plt.imshow(ImageOps.fit(image, (96,96), Image.ANTIALIAS), cmap='gray')
                            plt.imshow(heatmap, cmap='magma', alpha=min(0.5, probas[0]))
                            st.pyplot()  
                        
if __name__=="__main__":
    main()