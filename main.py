import streamlit as st
import subprocess
from PIL import Image
import webbrowser
import os

streamlit_ports = [8502,8503]

def main():
    global streamlit_ports
    st.title("GenAI on SageMaker")
    image = Image.open("images/genAI.png")
    st.image(image, caption="")

    app_selection = st.selectbox("Start an app", [""] + ["Multi task Prompt Engineering with SageMaker", "SageMakerChat (RAG)"])

    success = False  # Initialize the success variable

    if app_selection == "Multi task Prompt Engineering with SageMaker":
        success = open_app("invoke_endpoint.py", "Multi task Prompt Engineering with SageMaker", streamlit_ports[0] )
        if success:
            st.success(f"{app_selection} app started successfully!")

        st.markdown(get_button_html(app_selection,streamlit_ports[0]  ), unsafe_allow_html=True)
    elif app_selection == "SageMakerChat (RAG)":
        success = open_app("app.py", "SageMakerChat (RAG)", streamlit_ports[1])
        if success:
            st.success(f"{app_selection} app started successfully!")

        st.markdown(get_button_html(app_selection,streamlit_ports[1]  ), unsafe_allow_html=True)

def get_button_html(app_name,port):

    url = f"https://d-dleaobu4yqba.studio.us-east-1.sagemaker.aws/jupyter/default/proxy/{port}/"
    print(url)
    button_text = f"Open {app_name} App"
    button_html = f'<a href="{url}" style="text-decoration:none; background-color:#4F8BF9; color:#FFFFFF; padding: 10px 20px; border-radius: 5px; font-weight:bold;">{button_text}</a>'
    return button_html

def open_app(app_file, app_name,port):

    command = ["streamlit", "run", app_file, "--server.port", str(port)]
    try:
        subprocess.Popen(command)
        return True
    except Exception as e:
        print(e)  # You can handle the exception as per your requirements
        return False

if __name__ == "__main__":
    main()
