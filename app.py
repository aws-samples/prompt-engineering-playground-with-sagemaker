import streamlit as st
import uuid
import os


THUMBS_UP = 1
THUMBS_DOWN = -1



# Check if the user ID is already stored in the session state
if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
    print(f"User ID: {user_id}")

# If the user ID is not yet stored in the session state, generate a random UUID
else:
    user_id = str(uuid.uuid4())
    st.session_state['user_id'] = user_id
    
if "results" not in st.session_state:
    st.session_state.results = []

if "input" not in st.session_state:
    st.session_state.input = ""

st.markdown("""
        <style>
               .block-container {
                    padding-top: 32px;
                    padding-bottom: 32px;
                    padding-left: 0;
                    padding-right: 0;
                }
        </style>
        """, unsafe_allow_html=True)

def write_logo():
    col1, col2, col3 = st.columns([5, 1, 5])
    with col2:
        st.image('images/docchat-icon.png', use_column_width='always') 
        
def get_user_input():
    col1, col2 = st.columns(2)
    uploaded_file = col1.file_uploader("Upload a document", type=["csv", "txt","html"])
    user_file_path="."
    # user uploads an image
    if uploaded_file is not None:
        user_file_path = os.path.join("docs", "processed_data.csv")
        with open(user_file_path, "wb") as user_file:
            user_file.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
            
    else:
        st.info("Please upload a file.")



    return user_file_path
 

input_image_path = get_user_input()

    
from model import get_chain, call_chain, update_log_engagement_with_feedback
chain = get_chain()

if input_image_path is None:
        # Display a message and stop execution if no file is uploaded
    st.warning("Please upload a file.")



def write_top_bar():
    col1, col2, col3 = st.columns([1,10,2])
    with col1:
        st.image('images/docchat-icon.png', use_column_width='always')
    with col2:
        st.subheader("Chat with SageMaker")
    with col3:
        clear = st.button("Clear Chat")
    return clear

clear = write_top_bar()

if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
        
def handle_input():
    input = st.session_state.input
    print("Handling input: ", input)
    result = call_chain(chain, input, st.session_state['user_id'])
    st.session_state.results.append(result)
    st.session_state.input = ""

def write_user_message(result: dict):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image('images/user-icon.png', use_column_width='always')
    with col2:
        st.warning(result['question'])

def render_answer(result: dict):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image('images/docchat-icon.png', use_column_width='always')  
    with col2:
        st.markdown(result['answer'])
      
def render_sources(result: dict):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image('images/docchat-icon.png', use_column_width='always')  
    with col2:
        for i, source in enumerate(result['sources']):
            st.markdown(f"### Source {i}: {source['source']}")
            st.markdown(source['page_content'])
            st.markdown('---')

def update_feedback(user_id: str, i: int, feedback: int):
    st.session_state.results[i]['feedback'] = feedback
    update_log_engagement_with_feedback(user_id, i, feedback)

#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_docchat_message(i: int, result: dict):
    answer = st.tabs(['Answer'])
    chat = st.container()
    rating = st.container()
    with chat:
        render_answer(result)
    with rating:
        if result['feedback'] is None:
            positive, negative, spacer = st.columns([1,1,10])
            with positive:
                st.button(
                    '👍',
                    key=str(i)+'positive_rate',
                    type='secondary',
                    args=(st.session_state.user_id, i, THUMBS_UP),
                    on_click=update_feedback
                )
            with negative:
                st.button(
                    '👎',
                    key=str(i)+'negative_rate',
                    type='secondary',
                    args=(st.session_state.user_id, i, THUMBS_DOWN),
                    on_click=update_feedback
                )
        else:
            st.write("Feedback provided, thank you!")
    st.markdown("""---""")
                    

with st.container():
  for i, result in enumerate(st.session_state.results):
    write_user_message(result)
    write_docchat_message(i, result)

st.markdown('---')
input = st.text_input("Ask a question about your docs:", key="input", on_change=handle_input)