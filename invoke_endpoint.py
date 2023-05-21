import boto3
from collections import defaultdict
import io
import jinja2
import json
import os
from pathlib import Path
import random
from streamlit_ace import st_ace
import streamlit as st
import string
import ai21
from io import StringIO
import re


N = 7
sagemaker_runtime = boto3.client("runtime.sagemaker")
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)


code_example = """{
  "model_name": "example",
  "model_type": "AI21-SUMMARY",
  "endpoint_name": "summarize",
  "payload": {
    "parameters": {
      "max_length": {
        "default": 200,
        "range": [
          10,
          500
        ]
      },
      "num_return_sequences": {
        "default": 10,
        "range": [
          0,
          10
        ]
      },
      "num_beams": {
        "default": 3,
        "range": [
          0,
          10
        ]
      },
      "temperature": {
        "default": 0.5,
        "range": [
          0,
          1
        ]
      },
      "early_stopping": {
        "default": true,
        "range": [
          true,
          false
        ]
      },
      "stopwords_list": {
        "default": [
          "stop",
          "dot"
        ],
        "range": [
          "a",
          "an",
          "the",
          "and",
          "it",
          "for",
          "or",
          "but",
          "in",
          "my",
          "your",
          "our",
          "stop",
          "dot"
        ]
      }
    }
  }
}
"""

parameters_help_map = {
    "max_length": "Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a positive integer.",
    "num_return_sequences": "Number of output sequences returned. If specified, it must be a positive integer.",
    "num_beams": "Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.",
    "no_repeat_ngram_size": "Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.",
    "temperature": "Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.",
    "early_stopping": "If True, text generation is finished when all beam hypotheses reach the end of stence token. If specified, it must be boolean.",
    "do_sample": "If True, sample the next word as per the likelyhood. If specified, it must be boolean.",
    "top_k": "In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.",
    "top_p": "In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.",
    "seed": "Fix the randomized state for reproducibility. If specified, it must be an integer.",
}

example_list = [" ", "Table Q&A", "Product description", "Summarize reviews", "Generate SQL"]
example_context_ai21_qa = ["Sample Context ", "Financial", "Healthcare"]
example_prompts_ai21 = {
    "Table Q&A": "| Ship Name | Color | Total Passengers | Status | Captain | \n \
| Symphony | White | 7700 | Active | Mike | \n \
| Wonder | Grey | 7900 | Under Construction | Anna | \n \
| Odyssey | White | 5800 | Active | Mohammed | \n \
| Quantum | White | 5700 | Active | Ricardo | \n \
| Mariner | Grey | 4300 | Active | Saanvi | \n \
Q: Which active ship carries the most passengers? \n \
A: Symphony \n \
Q: What is the color of the ship whose captain is Saanvi? \n \
A: Grey \n \
Q: How many passengers does Ricardo's ship carry? \n \
A:",
    "Product description": "Write an engaging product description for clothing eCommerce site. Make sure to include the following features in the description. \n \
Product: Humor Men's Graphic T-Shirt with a print of Einstein's quote: \"artificial intelligence is no match for natural stupidity” \n \
Features: \n \
- Soft cotton \n \
- Short sleeve \n \
Description:",
    "Summarize reviews": "Summarize the following restaurant review \n \
Restaurant: Luigi's \n \
Review: We were passing through SF on a Thursday afternoon and wanted some Italian food. We passed by a couple places which were packed until finally stopping at Luigi's, mainly because it was a little less crowded and the people seemed to be mostly locals. We ordered the tagliatelle and mozzarella caprese. The tagliatelle were a work of art - the pasta was just right and the tomato sauce with fresh basil was perfect. The caprese was OK but nothing out of the ordinary. Service was slow at first but overall it was fine. Other than that - Luigi's great experience! \n \
Summary: Local spot. Not crowded. Excellent tagliatelle with tomato sauce. Service slow at first. \n \
## \n \
Summarize the following restaurant review \n \
Restaurant: La Taqueria \n \
Review: La Taqueria is a tiny place with 3 long tables inside and 2 small tables outside. The inside is cramped, but the outside is pleasant. Unfortunately, we had to sit inside as all the outside tables were taken. The tacos are delicious and reasonably priced and the salsa is spicy and flavorful. Service was friendly. Aside from the seating, the only thing I didn't like was the lack of parking - we had to walk six blocks to find a spot. \n \
Summary:",
    
"Generate SQL": "Create SQL statement from instruction. \n \
Database: Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)\n \
Request: all the countries we have customers in without repetitions.\n \
SQL statement:\n \
SELECT DISTINCT Country FROM Customers;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Orders(OrderID, CustomerID, EmployeeID, OrderDate, ShipperID)\n \
Request: select all the orders from customer id 1.\n \
SQL statement:\n \
SELECT * FROM Orders\n \
WHERE CustomerID = 1;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Products(ProductID, ProductName, SupplierID, CategoryID, Unit, Price)\n \
Request: selects all products from categories 1 and 7\n \
SQL statement:\n \
SELECT * FROM Products\n \
WHERE CategoryID = 1 OR CategoryID = 7;\n \
##\n \
Create SQL statement from instruction.\n \
Database: Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)\n \
Request: change the first customer's name to Alfred Schmidt who lives in Frankfurt city.\n \
SQL statement:",
}

example_context_ai21_qa = {
    "Financial": "n 2020 and 2021, enormous QE — approximately $4.4 trillion, or 18%, of 2021 gross domestic product (GDP) — and enormous fiscal stimulus (which has been and always will be inflationary) — approximately $5 trillion, or 21%, of 2021 GDP — stabilized markets and allowed companies to raise enormous amounts of capital. In addition, this infusion of capital saved many small businesses and put more than $2.5 trillion in the hands of consumers and almost $1 trillion into state and local coffers. These actions led to a rapid decline in unemployment, dropping from 15% to under 4% in 20 months — the magnitude and speed of which were both unprecedented. Additionally, the economy grew 7% in 2021 despite the arrival of the Delta and Omicron variants and the global supply chain shortages, which were largely fueled by the dramatic upswing in consumer spending and the shift in that spend from services to goods. Fortunately, during these two years, vaccines for COVID-19 were also rapidly developed and distributed. \n \
In today's economy, the consumer is in excellent financial shape (on average), with leverage among the lowest on record, excellent mortgage underwriting (even though we've had home price appreciation), plentiful jobs with wage increases and more than $2 trillion in excess savings, mostly due to government stimulus. Most consumers and companies (and states) are still flush with the money generated in 2020 and 2021, with consumer spending over the last several months 12% above pre-COVID-19 levels. (But we must recognize that the account balances in lower-income households, smaller to begin with, are going down faster and that income for those households is not keeping pace with rising inflation.) \n \
Today's economic landscape is completely different from the 2008 financial crisis when the consumer was extraordinarily overleveraged, as was the financial system as a whole — from banks and investment banks to shadow banks, hedge funds, private equity, Fannie Mae and many other entities. In addition, home price appreciation, fed by bad underwriting and leverage in the mortgage system, led to excessive speculation, which was missed by virtually everyone — eventually leading to nearly $1 trillion in actual losses.",
    "Healthcare": "A heart attack occurs when blood flow that brings \
oxygen-rich blood to the heart muscle is severely \
reduced or cut off. This is due to a buildup of fat,\
cholesterol and other substances (plaque) that narrows\
coronary arteries. This process is called atherosclerosis.\
When plaque in a heart artery breaks open, a blood clot\
forms. The clot can block blood flow. When it completely\
stops blood flow to part of the heart muscle, that\
portion of muscle begins to die. Damage increases the\
longer an artery stays blocked. Once some of the heart\
muscle dies, permanent heart damage results.\
The amount of damage to the heart muscle depends on\
the size of the area supplied by the blocked artery and\
the time between injury and treatment. The blocked\
artery should be opened as soon as possible to reduce\
heart damage. \n \
Atherosclerosis develops over time. It often has no symptoms\
until enough damage lessens blood flow to your heart\
muscle. That means you usually can’t feel it happening until\
blood flow to heart muscle is blocked. \n \
You should know the warning signs of heart attack so you\
can get help right away for yourself or someone else.\
Some heart attacks are sudden and intense. But most start\
slowly, with mild pain or discomfort. Signs of a heart attack\
include:\n\
• Uncomfortable pressure, squeezing, fullness or pain in the\
center of your chest. It lasts more than a few minutes, or\
goes away and comes back.\n\
• Pain or discomfort in one or both arms, your back, neck,\
jaw or stomach.\n\
• Shortness of breath with or without chest discomfort.\n\
• Other signs such as breaking out in a cold sweat, nausea\
or lightheadedness."
}
parameters_help_map = defaultdict(str, parameters_help_map)
example_prompts_ai21 = defaultdict(str, example_prompts_ai21)
example_context_ai21_qa = defaultdict(str, example_context_ai21_qa)

def list_templates(dir_path):
    # folder path
    templates = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            templates.append(path.split(".")[0])
    return templates


def read_template(template_path):
    template = template_env.get_template(template_path)
    output_text = template.render()
    return output_text


def is_valid_default(parameter, minimum, maximum):
    # parameter is a list
    if type(parameter) == list:
        return True

    # parameter is an int or float and is in valid range
    if parameter <= maximum and parameter >= minimum:
        return True

    # parameter is a bool
    if type(parameter) == bool and type(minimum) == bool and type(maximum) == bool:
        return True
    return False

def generate_text(payload, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read())
    
    # TO DO: results are either dictionary or list
    # - this works for faster transformr and DJL containers
    for item in result:
        #print(f" Item={item}, type={type(item)}")
        if isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    #print(f"List:element::is: {element['generated_text']} ")
                    return element["generated_text"]
        elif isinstance(item, str):
            # print(item["generated_text"])
            # return item["generated_text"]
            print(f"probably:Item:from:dict::result[item]={result[item]}")
            return result[item]
        else:
            # print(item["generated_text"])
            return item["generated_text"]

def generate_text_ai21(payload, endpoint_name):
    print("payload type: ", type(payload))
    response = ai21.Completion.execute(sm_endpoint=endpoint_name,
                                   prompt=payload["text_inputs"],
                                   maxTokens=payload["maxTokens"],
                                   temperature=payload["temperature"],
                                   numResults=payload["numResults"], 
                                   # stopSequences=['##']
                                      )

    print(response['completions'][0]['data']['text'])
    return response['completions'][0]['data']['text']

def generate_text_ai21_summarize(payload, endpoint_name):
    # segmented_list = ai21.Segmentation.execute(
    #     source=payload["text_inputs"],
    #     sourceType='TEXT')
    response = ai21.Summarize.execute(
        source=payload["text_inputs"],
        sourceType="TEXT",
        sm_endpoint=endpoint_name)
    return response['summary']

def generate_text_ai21_context_qa(payload, question, endpoint_name):
    print('----- Context -------', payload["text_inputs"])
    print('----- Question ------', question)
    response = ai21.Answer.execute(
    context=payload["text_inputs"],
    question=question,
    sm_endpoint=endpoint_name)
    return response['answer']

def get_user_input():
    uploaded_file = st.file_uploader(label="Upload JSON Template", type=["json"])
    uploaded_file_location = st.empty()

    # user uploads an image
    if uploaded_file is not None:
        input_str = json.load(uploaded_file)
        if validate_json_template(input_str):
            user_file_path = os.path.join(
                "templates", input_str["model_name"] + ".template.json"
            )
            with open(user_file_path, "wb") as user_file:
                user_file.write(uploaded_file.getbuffer())
            uploaded_file_location.write("Template Uploaded: " + str(user_file_path))
            st.session_state["new_template_added"] = True
        else:
            uploaded_file_location.warning(
                "Invalid Input: please upload a valid template.json"
            )
    else:
        user_file_path = None

    return user_file_path


@st.cache_data
def validate_json_template(json_dictionary):
    expected_keys = {"model_name", "endpoint_name", "payload"}
    actual_keys = set(json_dictionary.keys())

    if not expected_keys.issubset(actual_keys):
        st.warning(
            "Invalid Input: template.json must contain a modelName, endpoint_name, and payload keys"
        )
        return False

    if not "parameters" in json_dictionary["payload"].keys() and not type(
        json_dictionary["payload"]["parameters"] == list
    ):
        st.warning(
            "Invalid Input: template.json must contain a payload key with parameters listed"
        )
        return False
    return True


@st.cache_data
def handle_editor_content(input_str):
    if validate_json_template(input_str):
        try:
            model_name = input_str["model_name"]
            filename = model_name + ".template.json"
            user_file_path = os.path.join("templates", filename)
            with open(user_file_path, "w+") as f:
                json.dump(input_str, f)

            st.write("json saved at " + str(user_file_path))
            st.session_state["new_template_added"] = True

        except Exception as e:
            st.write(e)


def handle_parameters(parameters):
    for p in parameters:
        minimum = parameters[p]["range"][0]
        maximum = parameters[p]["range"][-1]
        default = parameters[p]["default"]
        parameter_range = parameters[p]["range"]
        parameter_help = parameters_help_map[p]
        if not is_valid_default(default, minimum, maximum):
            st.warning(
                "Invalid Default: "
                + p
                + " default value does not follow the convention default >= min and default <= max."
            )
        elif len(parameter_range) > 2:
            if not set(default).issubset(set(parameter_range)):
                st.warning(
                    "Invalid Default: "
                    + p
                    + " Every Multiselect default value must exist in options"
                )
            else:
                parameters[p] = st.sidebar.multiselect(
                    p, options=parameter_range, default=default
                )

        elif type(minimum) == int and type(maximum) == int and type(default) == int:
            parameters[p] = st.sidebar.slider(
                p,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1,
                help=parameter_help,
            )
        elif type(minimum) == bool and type(maximum) == bool and type(default) == bool:
            parameters[p] = st.sidebar.selectbox(
                p, ["True", "False"], help=parameter_help
            )
        elif (
            type(minimum) == float and type(maximum) == float and type(default) == float
        ):
            parameters[p] = st.sidebar.slider(
                p,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default),
                step=0.01,
                help=parameter_help,
            )
        else:
            st.warning(
                "Invalid Parameter: "
                + p
                + " is not a valid parameter for this model or the parameter type is not supported in this demo."
            )
    return parameters

def on_clicked():
    st.session_state.text = example_prompts_ai21[st.session_state.task]

def on_clicked_qa():
    st.session_state.text = example_context_ai21_qa[st.session_state.taskqa]

def main():
    default_endpoint_option = "Select"
    st.session_state["new_template_added"] = False
    sidebar_selectbox = st.sidebar.empty()
    selected_endpoint = sidebar_selectbox.selectbox(
        label="Select the endpoint to run in SageMaker",
        options=[default_endpoint_option] + list_templates("templates"),
    )

    st.sidebar.title("Model Parameters")
    st.image("./ml_image_prompt.png")
    
    # Adding your own model
    with st.expander("Add a New Model"):
        st.header("Add a New Model")
        st.write(
            """Add a new model by uploading a .template.json file or by pasting the dictionary
                in the editor. A model template is a json dictionary containing a model_name,
                endpoint_name, and payload with parameters.  \n \n Below is an example of a
                template.json"""
        )
        res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        get_user_input()

        # Spawn a new Ace editor and display editor's content as you type
        content = st_ace(
            theme="tomorrow_night",
            wrap=True,
            show_gutter=True,
            language="json",
            value=code_example,
            keybinding="vscode",
            min_lines=15,
        )

        if content != code_example:
            input_str = json.loads(content)
            handle_editor_content(input_str)
            templates = list_templates("templates")

        if st.session_state["new_template_added"]:
            res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
            selected_endpoint = sidebar_selectbox.selectbox(
                label="Select the endpoint to run in SageMaker",
                options=list_templates("templates"),
                key=res,
            )

    # Prompt Engineering Playground
    st.header("Prompt Engineering Playground")
    if selected_endpoint != default_endpoint_option:
        output_text = read_template(f"templates/{selected_endpoint}.template.json")
        output = json.loads(output_text)
        parameters = output["payload"]["parameters"]
        print("parameters ------------------ ", parameters)
        if parameters != "None":
            parameters = handle_parameters(parameters)

        st.markdown(
           output["description"]
        )
    if selected_endpoint == "AI21-J2-GRANDE-INSTRUCT":
        selected_task = st.selectbox(
            label="Example prompts",
            options=example_list, 
            on_change=on_clicked, 
            key="task"
            )
    if selected_endpoint == "AI21-CONTEXT-QA":
        selected_task = st.selectbox(
            label="Example context",
            options=example_context_ai21_qa, 
            on_change=on_clicked_qa, 
            key="taskqa"
            )
    if selected_endpoint == "AI21-SUMMARY" or selected_endpoint == "AI21-CONTEXT-QA":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

            # To read file as string:
            string_data = stringio.read()
            st.session_state.text = string_data
            prompt = st.session_state.text
        
    prompt = st.text_area("Enter your prompt here:", height=350, key="text")
    if selected_endpoint == "AI21-CONTEXT-QA":
        question = st.text_area("Enter your question here", height=80, key="question")
    placeholder = st.empty()

    if st.button("Run"):
        final_text=""
        if selected_endpoint != default_endpoint_option:
            placeholder = st.empty()
            endpoint_name = output["endpoint_name"]
            print(parameters)
            # payload = {
            #     "text_inputs": [
            #         prompt,
            #     ],
            #     parameters
            # }
            if parameters != "None":
                payload = {"text_inputs": prompt, **parameters}
            else: 
                payload = {"text_inputs": prompt}
            if output["model_type"] == "AI21":
                print('-------- Payload ----------', payload)
                generated_text = generate_text_ai21(payload, endpoint_name)
                final_text = f''' {generated_text} ''' # to take care of multi line prompt
                st.write(final_text)
            elif output["model_type"] == "AI21-SUMMARY":
                generated_text = generate_text_ai21_summarize(payload, endpoint_name)
                summaries = generated_text.split("\n")
                for summary in summaries:
                    st.markdown("- " + summary)
                    final_text+=summary
            elif output["model_type"] == "AI21-CONTEXT-QA":
                generated_text = generate_text_ai21_context_qa(payload, question, endpoint_name)
                final_text = f''' {generated_text} ''' # to take care of multi line prompt
                st.write(final_text)
            else: 
                generated_text = generate_text(payload, endpoint_name)
                final_text = f''' {generated_text} ''' # to take care of multi line prompt
                st.write(final_text)
        else:
            st.warning("Invalid Endpoint: Please select a valid endpoint")
        st.download_button("Download", final_text, file_name="output.txt")

if __name__ == "__main__":
    main()
