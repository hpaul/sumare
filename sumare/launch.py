
import os
import gradio as gr
import requests

auth_token = os.getenv("auth_token")
title = "Literature Review Summarization"
description = ""

# Summarisation models
summary_models = [
    "google/pegasus-large",
    "t5-large",
    "sshleifer/distilbart-cnn-12-6",
    "sshleifer/distilbart-xsum-12-6",
    "domenicrosati/t5-finetuned-parasci"
]

# Paraphrase models
paraphrase_models = [
    "prithivida/parrot_paraphraser_on_T5"
]


def transform_text(model, text):
    # Call huggingface API based on input model
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {auth_token}"}
    data = {"inputs": text}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Treat the first key of response as output
    output = response.json()[0]
    keys = list(output.keys())
    return output[keys[0]]


def summary_panel():
    # Input values
    model_selector = gr.Dropdown(
        choices=summary_models,
        value="google/pegasus-large",
        label="Alege un model pentru pentru sumarizare"
    )
    input_text = gr.Textbox(
        label="Text", lines=10,
        placeholder="Input literature review..."
    )

    # The output value
    output_text = gr.Textbox(
        label="Output", lines=10,
        placeholder="Aici vor fi paragrafele importante.."
    )

    return gr.Interface(
        fn=transform_text,
        inputs=[model_selector, input_text],
        outputs=output_text,
    )


def paraphrase_panel():
    # Input values
    model_selector = gr.Dropdown(
        choices=paraphrase_models,
        value="prithivida/parrot_paraphraser_on_T5",
        label="Alege un model pentru parafrazare"
    )
    input_text = gr.Textbox(
        label="Text", lines=10,
        placeholder="Input text sumarizat..."
    )

    # The output value
    output_text = gr.Textbox(
        label="Output", lines=10,
        placeholder="Textul parafrazat"
    )

    return gr.Interface(
        fn=transform_text,
        inputs=[model_selector, input_text],
        outputs=output_text,
    )


demo = gr.TabbedInterface(
    [summary_panel(), paraphrase_panel()],
    ["Sumarizare", "Parafrazare"],
)

demo.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=7680,
    debug=True,
)
