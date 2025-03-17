import gradio as gr
import requests
import time

def main(input, history):
    response = requests.post(url='http://127.0.0.1:8000/inference/', params={'inputs': input}, stream=True, headers={'Content-Type': 'application/json'})

    # Simulate streaming by yielding tokens one by one
    full_text = ""
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:  # filter out keep-alive chunks
            full_text += chunk
            yield full_text

gr.Interface(
    fn=main,
    inputs=gr.Textbox(label="Enter your message..."),
    outputs="text"
).launch()