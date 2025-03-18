import gradio as gr
import requests

def main(input, history):
    response = requests.post(url='http://127.0.0.1:8000/inference/', params={'inputs': input}, stream=True, headers={'Content-Type': 'application/json'})

    full_text = ""
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            full_text += chunk
            yield full_text

gr.ChatInterface(
    fn=main,
    type="messages",
).launch()