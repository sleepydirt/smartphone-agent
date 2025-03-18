import gradio as gr
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

def main(input, history):
    response = requests.post(url=f'{BACKEND_URL}/inference/',  params={'inputs': input}, stream=True, headers={'Content-Type': 'application/json'})

    full_text = ""
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            full_text += chunk
            yield full_text

demo = gr.ChatInterface(
    fn=main,
    type="messages",
    chatbot=gr.Chatbot(render_markdown=False)
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)