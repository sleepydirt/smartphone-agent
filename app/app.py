import gradio as gr
import aiohttp
import asyncio
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    async def user(user_message):
        return "", [{"role": "user", "content": user_message}]
    
    async def bot(messages):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BACKEND_URL}/inference/",
                json={"inputs": messages[0]["content"]}
            ) as response:
                async for chunk in response.content:
                    chunk = chunk.decode()
                    if chunk:
                        message = {"role": "assistant", "content": ""}
                        messages.append(message)
                        for character in chunk:
                            messages[-1]["content"] += character
                            await asyncio.sleep(0.01)
                            yield messages

    msg.submit(user, [msg], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)