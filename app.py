import gradio as gr
from src.inference import generate_answer

TITLE = "FastAPI Expert Assistant (TinyLlama + LoRA)"
DESCRIPTION = """
Ask anything about FastAPI and get expert-style answers.
This model is a TinyLlama-1.1B Chat fine-tuned with LoRA on a custom FastAPI Q&A dataset.
"""

def chat_fn(message, history):
    """
    Gradio chat handler.
    - message: latest user message (str)
    - history: list of (user, assistant) pairs
    """
    if not message.strip():
        return "", history

    answer = generate_answer(message)
    history = history + [(message, answer)]
    return "", history

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    chatbot = gr.Chatbot(label="Conversation")
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a FastAPI question...",
            show_label=False
        )
        send = gr.Button("Send")

    clear = gr.Button("Clear")

    send.click(
        fn=chat_fn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )
    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )
    clear.click(
        lambda: ("", []),
        outputs=[msg, chatbot],
    )

# HF Spaces runs this automatically; this is just for local testing
if __name__ == "__main__":
    demo.launch()
