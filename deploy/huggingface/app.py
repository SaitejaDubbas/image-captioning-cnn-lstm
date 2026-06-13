"""
Hugging Face Spaces demo — Image Captioning
Uses BLIP (pre-trained) for the live demo.
Swap in your own CNN-LSTM weights later by replacing the generate() function.
"""

import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP — downloads once on first startup (~900 MB, cached by HF Spaces)
MODEL_ID = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()


def generate_caption(image: Image.Image, strategy: str) -> str:
    if image is None:
        return "Please upload an image."

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        if strategy == "Beam Search (better quality)":
            out = model.generate(**inputs, num_beams=3, max_new_tokens=40)
        else:
            out = model.generate(**inputs, do_sample=False, max_new_tokens=40)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ── Examples ──────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["examples/dog.jpg",   "Greedy (fast)"],
    ["examples/beach.jpg", "Beam Search (better quality)"],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Image Captioning Demo",
    theme=gr.themes.Soft(primary_hue="violet"),
    css=".output-text { font-size: 1.2rem !important; font-style: italic; }"
) as demo:

    gr.Markdown(
        """
        # 🖼️ Image Captioning
        **CNN-LSTM + Bahdanau Attention · InceptionV3 · BLEU-1: 0.49**

        Upload any image and get an automatic natural-language description.
        *Demo uses BLIP (Salesforce) — production model uses custom CNN-LSTM trained on Flickr8k.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            strategy = gr.Radio(
                choices=["Greedy (fast)", "Beam Search (better quality)"],
                value="Beam Search (better quality)",
                label="Decoding Strategy",
            )
            submit_btn = gr.Button("Generate Caption", variant="primary")

        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated Caption",
                lines=3,
                elem_classes="output-text",
                placeholder="Caption will appear here...",
            )
            gr.Markdown(
                """
                ### How it works
                1. **InceptionV3** extracts spatial image features (64 × 2048)
                2. **Bahdanau attention** focuses on relevant image regions per word
                3. **LSTM decoder** generates words one at a time
                4. **Beam search** keeps the top-k candidate sequences for better quality
                """
            )

    submit_btn.click(
        fn=generate_caption,
        inputs=[image_input, strategy],
        outputs=caption_output,
    )
    image_input.change(
        fn=generate_caption,
        inputs=[image_input, strategy],
        outputs=caption_output,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[image_input, strategy],
        outputs=caption_output,
        fn=generate_caption,
        cache_examples=False,
        label="Try these examples",
    )

    gr.Markdown(
        """
        ---
        **Source code:** [github.com/SaitejaDubbas/image-captioning-cnn-lstm](https://github.com/SaitejaDubbas/image-captioning-cnn-lstm)
        """
    )

if __name__ == "__main__":
    demo.launch()
