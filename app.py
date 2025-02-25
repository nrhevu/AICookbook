import gradio as gr
import os
from PIL import Image
import numpy as np

css = '''
.gradio-container {width: 85% !important}
'''

def process(images, text_input):
    pass

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="Multi-Image and Text Processor", css=css) as demo:
    gr.Markdown("# Image and Text Processing Interface")
    gr.Markdown("Upload multiple images and provide text input to get analysis results.")
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Drag (Select) 1 or more photos of your ingredients",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200, object_fit="contain", interactive=True)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
          
            text_input = gr.Textbox(label="Enter the name of the dish you want to prepare") 
            submit_button = gr.Button("Submit")

        with gr.Column():
            text_output = gr.Textbox(label="Results", lines=17)

    files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
    remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

    
    submit_button.click(
        fn=process,
        inputs=[files, text_input],
        outputs=text_output
    )
    gr.Markdown("## Instructions")
    gr.Markdown("""
                """)
            
demo.launch()