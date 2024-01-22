# Optional stand-alone helper GUI to call the backend training functions.
import webbrowser

import modal

from .common import APP_NAME, VOLUME_CONFIG

stub = modal.Stub(f"{APP_NAME}-gui")
stub.q = modal.Queue.new()  # Pass back the URL to auto-launch

gradio_image = modal.Image.debian_slim().pip_install("gradio==4.5.0")


@stub.function(image=gradio_image, volumes=VOLUME_CONFIG, timeout=3600)
def gui():
    import glob

    import gradio as gr

    # Find the deployed function to call
    try:
        Inference = modal.Cls.lookup(APP_NAME, "Inference")
    except modal.exception.NotFoundError:
        raise Exception(
            "Must first deploy training backend."
        )

    def process_inference(model: str, input_text: str):
        text = f"{input_text}"
        yield text + "... (model loading)"
        try:
            for chunk in Inference(model.split("@")[-1]).completion.remote_gen(
                input_text
            ):
                text += chunk
                yield text
        except Exception as e:
            return repr(e)

    def get_model_choices():
        VOLUME_CONFIG["/runs"].reload()
        VOLUME_CONFIG["/pretrained"].reload()
        choices = [
            *glob.glob("/runs/*/lora-out/merged", recursive=True),
            *glob.glob("/pretrained/models--*/snapshots/*", recursive=True),
        ]
        choices = [f"{choice.split('/')[2]}@{choice}" for choice in choices]
        return reversed(sorted(choices))

    with gr.Blocks() as interface:
        with gr.Column():
            with gr.Group():
                model_dropdown = gr.Dropdown(
                    label="Select Model", choices=get_model_choices()
                )
                gr.Button("Refresh", size="sm")
            input_text = gr.Textbox(
                label="Input Text (please include prompt manually)",
                lines=10,
                value="<|im_start|>Ferdous ฿hai\n>>>gm<|im_end|>\n<|im_start|>Mourad\n",
            )
            inference_button = gr.Button(
                "Run Inference", variant="primary", size="sm"
            )
            inference_output = gr.Textbox(label="Output", lines=20)
            inference_button.click(
                process_inference,
                inputs=[model_dropdown, input_text],
                outputs=inference_output,
            )

    with modal.forward(8000) as tunnel:
        stub.q.put(tunnel.url)
        interface.launch(quiet=True, server_name="0.0.0.0", server_port=8000)


@stub.local_entrypoint()
def main():
    handle = gui.spawn()
    url = stub.q.get()
    print(f"GUI available at -> {url}\n")
    webbrowser.open(url)
    handle.get()
