# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal-labs--tgi-app.modal.run).

from pathlib import Path

import modal
from modal import Mount, asgi_app

from .common import APP_NAME, stub

frontend_path = Path(__file__).parent / "llm-frontend"
run_folder = "/runs/axo-2024-01-12-11-16-07-92e1" #TODO: get latest run folder

@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 10,
)
@asgi_app(label="web-app")
def app():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    # Find the deployed function to call
    try:
        Inference = modal.Cls.lookup(APP_NAME, "Inference")
    except modal.exception.NotFoundError:
        raise Exception(
            "Must first deploy training backend."
        )

    @web_app.get("/stats")
    async def stats():
        stats = await Inference(f"{run_folder}/lora-out/merged").completion.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners
        }

    @web_app.get("/completion/{input_text}")
    async def completion(input_text: str):
        from urllib.parse import unquote

        async def generate():
            async for chunk in Inference(f"{run_folder}/lora-out/merged").completion.remote_gen.aio(unquote(input_text)):
                yield f"data: {json.dumps(dict(text=chunk), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app
