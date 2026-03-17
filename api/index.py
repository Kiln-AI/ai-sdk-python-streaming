import logging
from typing import List
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Query, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from vercel.headers import set_headers

from api.utils.prompt import ClientMessage, convert_to_openai_message
from api.utils.storage import fake_storage
from .utils.stream import StreamTransportProtocol, get_stream_transport_func, patch_response_with_headers

from kiln_ai.utils.config import Config
from kiln_ai.adapters.model_adapters.base_adapter import BaseAdapter
from kiln_ai.adapters.adapter_registry import adapter_for_task
from kiln_ai.datamodel import Task, StructuredOutputMode
from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _vercel_set_headers(request: FastAPIRequest, call_next):
    set_headers(dict(request.headers))
    return await call_next(request)


class Request(BaseModel):
    messages: List[ClientMessage]

# Set to False to prevent writing the TaskRuns to filesystem automatically
Config.shared().autosave_runs = True


def task_adapter_factory(task_path: str | Path) -> BaseAdapter:
    task = Task.load_from_folder((Path(task_path)))

    adapter = adapter_for_task(task, KilnAgentRunConfigProperties(
        model_name="minimax_m2_5",
        model_provider_name=ModelProviderName.openrouter,
        prompt_id="id::123324894662",
        structured_output_mode=StructuredOutputMode.default,
        tools_config=ToolsRunConfig(tools=[
            "kiln_tool::add_numbers",
            "kiln_tool::subtract_numbers",
            "kiln_tool::multiply_numbers",
            "kiln_tool::divide_numbers",
        ]),
        thinking_level="medium",
    ))

    return adapter

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    messages = request.messages

    # we support two protocols:
    # - OpenAI Chat protocol -> the limitation is that it is not built around the idea of multiturn. It assumes the client is
    # involved in each toolcall and as such does not support toolcall output events (since they are client-produced).
    # 
    # - AI SDK protocol -> this is much more extensible and comes with events for toolcall outputs and so on. With this
    # protocol, you get all the events, reasoning, content, toolcall and their outputs, etc.
    # stream_protocol = StreamTransportProtocol.OPENAI
    stream_protocol = StreamTransportProtocol.AI_SDK
    transport_func = get_stream_transport_func(stream_protocol)

    # This is a toy task that expects the user to give numbers, and then it does random math operations (using tools)
    # and comes back with a made up factoid that uses the result of the math operations
    task_path = "./kiln_projects/kiln_factoid_export/tasks/155620075005 - Rocket science"
    kiln_adapter = task_adapter_factory(task_path)

    # In your code, you would probably have a session ID, and retrieve the TaskRun or TaskRun->trace from the DB
    current_session = fake_storage.get_task_run()
    if current_session:
        logger.info(f"Continuing session: {current_session.id} with trace length: {len(current_session.trace or [])}")
    else:
        logger.info("Starting new session")

    # The trace already holds all the messages up to this point, so we only need the new one coming out of the UI
    # it must be in OpenAI format (dict{ role: str, content: str })
    new_message = convert_to_openai_message(messages[-1])

    response = StreamingResponse(
        transport_func(kiln_adapter, new_message, current_session),
        media_type="text/event-stream",
    )
    return patch_response_with_headers(response, protocol)
