from kiln_ai.datamodel import TaskRun
from enum import Enum
from kiln_ai.adapters.model_adapters.base_adapter import BaseAdapter
import logging
import json
import traceback
import uuid
from typing import Any, Callable, Dict, AsyncIterator, Protocol

from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from api.utils.storage import fake_storage

logger = logging.getLogger(__name__)

class StreamTransportProtocol(str, Enum):
    """SSE protocol version."""
    OPENAI = "openai"
    AI_SDK = "ai-sdk"

class StreamTransportFunc(Protocol):
    def __call__(
        self,
        adapter: BaseAdapter,
        new_message: ChatCompletionMessageParam,
        task_run: TaskRun | None = None,
    ) -> AsyncIterator[str]:
        ...

async def stream_text_ai_sdk_transport(
    adapter: BaseAdapter,
    new_message: ChatCompletionMessageParam,
    task_run: TaskRun | None = None,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for a streaming chat completion using the AI SDK transport.
    """
    try:
        def format_sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"

        # Extract user input from last message for the adapter
        new_message_content = new_message.get("content", "") if new_message.get("content") else ""
        if not isinstance(new_message_content, str):
            raise ValueError(f"New message content must be a string, got {type(new_message_content)}")

        stream = adapter.invoke_ai_sdk_stream(
            input=new_message_content,
            prior_trace=task_run.trace if task_run else None,
        )

        # events coming out of the stream are already in AI SDK protocol and do not need conversion
        # we get granular events for every toolcall (including full input and output)
        async for chunk in stream:
            yield format_sse(chunk.model_dump())

        # after exhausting the stream, we get the full TaskRun object which contains the trace that
        # we can pass in on the next invoke to continue the conversation
        fake_storage.store_task_run(stream.task_run)
    
    except Exception:
        traceback.print_exc()
        raise

async def stream_text_openai(
    adapter: BaseAdapter,
    new_message: ChatCompletionMessageParam,
    task_run: TaskRun | None = None,
    tools: Dict[str, Callable[..., Any]] | None = None,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for a streaming chat completion using the OpenAI protocol.

    The OpenAI protocol cannot be used to stream tool calls from the stream - only toolcall args
    are supported.

    Use the AI SDK transport to stream tool calls with better control (args, output, error, etc.)
    """
    try:
        def format_sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"

        message_id = f"msg-{uuid.uuid4().hex}"
        step_index = 0
        text_stream_id = f"text-{step_index}"
        reasoning_stream_id = f"reasoning-{step_index}"
        text_started = False
        text_finished = False
        reasoning_started = False
        step_started = False
        finish_reason = None
        usage_data = None
        tool_calls_state: Dict[int, Dict[str, Any]] = {}

        yield format_sse({"type": "start", "messageId": message_id})

        # Extract user input from last message for the adapter
        new_message_content = new_message.get("content", "") if new_message.get("content") else ""
        if not isinstance(new_message_content, str):
            raise ValueError(f"New message content must be a string, got {type(new_message_content)}")

        stream = adapter.invoke_openai_stream(
            input=new_message_content,
            prior_trace=task_run.trace if task_run else None,
        )

        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage_data = chunk_usage
            for choice in chunk.choices:
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                # Emit start-step on the first meaningful content of each step
                has_content = (
                    bool(getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None))
                    or bool(delta.content)
                    or bool(delta.tool_calls)
                )
                if has_content and not step_started:
                    yield format_sse({"type": "start-step"})
                    step_started = True

                # Reasoning (AI SDK protocol: reasoning-start, reasoning-delta, reasoning-end)
                reasoning_content = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning_content is not None and reasoning_content != "":
                    if not reasoning_started:
                        yield format_sse(
                            {"type": "reasoning-start", "id": reasoning_stream_id}
                        )
                        reasoning_started = True
                    yield format_sse(
                        {
                            "type": "reasoning-delta",
                            "id": reasoning_stream_id,
                            "delta": reasoning_content,
                        }
                    )

                if delta.content:
                    if reasoning_started:
                        yield format_sse(
                            {"type": "reasoning-end", "id": reasoning_stream_id}
                        )
                        reasoning_started = False
                    if not text_started:
                        yield format_sse({"type": "text-start", "id": text_stream_id})
                        text_started = True
                    yield format_sse(
                        {"type": "text-delta", "id": text_stream_id, "delta": delta.content}
                    )

                if delta.tool_calls:
                    if reasoning_started:
                        yield format_sse(
                            {"type": "reasoning-end", "id": reasoning_stream_id}
                        )
                        reasoning_started = False
                    for tool_call_delta in delta.tool_calls:
                        index = tool_call_delta.index
                        state = tool_calls_state.setdefault(
                            index,
                            {
                                "id": None,
                                "name": None,
                                "arguments": "",
                                "started": False,
                            },
                        )

                        if tool_call_delta.id is not None:
                            state["id"] = tool_call_delta.id
                            if (
                                state["id"] is not None
                                and state["name"] is not None
                                and not state["started"]
                            ):
                                yield format_sse(
                                    {
                                        "type": "tool-input-start",
                                        "toolCallId": state["id"],
                                        "toolName": state["name"],
                                    }
                                )
                                state["started"] = True

                        function_call = getattr(tool_call_delta, "function", None)
                        if function_call is not None:
                            if function_call.name is not None:
                                state["name"] = function_call.name
                                if (
                                    state["id"] is not None
                                    and state["name"] is not None
                                    and not state["started"]
                                ):
                                    yield format_sse(
                                        {
                                            "type": "tool-input-start",
                                            "toolCallId": state["id"],
                                            "toolName": state["name"],
                                        }
                                    )
                                    state["started"] = True

                            if function_call.arguments:
                                if (
                                    state["id"] is not None
                                    and state["name"] is not None
                                    and not state["started"]
                                ):
                                    yield format_sse(
                                        {
                                            "type": "tool-input-start",
                                            "toolCallId": state["id"],
                                            "toolName": state["name"],
                                        }
                                    )
                                    state["started"] = True

                                state["arguments"] += function_call.arguments
                                if state["id"] is not None:
                                    yield format_sse(
                                        {
                                            "type": "tool-input-delta",
                                            "toolCallId": state["id"],
                                            "inputTextDelta": function_call.arguments,
                                        }
                                    )

                # When this turn ends with tool_calls, emit tool events and finish-step so the stream continues
                if choice.finish_reason == "tool_calls":
                    # Close any open blocks before finishing the step
                    if reasoning_started:
                        yield format_sse(
                            {"type": "reasoning-end", "id": reasoning_stream_id}
                        )
                        reasoning_started = False
                    if text_started and not text_finished:
                        yield format_sse({"type": "text-end", "id": text_stream_id})
                        text_finished = True
                    tool_map = tools if tools is not None else {}
                    for index in sorted(tool_calls_state.keys()):
                        state = tool_calls_state[index]
                        tool_call_id = state.get("id")
                        tool_name = state.get("name")
                        if tool_call_id is None or tool_name is None:
                            continue
                        if not state["started"]:
                            yield format_sse(
                                {
                                    "type": "tool-input-start",
                                    "toolCallId": tool_call_id,
                                    "toolName": tool_name,
                                }
                            )
                            state["started"] = True
                        raw_arguments = state["arguments"]
                        try:
                            parsed_arguments = (
                                json.loads(raw_arguments) if raw_arguments else {}
                            )
                        except Exception as error:
                            yield format_sse(
                                {
                                    "type": "tool-input-error",
                                    "toolCallId": tool_call_id,
                                    "toolName": tool_name,
                                    "input": raw_arguments,
                                    "errorText": str(error),
                                }
                            )
                            continue
                        yield format_sse(
                            {
                                "type": "tool-input-available",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": parsed_arguments,
                            }
                        )
                        tool_function = tool_map.get(tool_name)
                        if tool_function is None:
                            yield format_sse(
                                {
                                    "type": "tool-output-error",
                                    "toolCallId": tool_call_id,
                                    "errorText": f"Tool '{tool_name}' not found.",
                                }
                            )
                            continue
                        try:
                            tool_result = tool_function(**parsed_arguments)
                        except Exception as error:
                            yield format_sse(
                                {
                                    "type": "tool-output-error",
                                    "toolCallId": tool_call_id,
                                    "errorText": str(error),
                                }
                            )
                        else:
                            yield format_sse(
                                {
                                    "type": "tool-output-available",
                                    "toolCallId": tool_call_id,
                                    "output": tool_result,
                                }
                            )
                    yield format_sse({"type": "finish-step"})
                    tool_calls_state.clear()
                    # Reset per-step state so the next step gets fresh IDs and clean flags
                    step_index += 1
                    text_stream_id = f"text-{step_index}"
                    reasoning_stream_id = f"reasoning-{step_index}"
                    text_started = False
                    text_finished = False
                    reasoning_started = False
                    step_started = False

        if reasoning_started:
            yield format_sse({"type": "reasoning-end", "id": reasoning_stream_id})
            reasoning_started = False

        if finish_reason == "stop" and text_started and not text_finished:
            yield format_sse({"type": "text-end", "id": text_stream_id})
            text_finished = True

        if finish_reason == "stop" and step_started:
            yield format_sse({"type": "finish-step"})

        if finish_reason == "tool_calls":
            for index in sorted(tool_calls_state.keys()):
                state = tool_calls_state[index]
                tool_call_id = state.get("id")
                tool_name = state.get("name")

                if tool_call_id is None or tool_name is None:
                    continue

                if not state["started"]:
                    yield format_sse(
                        {
                            "type": "tool-input-start",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                        }
                    )
                    state["started"] = True

                raw_arguments = state["arguments"]
                try:
                    parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
                except Exception as error:
                    yield format_sse(
                        {
                            "type": "tool-input-error",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": raw_arguments,
                            "errorText": str(error),
                        }
                    )
                    continue

                yield format_sse(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": parsed_arguments,
                    }
                )

                # NOTE: not supported in OpenAI protocol (need AI SDK protocol coming out of Kiln adapter for this)
                tool_function = {}.get(tool_name)
                if tool_function is None:
                    yield format_sse(
                        {
                            "type": "tool-output-error",
                            "toolCallId": tool_call_id,
                            "errorText": f"Tool '{tool_name}' not found.",
                        }
                    )
                    continue

                try:
                    tool_result = tool_function(**parsed_arguments)
                except Exception as error:
                    yield format_sse(
                        {
                            "type": "tool-output-error",
                            "toolCallId": tool_call_id,
                            "errorText": str(error),
                        }
                    )
                else:
                    yield format_sse(
                        {
                            "type": "tool-output-available",
                            "toolCallId": tool_call_id,
                            "output": tool_result,
                        }
                    )

        if text_started and not text_finished:
            yield format_sse({"type": "text-end", "id": text_stream_id})
            text_finished = True

        finish_metadata: Dict[str, Any] = {}
        if finish_reason is not None:
            finish_metadata["finishReason"] = finish_reason.replace("_", "-")

        if usage_data is not None:
            usage_payload = {
                "promptTokens": usage_data.prompt_tokens,
                "completionTokens": usage_data.completion_tokens,
            }
            total_tokens = getattr(usage_data, "total_tokens", None)
            if total_tokens is not None:
                usage_payload["totalTokens"] = total_tokens
            finish_metadata["usage"] = usage_payload

        if finish_metadata:
            yield format_sse({"type": "finish", "messageMetadata": finish_metadata})
        else:
            yield format_sse({"type": "finish"})

        yield "data: [DONE]\n\n"

        # after exhausting the stream, we get the full TaskRun object which contains the trace that
        # we can pass in on the next invoke to continue the conversation
        fake_storage.store_task_run(stream.task_run)


    except Exception:
        traceback.print_exc()
        raise


def patch_response_with_headers(
    response: StreamingResponse,
    protocol: str = "data",
) -> StreamingResponse:
    """Apply the standard streaming headers expected by the Vercel AI SDK."""

    response.headers["x-vercel-ai-ui-message-stream"] = "v1"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"

    if protocol:
        response.headers.setdefault("x-vercel-ai-protocol", protocol)

    return response



def get_stream_transport_func(
    protocol: StreamTransportProtocol,
) -> StreamTransportFunc:
    match protocol:
        case StreamTransportProtocol.OPENAI:
            return stream_text_openai
        case StreamTransportProtocol.AI_SDK:
            return stream_text_ai_sdk_transport
        case _:
            raise ValueError(f"Invalid protocol: {protocol}")
