"""Qwen3-VL evaluation subject powered by vLLM.

This subject loads a multi-modal Large Language Model through vLLM and
queries it with the generated video together with a textual prompt.  The
prompt can be produced from a user supplied template which receives the
per-video text file contents and a few metadata fields.  The model
response is then stored on each :class:`~video.VideoData` instance under
the ``"qwen_vl_vllm"`` key.

Example ``--model_args`` value for OpenAI-compatible API::

    '{
        "use_openai_api": true,
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model_name": "qwen3-vl",
        "prompt_template_path": "./templates/video_eval.txt",
        "json_template_path": "./templates/to_json.txt",
        "system_prompt": ""
    }'

The template receives the following fields::

    {video_text}         -> Contents of ``VideoData.text_path`` (empty when missing).
    {video_filename}     -> Basename of the video without extension.
    {video_path}         -> Absolute path to the video file.
    {audio_path}         -> Absolute path to the reference audio or empty string.
    {text_path}          -> Absolute path to the associated text file or empty string.

The json_template (optional) receives the first template's response::

    {response}           -> The text response from the first template evaluation.

If json_template is specified, the model will be called a second time to convert
the response to JSON format, which will be parsed and stored as structured data.

Additional placeholders can be supported by providing default values via
``template_variables`` in ``model_args``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

from tqdm import tqdm

from video import VideoData

# Try to import vLLM (optional for local inference)
try:  # pragma: no cover - optional dependency
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Try to import OpenAI client (optional for API usage)
try:  # pragma: no cover - optional dependency
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


DEFAULT_MODEL_NAME = "qwen3-vl"
DEFAULT_PLACEHOLDER = "{video_text}"


class _SafeDict(dict):
    """Dictionary that preserves unknown placeholders during ``format_map``."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return "{" + key + "}"


@dataclass
class TemplateRenderer:
    """Render prompts using ``str.format`` with sensible defaults."""

    template: str
    base_context: Mapping[str, Any]

    def render(self, extra_context: Mapping[str, Any]) -> str:
        context = _SafeDict(self.base_context)
        context.update(extra_context)
        return self.template.format_map(context)


def _load_template(
    model_args: Mapping[str, Any],
    template_key: str = "prompt_template",
    template_path_key: str = "prompt_template_path"
) -> TemplateRenderer:
    template_text: Optional[str] = model_args.get(template_key)
    template_path = model_args.get(template_path_key)
    if template_text and template_path:
        raise ValueError(
            f"Specify only one of '{template_key}' or '{template_path_key}'."
        )
    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(
                f"Prompt template file '{template_file}' does not exist."
            )
        template_text = template_file.read_text(encoding="utf-8")

    if template_text is None:
        template_text = DEFAULT_PLACEHOLDER

    base_variables = model_args.get("template_variables", {})
    if not isinstance(base_variables, Mapping):
        raise TypeError("'template_variables' must be a mapping of placeholder defaults.")

    return TemplateRenderer(template=template_text, base_context=base_variables)


def _prepare_messages(
    video_path: Path,
    prompt: str,
    system_prompt: Optional[str],
    include_video: bool
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            }
        )

    user_content: List[Dict[str, Any]] = []
    if include_video:
        user_content.append(
            {
                "type": "video_url",
                "video_url": {"url": f"file://{video_path}"},
            }
        )

    if prompt:
        user_content.append({"type": "text", "text": prompt})

    if not user_content:
        raise ValueError("At least one of video or prompt content must be provided.")

    messages.append({"role": "user", "content": user_content})
    return messages


def _build_sampling_params(options: Mapping[str, Any]) -> Dict[str, Any]:
    """Build sampling parameters for vLLM or return dict for OpenAI API."""
    if options and not isinstance(options, Mapping):
        raise TypeError("'sampling_options' must be a mapping of SamplingParams arguments.")

    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": 512,
        "temperature": 0.0,
    }
    if options:
        sampling_kwargs.update(options)

    # Return dict for flexibility (can be used for both vLLM and OpenAI API)
    return sampling_kwargs


def _collect_llm_kwargs(model_args: Mapping[str, Any]) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = dict(model_args.get("engine_options", {}))
    if not isinstance(llm_kwargs, MutableMapping):
        raise TypeError("'engine_options' must be a mapping of keyword arguments.")

    for key in ("tensor_parallel_size", "dtype", "gpu_memory_utilization", "max_model_len", "quantization", "revision", "enforce_eager"):
        if key in model_args and key not in llm_kwargs:
            llm_kwargs[key] = model_args[key]

    if "trust_remote_code" not in llm_kwargs:
        llm_kwargs["trust_remote_code"] = model_args.get("trust_remote_code", True)

    return llm_kwargs


def _encode_video_base64(video_path: Path) -> str:
    """Encode video file to base64 string."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def _prepare_openai_messages(
    video_path: Path,
    prompt: str,
    system_prompt: Optional[str],
    include_video: bool
) -> List[Dict[str, Any]]:
    """Prepare messages for OpenAI-compatible API."""
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message content
    if include_video:
        # For OpenAI API, we use base64 encoding for video
        video_base64 = _encode_video_base64(video_path)
        user_content = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}
            }
        ]
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
    else:
        # Text-only message
        if prompt:
            messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("At least one of video or prompt content must be provided.")

    return messages


def _call_openai_api(
    client: Any,
    messages: List[Dict[str, Any]],
    model_name: str,
    sampling_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Call OpenAI-compatible API and return response."""
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **sampling_params
    )
    
    return {
        "text": response.choices[0].message.content if response.choices else "",
        "finish_reason": response.choices[0].finish_reason if response.choices else None,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        }
    }



def evaluate(
    data_list: Iterable[VideoData],
    device: str = "cuda",
    batch_size: int = 1,
    sampling: int = 16,
    model_args: Optional[Dict[str, Any]] = None,
) -> List[VideoData]:
    """Run the Qwen3-VL model via vLLM on each video entry."""

    del device, batch_size, sampling  # Parameters kept for API compatibility.

    model_args = model_args or {}
    use_openai_api = model_args.get("use_openai_api", False)
    model_name = model_args.get("model_name", DEFAULT_MODEL_NAME)
    include_video = model_args.get("include_video", True)
    system_prompt = model_args.get("system_prompt")
    keep_full_messages = model_args.get("store_messages", False)

    template = _load_template(model_args)
    
    # Check if json_template is specified
    json_template: Optional[TemplateRenderer] = None
    if model_args.get("json_template") or model_args.get("json_template_path"):
        json_template = _load_template(
            model_args,
            template_key="json_template",
            template_path_key="json_template_path"
        )
    
    sampling_params = _build_sampling_params(model_args.get("sampling_options", {}))
    
    # Initialize the appropriate client
    llm = None
    openai_client = None
    
    if use_openai_api:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for OpenAI API usage. "
                "Please install it with `pip install openai`."
            )
        
        api_base = model_args.get("api_base", "http://localhost:8000/v1")
        api_key = model_args.get("api_key", "EMPTY")
        
        openai_client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
    else:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "The 'vllm' package is required for local inference. "
                "Please install it with `pip install vllm` or set 'use_openai_api': true "
                "to use an OpenAI-compatible API instead."
            )
        
        llm_kwargs = _collect_llm_kwargs(model_args)
        llm = LLM(model=model_name, allowed_local_media_path="/", **llm_kwargs)

    results: List[VideoData] = []
    desc = "Evaluating Qwen3-VL via OpenAI API" if use_openai_api else "Evaluating Qwen3-VL via vLLM"
    
    for data in tqdm(data_list, desc=desc):
        video_path = Path(data.video_path).resolve()
        text_content = data.get_text()
        prompt = template.render(
            {
                "video_text": text_content,
                "video_filename": data.video_filename,
                "video_path": str(video_path),
                "audio_path": str(Path(data.audio_path).resolve()) if data.audio_path else "",
                "text_path": str(Path(data.text_path).resolve()) if data.text_path else "",
            }
        ).strip()

        # Call the appropriate backend
        if use_openai_api:
            messages = _prepare_openai_messages(video_path, prompt, system_prompt, include_video)
            response = _call_openai_api(openai_client, messages, model_name, sampling_params)
            responses = [response]
        else:
            messages = _prepare_messages(video_path, prompt, system_prompt, include_video)
            raw_output = llm.chat(messages=messages, sampling_params=SamplingParams(**sampling_params))
            responses = [
                {
                    "text": output.outputs[0].text if output.outputs else "",
                    "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                    "usage": {
                        "prompt_tokens": getattr(output, "prompt_token_ids", None),
                        "completion_tokens": getattr(output.outputs[0], "token_ids", None)
                        if output.outputs
                        else None,
                    },
                }
                for output in raw_output
            ]

        if not responses:
            raise RuntimeError("Model returned no outputs for the given request.")

        primary_response = responses[0]["text"]
        record: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "response": primary_response,
        }

        if keep_full_messages:
            record["messages"] = messages
            record["raw_outputs"] = responses

        # If json_template is specified, run a second query to convert to JSON
        if json_template is not None:
            json_prompt = json_template.render({"response": primary_response}).strip()
            
            # Call the appropriate backend for JSON conversion
            if use_openai_api:
                json_messages = _prepare_openai_messages(
                    video_path, json_prompt, system_prompt, include_video=False
                )
                json_response = _call_openai_api(openai_client, messages, model_name, sampling_params)
                json_responses = [json_response]
            else:
                json_messages = _prepare_messages(
                    video_path, json_prompt, system_prompt, include_video=False
                )
                json_raw_output = llm.chat(messages=json_messages, sampling_params=SamplingParams(**sampling_params))
                json_responses = [
                    {
                        "text": output.outputs[0].text if output.outputs else "",
                        "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                    }
                    for output in json_raw_output
                ]
            
            if not json_responses:
                raise RuntimeError("Model returned no outputs for JSON conversion request.")
            
            json_response_text = json_responses[0]["text"]
            print("JSON Response Text:", json_response_text)
            # Try to parse the JSON response
            try:
                # 分行
                json_response_lines = json_response_text.splitlines()
                # 如果有某一行是```开头，就认为刚开始不处于json内容，否则默认处于json内
                in_json = any(True for line in json_response_lines if line.strip().startswith("```"))
                json_content_lines = []

                for line in json_response_lines:
                    if in_json:
                        if line.strip().startswith("```"):
                            in_json = False
                        else:
                            json_content_lines.append(line)
                    elif line.strip().startswith("```"):
                        in_json = True
                json_response_text = "\n".join(json_content_lines).strip()
                parsed_json = json.loads(json_response_text)
                record["json_prompt"] = json_prompt
                record["json_response_text"] = json_response_text
                record["parsed_result"] = parsed_json
            except json.JSONDecodeError as e:
                # If JSON parsing fails, store the error and raw text
                record["json_prompt"] = json_prompt
                record["json_response_text"] = json_response_text
                record["json_parse_error"] = str(e)
            
            if keep_full_messages:
                record["json_messages"] = json_messages
                record["json_raw_outputs"] = json_responses

        data.register_result("qwen_vl_vllm", record)
        results.append(data)

    return results
