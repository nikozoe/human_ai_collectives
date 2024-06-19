import boto3
import google.auth.transport.requests
import json
import re
import requests
import openai
from anthropic import Anthropic
from google.oauth2 import service_account
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import api_keys
from machine_model import MachineModel

openai.organization = api_keys.OPENAI_ORGANIZATION
openai.api_key = api_keys.OPENAI_API_KEY


ANTHROPIC_SDK_CLIENT = Anthropic(api_key=api_keys.ANTHROPIC_API_KEY)

MISTRAL_CLIENT = MistralClient(api_key=api_keys.MISTRAL_API_KEY)

RESPONSE_PRETEXT_BLOCKLIST = [
    "Sure, ",
    "Here is the ",
    "Here are ",
    "### Response:",
    "The probable",
    "The differential",
    "The most probable",
]

OPENAI_MODEL = "gpt-4-1106-preview"
GOOGLE_MODEL = "gemini-1.0-pro"
ANTHROPIC_MODEL = "claude-3-opus-20240229"
MISTRAL_MODEL = "mistral-large-latest"

credentials = service_account.Credentials.from_service_account_file(
    api_keys.GOOGLE_SERVICE_CREDENTIALS_PATH,
    scopes = ["https://www.googleapis.com/auth/cloud-platform"],
)
request = google.auth.transport.requests.Request()

GOOGLE_API_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/"
    f"{api_keys.GOOGLE_PROJECT_ID}/locations/us-central1/publishers/google/"
    f"models/{GOOGLE_MODEL}:streamGenerateContent"
)

def get_open_ai_solve(case_text, messages_override=None):
    model = OPENAI_MODEL
    max_tokens = 128
    temperature = 0
    presence_penalty = 0
    frequency_penalty = 0
    messages = [
        {
            "role": "system",
            "content": (
                "You are a physician using common shorthand non-abbreviated diagnoses "
                "providing the shortest differentials (max 5) without explanations, "
                "maximizing likelihood of the right answer, but minimizing cost (each "
                "answer doubles cost). Remove list numbering, and respond with each "
                "answer on a new line."
            ),
        }
    ] + [
        {
            "role": "user",
            "content": f"{case_text}\n\nWhat is the differential diagnosis?",
        }
    ]
    messages = messages_override or messages
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    choices = completion.choices
    raw_resp = choices[0].message.content
    final_dxs = _clean_response(response=raw_resp)
    return MachineModel(
        org="OpenAI",
        name=model,
        parameters={
            "prompt": messages,
            "max_token": max_tokens,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
    ), raw_resp, final_dxs


def get_google_ai_rest_solve(case_text, prompt_override=None):
    model = GOOGLE_MODEL
    max_tokens = 128
    temperature = 0
    top_k = 1
    top_p = 0

    # Update credentials for private access models due to token expiry
    credentials.refresh(request)

    prompt = f"""{case_text}\n\nWhat is the differential (list format of common \
    shorthand non-abbreviated diagnoses) for the above case? Respond with ONLY \
    diagnosis names (one per line), up to a max of 5."""
    prompt = prompt_override or prompt

    payload = {
        "contents": [
            {
            "role": "user",
            "parts": [
                {
                    "text": prompt,
                }
            ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens,
        }
    }
    headers = {
        "Authorization": "Bearer " + credentials.token
    }

    response = requests.post(GOOGLE_API_URL, json=payload, headers=headers)
    candidates = [
        candidate.get("candidates", [])
        for candidate in response.json()
    ]
    content_text = []
    for candidate in candidates:
        if len(candidate) > 0:
            content_text.append(
                candidate[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            )
    raw_resp = "".join(content_text)
    final_dxs = _clean_response(response=raw_resp)
    return MachineModel(
        org="GoogleAI",
        name=model,
        parameters={
            "prompt": prompt,
            "max_token": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    ), raw_resp, final_dxs


def get_meta_llama_solve(case_text, messages_override=None):
    model = "llama-2-70b-f"
    max_tokens = 128
    temperature = 0.01
    top_p = 0.01
    prompt = [[
        {
            "role": "system",
            "content": (
                "You are a physician using common shorthand non-abbreviated diagnoses "
                "providing a differential (length 5) without explanations, "
                "maximizing likelihood of the right answer. Remove list numbering, any summaries, and "
                "respond with each answer on a new line."
            ),
        },
        {
            "role": "user",
            "content": f"{case_text}\n\nWhat is the differential diagnosis?",
        }
    ]]
    prompt = messages_override or prompt
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "return_full_text": False
        },
    }
    endpoint_name = f"jumpstart-dft-meta-textgeneration-{model}"
    region_name = "us-west-2"
    client = boto3.client("sagemaker-runtime", region_name=region_name)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    choices = json.loads(response)
    raw_resp = choices[0].get("generation").get("content") if choices else ""
    final_dxs = _clean_response(response=raw_resp)
    return MachineModel(
        org="MetaAI",
        name=model,
        parameters={
            "prompt": prompt,
            "max_token": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    ), raw_resp, final_dxs


def get_anthropic_ai_solve(case_text, messages_override=None):
    model = ANTHROPIC_MODEL
    max_tokens = 128
    temperature = 0
    top_k = 1
    top_p = 0
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a physician using common shorthand non-abbreviated diagnoses "
                "providing the shortest differentials (max 5) without explanations, "
                "maximizing likelihood of the right answer, but minimizing cost (each "
                "answer doubles cost). Remove list numbering, and respond with each "
                "answer on a new line."
            ),
        }
    ] + [
        {
            "role": "user",
            "content": f"{case_text}\n\nWhat is the differential diagnosis?",
        }
    ]
    prompt = messages_override or messages
    # Anthropic separates system prompt from user messages, so we need to split
    # the system prompt from user prompt
    system = prompt[0].get("content")
    messages = prompt[1:]

    response = ANTHROPIC_SDK_CLIENT.messages.create(
        model = model,
        messages = messages,
        system = system,
        **params,
    )
    content = response.content
    raw_resp = content[0].text
    final_dxs = _clean_response(response=raw_resp)
    return MachineModel(
        org="AnthropicAI",
        name=model,
        parameters={
            "prompt": prompt,
            "max_token": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    ), raw_resp, final_dxs


def get_mistral_ai_solve(case_text, messages_override=None):
    model = MISTRAL_MODEL
    max_tokens = 128
    temperature = 0
    messages = [
        {
            "role": "system",
            "content": (
                "You are a physician using common shorthand non-abbreviated diagnoses "
                "providing the shortest differentials (max 5) without explanations, "
                "maximizing likelihood of the right answer, but minimizing cost (each "
                "answer doubles cost). Remove list numbering, and respond with each "
                "answer on a new line."
            ),
        }
    ] + [
        {
            "role": "user",
            "content": f"{case_text}\n\nWhat is the differential diagnosis?",
        }
    ]
    messages_json = messages_override or messages

    # Tranform JSON to `ChatMessage`
    messages = [
        ChatMessage(role=message.get("role"), content=message.get("content"))
        for message in messages_json
    ]

    response = MISTRAL_CLIENT.chat(
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = max_tokens,
    )

    choices = response.choices
    raw_resp = choices[0].message.content
    final_dxs = _clean_response(response=raw_resp)
    return MachineModel(
        org="MistralAI",
        name=model,
        parameters={
            "prompt": messages,
            "max_token": max_tokens,
            "temperature": temperature,
        },
    ), raw_resp, final_dxs


def _clean_response(response):
    response = response.strip()
    for blocklist_item in RESPONSE_PRETEXT_BLOCKLIST:
        if response.startswith(blocklist_item):
            response = "\n".join(response.split("\n")[1:])
            break

    responses = [
        re.sub("^(?:-|\*|•|\d+\.)", "", dx.strip()).strip()
        for dx in response.split("\n")
    ]

    responses = [re.sub("(\(SNOMED CT:.+)\)", "", dx.strip()).strip() for dx in responses]
    responses = [re.sub("(\(\d+)\)", "", dx.strip()).strip() for dx in responses]
    responses = [re.sub("(\([A-Z]\d+.+)\)", "", dx.strip()).strip() for dx in responses]
    responses = [re.sub("(- \d\d\d+)", "", dx.strip()).strip() for dx in responses]
    responses = [res for res in responses if res!='']
    responses = [res.split(':')[0] if len(res.split())>6 else res for res in responses]
    responses = [res.split(' - ')[0] if len(res.split())>6 else res for res in responses]


    return responses
