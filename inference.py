import os
from typing import Any

from openai import OpenAI
import orjson
from pydantic import ValidationError

from gbill_env import GbillAction, GbillEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

client: OpenAI | None = None


def get_client() -> OpenAI:
    global client
    if client is None:
        if not API_KEY:
            raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py")
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["approve", "reject", "finish"],
        },
        "invoice_id": {
            "type": ["string", "null"],
            "description": "Invoice ID for approve/reject. Leave null or omit for finish.",
        },
    },
    "required": ["action_type"],
    "additionalProperties": False,
}

ALLOWED_ACTION_FIELDS = set(GbillAction.model_fields)


def json_dumps(value: Any, *, indent: bool = False) -> str:
    option = orjson.OPT_INDENT_2 if indent else 0
    return orjson.dumps(value, option=option).decode()


def serialize_observation(observation: Any) -> dict[str, Any]:
    return {
        "invoices": [invoice.model_dump() for invoice in observation.invoices],
        "account_balance": observation.account_balance,
        "task_description": observation.task_description,
        "last_feedback": observation.last_feedback,
    }


def build_prompt(observation: dict[str, Any]) -> str:
    return (
        "You are an AI billing agent.\n"
        f"Task: {observation['task_description']}\n"
        f"Current balance: {observation['account_balance']}\n"
        f"Last feedback: {observation['last_feedback']}\n"
        f"Pending invoices: {json_dumps(observation['invoices'], indent=True)}\n\n"
        "Respond with one JSON object only.\n"
        "Choose exactly one next action.\n"
        "Use approve/reject only for pending invoices.\n"
        "Use finish only when you are ready to be graded.\n"
        "Do not include explanations, balances, confidence scores, or any keys other than "
        "action_type, invoice_id, and metadata."
    )


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    data = orjson.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


def normalize_action(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)

    if "action_type" not in normalized:
        for alias in ("action", "decision", "type"):
            if alias in normalized:
                normalized["action_type"] = normalized.pop(alias)
                break

    action_type = normalized.get("action_type")
    if isinstance(action_type, str):
        normalized["action_type"] = action_type.strip().lower()

    if normalized.get("action_type") == "finish":
        normalized.pop("invoice_id", None)

    return {
        key: value
        for key, value in normalized.items()
        if key in ALLOWED_ACTION_FIELDS
    }


def choose_action(observation: dict[str, Any]) -> GbillAction:
    prompt = build_prompt(observation)
    client = get_client()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "gbill_action",
                    "strict": True,
                    "schema": ACTION_SCHEMA,
                }
            },
        )
        raw_action = parse_json_content(response.output_text)
    except Exception:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        raw_action = parse_json_content(content)

    normalized_action = normalize_action(raw_action)
    try:
        return GbillAction.model_validate(normalized_action)
    except ValidationError as exc:
        raise ValueError(
            f"Model returned an invalid action. Raw={raw_action!r} Normalized={normalized_action!r}"
        ) from exc


def run_task(task_level: str) -> float:
    print(f"\n--- Starting task: {task_level.upper()} ---")

    with GbillEnv(base_url=ENV_URL).sync() as env:
        result = env.reset(task_level=task_level)
        observation = result.observation

        for step_idx in range(1, MAX_STEPS + 1):
            action = choose_action(serialize_observation(observation))
            print(f"Step {step_idx}: agent chose -> {action.model_dump()}")

            result = env.step(action)
            observation = result.observation
            print(f"Feedback: {observation.last_feedback}")

            if result.done:
                final_score = result.reward or 0.0
                print(f"Task complete. Final score: {final_score:.2f}")
                return final_score

        final_result = env.step(GbillAction(action_type="finish"))
        final_score = final_result.reward or 0.0
        print(f"Max steps reached. Forced finish score: {final_score:.2f}")
        return final_score


if __name__ == "__main__":
    scores = {level: run_task(level) for level in ("easy", "medium", "hard")}
    print("\n--- Summary ---")
    for level, score in scores.items():
        print(f"{level}: {score:.2f}")
