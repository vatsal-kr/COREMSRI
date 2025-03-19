import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import openai
import pandas as pd
from tenacity import (
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)
from vllm import SamplingParams

DEFAULT_SYSTEM_MESSAGE = ""
COST_DICT = {
    "gpt-4o-2024-11-20": (2.5e-6, 10e-6),
    "gpt-4o": (2.5e-6, 10e-6),
    "gpt-4o-mini": (0.15e-6, 0.6e-6),
    "o1-mini": (1.1e-6, 4.4e-6),
    "o1": (15e-6, 60e-6),
    "o3-mini": (1.1e-6, 4.4e-6),
}


class OpenAICost:
    def __init__(self, prompt_tokens, response_tokens, model_name):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.model_name = model_name
        self.cost = COST_DICT[model_name][0] * self.prompt_tokens + COST_DICT[model_name][1] * self.response_tokens


class OpenAIResponse(NamedTuple):
    text: str
    finish_reason: str
    success: bool
    exception: Exception | None


class OpenAIHandler:
    def __init__(self, config):
        self.system_message = config["system_message"] if "system_message" in config else DEFAULT_SYSTEM_MESSAGE

        self.api_key = config["api_key"] if "api_key" in config else os.environ["OPENAI_API_KEY"]
        self.api_version = config["api_version"] if "api_version" in config else os.environ["OPENAI_API_VERSION"]
        self.api_base = config["api_base"] if "api_base" in config else os.environ["OPENAI_API_BASE"]
        self.api_type = config["api_type"] if "api_type" in config else os.environ["OPENAI_API_TYPE"]

        self.n = config["n"] if "n" in config else 1
        self.model = config["model"] if "model" in config else "gpt-4-32k"
        self.temperature = config["temperature"] if "temperature" in config else 0.0
        self.seed = config["seed"] if "seed" in config else 42

        self.max_retry_attempts = config["max_retry_attempts"] if "max_retry_attempts" in config else 8
        self.retry_timeout = config["retry_timeout"] if "retry_timeout" in config else 60
        self.retry_when_blank = config["retry_when_blank"] if "retry_when_blank" in config else False
        self.fixed_retry_interval = config["fixed_retry_interval"] if "fixed_retry_interval" in config else 10
        self.retry_interval = config["retry_interval"] if "retry_interval" in config else 1
        self.max_retry_interval = config["max_retry_interval"] if "max_retry_interval" in config else 60
        self.stop = config["stop"] if "stop" in config else None
        self.max_tokens = config["max_tokens"] if "max_tokens" in config else 4000

        if "exponential_backoff" in config and config["exponential_backoff"]:
            self.wait_config = wait_exponential(
                multiplier=2,
                min=self.retry_interval,
                max=self.max_retry_interval,
            )
        else:
            self.wait_config = wait_fixed(self.fixed_retry_interval)

    def get_responses(self, prompts):
        responses = []
        costs = []
        for prompt in prompts:
            try:
                for attempt in Retrying(
                    retry=retry_if_exception_type(openai.RateLimitError)
                    | retry_if_exception_type(openai.APIError)
                    | retry_if_exception_type(openai.OpenAIError)
                    | retry_if_result(lambda result: self.retry_when_blank and any([r.message.content == "" for r in result.choices])),
                    stop=stop_after_attempt(self.max_retry_attempts) | stop_after_delay(self.retry_timeout),
                    wait=self.wait_config,
                ):
                    with attempt:
                        try:
                            logging.info(f"Querying for prompt. Attempt {attempt.retry_state.attempt_number}")
                            if self.api_type == "azure":
                                client = openai.AzureOpenAI(
                                    api_key=self.api_key,
                                    azure_endpoint=self.api_base,
                                    api_version=self.api_version,
                                )
                            else:
                                client = openai.OpenAI(
                                    api_key=self.api_key,
                                    base_url=self.api_base,
                                )
                            response = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": self.system_message},
                                    {"role": "user", "content": prompt},
                                ],
                                model=self.model,
                                n=self.n,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                seed=self.seed,
                            )
                            responses.append(
                                [
                                    OpenAIResponse(
                                        text=choice.message.content,
                                        finish_reason=choice.finish_reason,
                                        success=True,
                                        exception=None,
                                    )
                                    for choice in response.choices
                                ]
                            )
                            costs.append(
                                OpenAICost(
                                    prompt_tokens=response.usage.prompt_tokens,
                                    response_tokens=response.usage.completion_tokens,
                                    model_name=self.model,
                                )
                            )
                        except Exception as e:
                            logging.warning(traceback.format_exc())
                            logging.warning(f"Error while fetching prompt (attempt {attempt.retry_state.attempt_number}) : {e}")
                            continue
            except Exception as e:
                logging.warning(f"Error while querying for prompt (MAX ATTEMPTS REACHED) : {e}")
                responses.append(
                    [
                        OpenAIResponse(
                            text="",
                            finish_reason="",
                            success=False,
                            exception=e,
                        )
                    ]
                )
                continue
        df = pd.DataFrame(
            {
                "model": [self.model],
                "cost": [sum([x.cost for x in costs])],
                "prompt_tokens": [sum([x.prompt_tokens for x in costs])],
                "response_tokens": [sum([x.response_tokens for x in costs])],
                "timestamp": [datetime.now().strftime("%Y-%m-%d")],
            }
        )
        save_path = Path(__file__).parent / "openai_costs.csv"
        df.to_csv(
            save_path,
            mode="a",
            header=not save_path.exists(),
            index=False,
        )
        return responses


class LLMHandler:
    def __init__(self, config, llm):
        self.system_message = config["system_message"] if "system_message" in config else DEFAULT_SYSTEM_MESSAGE

        self.n = config["n"] if "n" in config else 1
        self.model = config["model"] if "model" in config else "gpt-4-32k"
        self.temperature = config["temperature"] if "temperature" in config else 0.0
        self.seed = config["seed"] if "seed" in config else 42

        self.stop = config["stop"] if "stop" in config else None
        self.max_tokens = config["max_tokens"] if "max_tokens" in config else 4000

        self.llm = llm
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop,
            n=self.n,
            seed=self.seed,
        )

    def get_responses(self, prompts):
        messages = [
            [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        responses = self.llm.chat(messages, self.sampling_params)
        responses = [
            [
                OpenAIResponse(
                    text=response_n.text,
                    finish_reason=response_n.finish_reason,
                    success=True,
                    exception=None,
                )
                for response_n in response_prompt.outputs
            ]
            for response_prompt in responses
        ]
        return responses
