import time
import asyncio
import os
from tqdm import tqdm
from typing import Any
from together import Together, AsyncTogether
from deer.base import LanguageModel, LMOutput


class TogetherAIModel(LanguageModel):
    """
    A wrapper for TogetherAI's language model API.
    """
    def __init__(self, model_name, async_mode=True):
        """
        Initialize the TogetherAI model.

        Args:
            model_name (str): The name of the TogetherAI model (e.g., "gpt-4o-mini-2024-07-18").
            async_mode (bool): Whether to use asynchronous mode for API requests. Default is True.
        """
        super().__init__(model_name)
        self.async_mode = async_mode

        # Initialize the TogetherAI API client
        if async_mode:
            self.client = AsyncTogether(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )
        else:
            self.client = Together(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )

    async def dispatch_chatcompletion_requests(
        self,
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: str,
        response_format: str,
        logprobs: bool,
    ) -> list[LMOutput]:
        """
        Dispatches requests to TogetherAI API asynchronously.

        Args:
            messages_list (list[list[dict]]): List of message dictionaries for each prompt.
            model (str): Model name.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens.
            top_p (float): Nucleus sampling probability.
            stop (str): Stop sequence.
            response_format (str): Format of the response.
            logprobs (bool): Whether to return log probabilities.

        Returns:
            list[LMOutput]: Asynchronous responses from the TogetherAI API.
        """
        async_responses = [
            self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                response_format={"type": response_format} if response_format else None,
                logprobs=logprobs,
            )
            for messages in messages_list
        ]
        return await asyncio.gather(*async_responses)

    def generate(
        self,
        prompt_list,
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        stop=None,
        batch_size=1,
        response_format=None,
        logprobs=False,
        sleep_time=0,
    ):
        """
        Generate responses for a list of prompts.

        Args:
            prompt_list (list): A list of prompts to generate responses for.
            temperature (float): Sampling temperature. Higher values mean more randomness.
            top_p (float): Nucleus sampling probability. Limits to the smallest set of tokens.
            max_tokens (int): Maximum number of tokens in the output.
            stop (list): A list of stop sequences to end generation.
            batch_size (int): Number of prompts to process in each API call.
            response_format (dict or str): Specifies the format of the model's output.
            logprobs (bool): Whether to return token log probabilities.
            sleep_time (int): Delay (in seconds) between batch requests to respect rate limits.

        Returns:
            list: Generated responses for each prompt in `prompt_list`.
        """
        responses = []

        # Split prompts into batches
        for i in tqdm(range(0, len(prompt_list), batch_size), desc=f"Processing Batches (sleep time: {sleep_time}s)"):
            batch_prompts = prompt_list[i : i + batch_size]
            messages_list = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]

            if self.async_mode:
                # Async Mode
                try:
                    response_list_batch = asyncio.run(
                        self.dispatch_chatcompletion_requests(
                            messages_list=messages_list,
                            model=self.model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            stop=stop,
                            response_format=None,
                            logprobs=logprobs,
                        )
                    )
                    batch_responses = [
                        LMOutput(
                            text=response.choices[0].message.content,
                            logprobs=response.choices[0].logprobs,
                        )
                        for response in response_list_batch
                    ]
                except Exception as e:
                    raise RuntimeError(f"TogetherAI API async request failed: {e}")
            else:
                # Sync Mode
                try:
                    batch_responses = []
                    for messages in messages_list:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            stop=stop,
                            # response_format={"type": response_format} if response_format else None,
                            logprobs=logprobs,
                        )
                        batch_responses.append(
                            LMOutput(
                                text=response.choices[0].message.content,
                                log_prob=response.choices[0].logprobs,
                            )
                        )
                except Exception as e:
                    raise RuntimeError(f"TogetherAI API sync request failed: {e}")

            responses.extend(batch_responses)
            time.sleep(sleep_time)

        return responses