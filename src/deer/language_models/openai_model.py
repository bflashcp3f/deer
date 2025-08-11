import time
import asyncio

from typing import Any
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from deer.base import LanguageModel, LMOutput


class OpenAIModel(LanguageModel):
    """
    A wrapper for OpenAI's language model API.
    """
    def __init__(self, model_name, async_mode=True):
        """
        Initialize the OpenAI model.

        Args:
            model_name (str): The name of the OpenAI model (e.g., "gpt-4o-mini-2024-07-18").
            async_mode (bool): Whether to use asynchronous mode for API requests. Default is True.
        """
        super().__init__(model_name)
        self.async_mode = async_mode

        # Initialize the OpenAI API client
        if async_mode:
            self.client = AsyncOpenAI()
        else:
            self.client = OpenAI()

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
        max_retries: int = 3,
    ) -> list[LMOutput]:
        """
        Dispatches requests to OpenAI API asynchronously with retry logic.

        Args:
            messages_list (list[list[dict]]): List of message dictionaries for each prompt.
            model (str): Model name.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens.
            top_p (float): Nucleus sampling probability.
            stop (str): Stop sequence.
            response_format (str): Format of the response.
            logprobs (bool): Whether to return log probabilities.
            max_retries (int): Maximum number of retries for failed requests.

        Returns:
            list[LMOutput]: Asynchronous responses from the OpenAI API.
        """
        async def make_request_with_retry(messages, retry_count=0):
            try:
                return await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                    response_format={"type": response_format} if response_format else None,
                    logprobs=logprobs,
                )
            except Exception as e:
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Request failed, retrying in {wait_time}s... (attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    return await make_request_with_retry(messages, retry_count + 1)
                else:
                    print(f"Request failed after {max_retries} retries: {e}")
                    raise
        
        async_responses = [
            make_request_with_retry(messages) for messages in messages_list
        ]
        return await asyncio.gather(*async_responses, return_exceptions=False)

    async def _process_all_batches_async(
        self,
        prompt_list,
        temperature,
        top_p,
        max_tokens,
        stop,
        batch_size,
        response_format,
        logprobs,
        sleep_time,
    ):
        """
        Process all batches asynchronously in a single event loop.
        """
        responses = []
        
        for i in tqdm(range(0, len(prompt_list), batch_size), desc=f"Processing Batches (sleep time: {sleep_time}s)"):
            batch_prompts = prompt_list[i : i + batch_size]
            messages_list = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]
            
            try:
                response_list_batch = await self.dispatch_chatcompletion_requests(
                    messages_list=messages_list,
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                    response_format=response_format,
                    logprobs=logprobs,
                )
                batch_responses = [
                    LMOutput(
                        text=response.choices[0].message.content,
                        logprobs=response.choices[0].logprobs,
                    )
                    for response in response_list_batch
                ]
                responses.extend(batch_responses)
                
                # Sleep between batches if needed
                if sleep_time > 0 and i + batch_size < len(prompt_list):
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # You can choose to retry or handle the error differently
                raise RuntimeError(f"OpenAI API async request failed: {e}")
        
        return responses

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
        if self.async_mode:
            # Process all batches in a single event loop
            try:
                responses = asyncio.run(
                    self._process_all_batches_async(
                        prompt_list=prompt_list,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        batch_size=batch_size,
                        response_format=response_format,
                        logprobs=logprobs,
                        sleep_time=sleep_time,
                    )
                )
                return responses  # Success - return immediately, don't run sync mode
            except Exception as e:
                print(f"Async mode failed: {e}. Falling back to sync mode for this request.")
                # Don't permanently change self.async_mode, just fall through for this request
        
        # Sync mode processing (only reached if async_mode is False or async failed)
        responses = []
        for i in tqdm(range(0, len(prompt_list), batch_size), desc=f"Processing Batches (sleep time: {sleep_time}s)"):
            
            batch_prompts = prompt_list[i : i + batch_size]
            messages_list = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]

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
                        response_format={"type": response_format}
                        if response_format
                        else None,
                        logprobs=logprobs,
                    )
                    batch_responses.append(
                        LMOutput(
                            text=response.choices[0].message.content,
                            logprobs=response.choices[0].logprobs,
                        )
                    )
            except Exception as e:
                raise RuntimeError(f"OpenAI API sync request failed: {e}")

            responses.extend(batch_responses)
            
            # Sleep between batches if needed
            if sleep_time > 0 and i + batch_size < len(prompt_list):
                time.sleep(sleep_time)

        return responses