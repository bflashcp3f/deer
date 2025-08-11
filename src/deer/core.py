from deer.language_models.openai_model import OpenAIModel
from deer.language_models.togetherai_model import TogetherAIModel
from deer.language_models.vllm_model import VllmModel


def run_inference(task, args):
    """
    Perform inference using the specified language model.

    Note:
        - This function uses `task` and `args` for flexibility in the early stages of development.
        - In the future, it may be refactored to use explicit arguments for improved clarity and maintainability.

    Args:
        task: The task object containing data information.
        args: Parsed command-line arguments with configuration details.

    Returns:
        list: Model predictions.
    """
    # Extract necessary attributes
    model_type = args.model_type
    model_name = args.model_name
    
    # Generate and get the prompts for inference
    task.load_retriever()
    task.generate_icl_prompts()
    prompt_list = task.get_icl_prompt()

    # Initialize the language model
    if model_type == "openai":
        model = OpenAIModel(model_name, async_mode=args.async_mode)
    elif model_type == "togetherai":
        model = TogetherAIModel(model_name, async_mode=args.async_mode)
    elif model_type == "vllm":
        model = VllmModel(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Perform inference
    response_list = model.generate(
        prompt_list=prompt_list,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        batch_size=args.batch_size,
        response_format=args.response_format,
        logprobs=args.logprobs,
        sleep_time=args.sleep_time
    )
    
    # Update inference results
    task.update_icl_results(response_list)

    # Save the task output
    task.save_results()

def run_reflection(task, args):
    """
    Perform reflection using the specified language model.

    Note:
        - This function uses `task` and `args` for flexibility in the early stages of development.
        - In the future, it may be refactored to use explicit arguments for improved clarity and maintainability.

    Args:
        task: The task object containing data information.
        args: Parsed command-line arguments with configuration details.

    Returns:
        list: Model predictions.
    """
    # Extract necessary attributes
    model_type = args.model_type
    model_name = args.model_name
    output_dir = args.output_dir
    
    # Generate and get the prompts for reflection
    task.load_retriever()
    task.generate_reflection_data()
    task.generate_reflection_prompts()
    prompt_list = task.get_reflection_prompt()

    # Initialize the language model
    if model_type == "openai":
        model = OpenAIModel(model_name)
    elif model_type == "togetherai":
        model = TogetherAIModel(model_name)
    elif model_type == "vllm":
        model = VllmModel(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Perform reflection
    response_list = model.generate(
        prompt_list=prompt_list,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        batch_size=args.batch_size,
        response_format=args.response_format,
        logprobs=args.logprobs,
        sleep_time=args.sleep_time
    )
    
    # Update reflection results
    task.update_reflection_results(response_list)

    # Save the task output
    task.save_results()