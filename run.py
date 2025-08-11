import argparse
import sys
from deer.datasets.ncbi import NCBI
from deer.datasets.conll03 import CONLL03
from deer.datasets.bc2gm import BC2GM
from deer.datasets.ontonotes import OntoNotes
from deer.datasets.tweetner7 import TweetNER7
from deer.core import run_inference, run_reflection


def main(args):
    """
    Main entry point for the pipeline.
    """
    # Initialize the task
    if args.data_name == "ncbi":
        task = NCBI(args)
    elif args.data_name == "conll03":
        task = CONLL03(args)
    elif args.data_name == "bc2gm":
        task = BC2GM(args)
    elif args.data_name == "ontonotes":
        task = OntoNotes(args)
    elif args.data_name == "tweetner7":
        task = TweetNER7(args)
    else:
        raise ValueError(f"Invalid dataset name: {args.data_name}. Supported datasets: ncbi, conll03, bc2gm, ontonotes, tweetner7")

    # Run inference if specified
    if args.icl_inference:
        print("Running ICL inference step...")
        run_inference(task, args)
    else:
        print(f"Running reflection step {args.prompt_template_name} for {args.data_name} dataset...")
        run_reflection(task, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for active learning and in-context learning.")
    
    # General arguments
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--emb_model_type", type=str, default="openai", choices=["openai", "sbert", "huggingface"],
                        help="Type of embedding model to use (default: openai).")
    parser.add_argument("--emb_model_name", type=str, default="text-embedding-3-small",
                        help="Name of the embedding model to use (default: text-embedding-3-small).")
    parser.add_argument("--emb_batch_size", type=int, default=128, help="Batch size for embedding generation (default: 128).")
    parser.add_argument("--emb_sleep_time", type=int, default=1, help="Sleep time between API requests (default: 1 seconds).")
    parser.add_argument("--train_num", type=int, help="Number of sampled training examples used. If not specified, use all examples.")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "dev", "test", "test_sample_1000"], help="Evaluation split to use (default: test).")
    parser.add_argument("--eval_num", type=int, help="Number of sampled evaluation examples used. If not specified, use all examples.")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for sampling examples (default: 42).")

    # Inference step arguments
    parser.add_argument("--icl_inference", action="store_true", help="Run the in-context learning (ICL) inference step.")
    parser.add_argument("--model_type", type=str, default="openai", choices=["openai", "togetherai", "vllm"],
                        help="Type of language model to use (default: openai).")
    parser.add_argument("--model_name", type=str, required="--icl_inference" in sys.argv, 
                        help="Name of the language model (required for ICL inference).")
    parser.add_argument("--prompt_template_name", type=str, required=True, 
                        help="Template for prompt generation (default: icl_json_format).")
    parser.add_argument("--prior_prompt_template_name", type=str, 
                        help="Template for prompt generation (default: icl_json_format).")
    parser.add_argument("--icl_demo_num", type=int, default=8, 
                        help="Number of ICL demonstrations to use (default: 8).")
    parser.add_argument("--icl_span_demo_num", type=int, default=4, 
                        help="Number of ICL span demonstrations to use (default: 4).")
    parser.add_argument("--icl_demo_retrieval_method", type=str, default="kate", choices=["random", "kate", "deer"], 
                        help="Method for retrieving ICL demonstrations (default: kate).")
    parser.add_argument("--retrieval_seed", type=int, default=42, help="Random seed for retrieval reproducibility.")
    
    # DEER arguments
    parser.add_argument("--alpha_token_match", type=float, default=1.0, help="Weight for token match in demonstration retrieval (default: 1.0).")
    parser.add_argument("--alpha_embed_sim", type=float, default=1.0, help="Weight for embedding similarity in demonstration retrieval (default: 1.0).")
    parser.add_argument("--entity_weight", type=float, default=1.0, help="Weight for entity tokens in demonstration retrieval (default: 1.0).")
    parser.add_argument("--context_weight", type=float, default=1.0, help="Weight for context tokens in demonstration retrieval (default: 1.0).")
    parser.add_argument("--other_weight", type=float, default=0.01, help="Weight for other tokens in demonstration retrieval (default: 0.01).")
    parser.add_argument("--context_len", type=int, default=2, help="Number of context tokens of each entity to consider in demonstration retrieval (default: 2).")
    parser.add_argument("--entity_bound_unseen", type=float, default=0.5, help="Threshold for recognizting unseen entity token through potential entity tokens in the context (default: 0.5).")
    parser.add_argument("--context_bound_unseen", type=float, default=0.5, help="Threshold for recognizting unseen entity token through potential context tokens (default: 0.5).")
    parser.add_argument("--entity_bound_fn", type=float, default=0.95, help="Threshold for recognizting false negative entity token (default: 0.95).")
    parser.add_argument("--process_abbrev", type=str, default="none", choices=["none", "only", "unseen"])
    parser.add_argument("--ignore_rare", action="store_true", help="Ignore rare tokens in demonstration retrieval.")
    parser.add_argument("--ignore_article", action="store_true", help="Ignore article tokens in demonstration retrieval.")
    parser.add_argument("--include_unseen_boundary", action="store_true", help="Include unseen boundary tokens in demonstration retrieval.")
    parser.add_argument("--filter_single_token_fp", action="store_true", help="Filter single-token false positives that never appear as standalone entities.")
    
    # Language model arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation (default: 64).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling (default: 0.0).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling probability (default: 1.0).")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens in the output (default: 1024).")
    parser.add_argument("--stop", type=list, default=None, help="Stop sequences to end generation (default: None).")
    parser.add_argument('--response_format', type=str, default='text', choices=['text', 'json'],
                        help="Format of the model's output (default: text).")
    parser.add_argument("--sleep_time", type=int, default=5, help="Delay between batch requests (default: 0 seconds).")
    parser.add_argument("--logprobs", action="store_true", help="Return token log probabilities.")

    parser.add_argument("--no_async_mode", dest="async_mode", default=True, action="store_false",
                        help="Disable asynchronous mode for LLM inference (default: async mode is enabled).")

    args = parser.parse_args()
    print(args)

    main(args)