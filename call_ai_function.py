from config import Config

cfg = Config()
from llm_utils import create_chat_completion

def call_ai_function(function, args, description, model=None):
    """Call an AI function."""
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # Parse args to a comma-separated string
    args = ", ".join(args)
    # Ensure that args is not empty before sending to the API
    if not args.strip():
        args = "[No input provided]"

    messages = [
        {
            "role": "system",
            "content": (
                f"You are now the following python function: ```# {description}\n{function}```\n\n"
                "Only respond with your `return` value."
            ),
        },
        {"role": "user", "content": args},
    ]

    response = create_chat_completion(
        model=model, messages=messages, temperature=0
    )

    return response
