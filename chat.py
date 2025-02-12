import time
import openai
from dotenv import load_dotenv
from config import Config
import token_counter
from llm_utils import create_chat_completion
from logger import logger
import logging

cfg = Config()

def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.
    """
    return {"role": role, "content": content}

def generate_context(prompt, relevant_memory, full_message_history, model):
    current_context = [
        create_chat_message("system", prompt),
        create_chat_message("system", f"The current time and date is {time.strftime('%c')}"),
        create_chat_message("system", f"This reminds you of these events from your past:\n{relevant_memory}\n\n")
    ]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = token_counter.count_message_tokens(current_context, model)
    return next_message_to_add_index, current_tokens_used, insertion_index, current_context

def chat_with_ai(prompt, user_input, full_message_history, permanent_memory, token_limit):
    """Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory."""
    while True:
        try:
            model = cfg.fast_llm_model  # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response
            logger.debug(f"Token limit: {token_limit}")
            send_token_limit = token_limit - 1000

            relevant_memory = '' if len(full_message_history) == 0 else permanent_memory.get_relevant(str(full_message_history[-9:]), 10)
            logger.debug(f'Memory Stats: {permanent_memory.get_stats()}')

            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model
            )

            while current_tokens_used > 2500:
                # Remove memories until we are under 2500 tokens
                relevant_memory = relevant_memory[1:]
                next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                    prompt, relevant_memory, full_message_history, model
                )

            # Only account for user input if it is non-empty
            if user_input.strip():
                user_message = create_chat_message("user", user_input)
                current_tokens_used += token_counter.count_message_tokens([user_message], model)

            while next_message_to_add_index >= 0:
                message_to_add = full_message_history[next_message_to_add_index]
                tokens_to_add = token_counter.count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                # Add the most recent message to the context after the initial system messages.
                current_context.insert(insertion_index, message_to_add)
                current_tokens_used += tokens_to_add
                next_message_to_add_index -= 1

            # Append user input if it is non-empty
            if user_input.strip():
                current_context.append(create_chat_message("user", user_input))

            tokens_remaining = token_limit - current_tokens_used

            logger.debug(f"Token limit: {token_limit}")
            logger.debug(f"Send Token Count: {current_tokens_used}")
            logger.debug(f"Tokens remaining for response: {tokens_remaining}")
            logger.debug("------------ CONTEXT SENT TO AI ---------------")
            for message in current_context:
                # Skip printing the prompt
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                logger.debug(f"{message['role'].capitalize()}: {message['content']}")
                logger.debug("")
            logger.debug("----------- END OF CONTEXT ----------------")

            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Update full message history only if user_input is non-empty
            if user_input.strip():
                full_message_history.append(create_chat_message("user", user_input))
                full_message_history.append(create_chat_message("assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            print("Error: API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
