# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License
# Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """Basic Message class for chat messages.

    Attributes:
        role (Role): The role of the message, can be `system`, `user`, or `assistant`.
        content (str): The content of the message.
    """

    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    """CompletionPrediction class for text completion predictions.

    Attributes:
        generation (str): The generated text.
        tokens (List[str]): List of generated tokens. Optional.
        logprobs (List[float]): List of log probabilities for the generated tokens. Optional.

    Note:
        `total=False` means the fields in `TypedDict` are optional.
        By default, all fields are required.

    """

    generation: str
    tokens: List[str]
    logprobs: List[float]


class ChatPrediction(TypedDict, total=False):
    """ChatPrediction class for chat completion predictions.

    Attributes:
        generation (Message): The generated message.
        tokens (List[str]): List of generated tokens. Optional.
        logprobs (List[float]): List of log probabilities for the generated tokens. Optional.

    Note:
        `total=False` means the fields in `TypedDict` are optional.
        By default, all fields are required.

    """

    generation: Message
    tokens: List[str]
    logprobs: List[float]


# Define Dialog as a list of messages.
Dialog = List[Message]


# Define some special tags.
# INST: Represents Instruction.
# SYS: Represents System.
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

# Define the error message when special tags are included in the prompt.
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    """Llama class for text generation using the language model."""

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        # Check if the torch distributed environment is not initialized.
        if not torch.distributed.is_initialized():
            # Initialize the process group for distributed operations using the NCCL backend.
            torch.distributed.init_process_group("nccl")

        # Check if model parallel is initialized.
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                # Get the model parallel size from the environment.
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            # Initialize model parallel.
            initialize_model_parallel(model_parallel_size)

        # Get the local rank from the environment.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set the device to the local rank.
        torch.cuda.set_device(local_rank)

        # TODO [keli]: Refer to https://github.com/meta-llama/llama/issues/1114
        # `seed` must be the same in all processes.
        torch.manual_seed(seed)

        # If not the master process, redirect stdout to /dev/null.
        if local_rank > 0:
            # os.devnull is the null device, which is used to discard output.
            sys.stdout = open(os.devnull, "w")

        # Start the timer.
        start_time = time.time()

        # Get the checkpoint files from the specified directory.
        # Q: Why sort the checkpoints?
        # A: To ensure that the model parallel rank matches the checkpoint index.
        # (Premise: The checkpoint filenames follow a format similar to 0.pth, 1.pth, and so on.)
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

        # Check if checkpoint files are found.
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

        # Check if the model parallel size matches the number of checkpoint files.
        # Q: Why do we need to check this? Why each process needs to load a different checkpoint?
        # A: Since we are using model parallelism, a big model is split into multiple parts and
        # each part is loaded by a different process.
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

        # Get the checkpoint path for the current model parallel rank.
        ckpt_path = checkpoints[get_model_parallel_rank()]

        # Load the checkpoint into CPU memory.
        # Q: Why load the checkpoint into CPU memory instead of GPU memory?
        # A: Personal guess is to unify the output device, ensuring the output is definitely on the
        # CPU. This improves compatibility and makes it easier for users. Otherwise, you can't
        # predict the output device, it could be the current process's GPU or another process's GPU.
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Load the parameters related to the construction of the model.
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Construct the model arguments.
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        # Load the tokenizer.
        tokenizer = Tokenizer(model_path=tokenizer_path)

        # Set the vocabulary size in the model arguments.
        model_args.vocab_size = tokenizer.n_words

        # Set the default tensor type to half precision (fp16).
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # Construct the model with the model arguments.
        model = Transformer(model_args)

        # Load the model state dictionary from the checkpoint.
        # Q: Why do we use strict=False?
        # A: When strict=False, it allows loading model parameters even if they do not perfectly
        # match the checkpoint. Some of the model's parameters may not be present in the
        # checkpoint, but it won't affect the loading process. Partial mismatches can occur when
        # we're in a model parallel environment.
        model.load_state_dict(checkpoint, strict=False)

        # Print the time taken to load the model.
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        # Return the constructed Llama instance.
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is
                represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in
                sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling.
                Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.
                Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the
                generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated
                token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs
                nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # Get model build parameters.
        params: ModelArgs = self.model.params
        bsz = len(prompt_tokens)

        # Ensure the current batch size does not exceed the maximum batch size.
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        # Ensure the longest prompt length does not exceed the maximum sequence length.
        assert max_prompt_len <= params.max_seq_len

        # Calculate the expected total length of the generated sequence. (for memory preallocation)
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        # Pre-construct the generation sequence tensor, filling it entirely with padding tokens.
        # [Shape] tokens: (batch_size, total_len)
        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device="cuda"
        )

        # Initialize the generation sequence tensor based on the input prompts.
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(
                t, dtype=torch.long, device="cuda"
            )

        # Initialize log probabilities if needed. Clearly, the shape of the log
        # probability tensor matches the generation sequence tensor.
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # Record the position of the last sampled token.
        prev_pos = 0

        # Determine if any sequence has reached the end-of-sequence token.
        # (i.e., if the sequence generation is complete)
        eos_reached = torch.tensor([False] * bsz, device="cuda")

        # Input text mask, used to determine if the current position is an input token.
        # eg. input_text_mask[0, 0:3] = [True, True, False] means the first two
        # positions are input tokens.
        input_text_mask = tokens != pad_id

        # If the shortest prompt length equals the total length, no new tokens will be generated,
        # thus directly calculating log probabilities.
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            ##################################################################################
            # Q: Why perform the logits.transpose(1, 2) operation when calculating log
            # probabilities?
            # A: The shape of logits is (batch_size, seq_len, vocab_size).
            # The documentation specifies that input must be in the form [C], [N, C],
            # [N, C, d1, d2, ...], etc.
            # Where C represents the number of classes, and N represents the batch size.
            # Therefore, a transpose operation is needed.
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
            ##################################################################################
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        # Start generating tokens from the minimum prompt length.
        # This ensures that the first generated tokens are meaningful.
        # If token generation starts from a position less than min_prompt_len,
        # the generated tokens will all be part of the already given prompt tokens.
        # ┌────────────────────────────┐
        # │ x: prompt token.           │
        # │ .: padding token.          │
        # │ g: generated token.        │
        # │                            │
        # │ Start from min_prompt_len. │
        # │           ↓                │
        # │ B1 [x][x][x][.][.][.]      │
        # │ B2 [x][x][.][.][.][.]      │
        # │ B3 [x][x][x][x][.][.]      │
        # │                            │
        # │ After one generation step. │
        # │                            │
        # │              ↓             │
        # │ B1 [x][x][x][.][.][.]      │
        # │ B2 [x][x][g][.][.][.]      │
        # │ B3 [x][x][x][x][.][.]      │
        # └────────────────────────────┘
        for cur_pos in range(min_prompt_len, total_len):
            ###############################################################################
            # Q: Why only provide tokens from [prev_pos:cur_pos]? Why not the entire tokens?
            # A: First, prev_pos represents the position from the last processing step,
            # initially 0.
            # The reason for providing only the tokens from [prev_pos:cur_pos] is due to the
            # KV Cache.
            # This way, there's no need to input the entire tokens each time.
            #
            # prev_pos mainly has two cases:
            # 1) When prev_pos = 0. This indicates the first generation step,
            # so we need to input tokens from [0:min_prompt_len].
            #    Compute the KV Cache for [0:min_prompt_len] and generate the next token.
            # 2) When prev_pos > 0. This means some tokens have already been generated,
            #    and cur_pos is always prev_pos + 1.
            #    In this case, input only one token each time, leveraging the cached KV Cache
            #    to generate the next token.
            ###############################################################################
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                # Adjust probability distribution via temperature to achieve controlled randomness.
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # Use top-p sampling to get the next token.
                # [Shape] next_token: (batch_size, 1)
                next_token = sample_top_p(probs, top_p)
            else:
                # Otherwise, directly select the token with the highest probability. (no randomness)
                # [Shape] next_token: (batch_size, 1)
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # [Shape] next_token: (batch_size, 1) -> (batch_size, )
            next_token = next_token.reshape(-1)

            # If the generated token is an input token, do not update.
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            if logprobs:
                # Update log probabilities. Refer to the earlier comment for understanding
                # logits.transpose(1, 2).
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = (
                    -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                )

            # Check if any sequence has finished generation.
            # 1. (~input_text_mask[:, cur_pos]) indicates the current position is a
            # generated token, not an input token.
            # 2. next_token == self.tokenizer.eos_id indicates the generated token is the
            # end-of-sequence token.
            # If both conditions are met, the generation process is complete for that sequence.
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            # End the process if all sequences have reached the end-of-sequence token.
            if all(eos_reached):
                break

        # Steps to construct the output result.
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # If echo is True, the output should include the original prompt, so start from 0.
            start = 0 if echo else len(prompt_tokens[i])
            # Truncate to the maximum generation length.
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][
                    start : len(prompt_tokens[i]) + max_gen_len
                ]
            # If the sequence contains an end-of-sequence token (indicating early termination),
            # truncate to the EOS position.
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in
                sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling.
                Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion
                sequence. If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.
                Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the
                generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the
                generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus
                sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        # Encode the prompts and add the beginning-of-sequence token.
        #######################################################################################
        # Q: Why don't we add the EOS token?
        # A: Because we need to generate text based on the prompts, so we don't need to add it.
        #######################################################################################
        prompt_tokens = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
        ]

        # Generate text completions based on the prompts.
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )

        # If `logprobs` is True, return the text, tokens, and log probabilities.
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]

        # Otherwise, return the text completions.
        return [
            {"generation": self.tokenizer.decode(t)} for t in generation_tokens
        ]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """Generate assistant responses for a list of conversational dialogs using the language
        generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is
                a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in
                sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling.
                Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response
                sequence. If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.
                Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant'
                generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and
                optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """  # noqa: D205
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        # Since the input dialogs are in the form of messages, we need to encode them into tokens.
        prompt_tokens = []
        # Check if the dialog contains special tags.
        unsafe_requests = []

        for dialog in dialogs:
            # Check if the dialog contains special tags.
            unsafe_requests.append(
                any(
                    [
                        tag in msg["content"]
                        for tag in SPECIAL_TAGS
                        for msg in dialog
                    ]
                )
            )

            # If the role of the first message is 'system', combine it with the second message.
            if dialog[0]["role"] == "system":
                # Use (B_SYS, E_SYS) to wrap the system message.
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": (
                            B_SYS
                            + dialog[0]["content"]
                            + E_SYS
                            + dialog[1]["content"]
                        ),
                    }
                ] + dialog[2:]

            # Check if the order of dialog roles is correct.
            # Must start with 'system', then 'user', and alternate between 'user' and 'assistant'.
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            # Enumerate the (prompt, answer) pairs in the dialog.
            # Use (B_INST, E_INST) to wrap the instruction.
            # `sum` operations is used to concatenate the tokenized dialog. The second parameter
            # denotes the initial value of the sum operation.
            # eg. sum([[1, 2], [3, 4]], []) -> [1, 2, 3, 4]
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )

            # Check if the last message in the dialog is from the user.
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"

            # Encode the last user message and append it to the dialog tokens. (Note, `eos=False`)
            #######################################################################################
            # Q: Why do we need to encode the last user message? Has it not been processed yet?
            # A: Although it seems that zip(dialog[::2], dialog[1::2]) has already enumerated
            # all the dialogs, it's not the case. That is because if the number of messages in
            # the dialog is odd, the last message will not be enumerated, as `zip` will
            # automatically cut off. eg. zip([1, 4, 5], [2, 3]) -> [(1, 2), (3, 4)].
            # Therefore, we need to encode the last user message separately.
            #######################################################################################
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        # Generate assistant responses based on the encoded dialogs.
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        # If logprobs is True, return a list of dictionaries containing the generated text,
        # generated tokens, and their log probabilities.
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": (
                            self.tokenizer.decode(t)
                            if not unsafe
                            else UNSAFE_ERROR
                        ),
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        # Otherwise, return a list of dictionaries containing only the generated text.
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": (
                        self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR
                    ),
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor. Shape: (batch_size, vocab_size).
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices. Shape: (batch_size, 1).

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        The resaon why this method can control the randomness and diversity of generated text is
        that by setting the threshold `p`, the number of small weighted tokens in the sampled set
        can be controlled.
        - When `p` is close to 1, the sampled set contains almost all tokens, which means the
          generated text is more random.
        - When `p` is close to 0, the sampled set contains only a few high-weighted tokens, which
          means the generated text is more deterministic.

        Refer: https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542
    """
    # Sort in descending order because nucleus sampling selects the token set from highest
    # to lowest probability.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Calculate the cumulative sum of probabilities.
    # This is to quickly perform subtraction later to determine if a token is within the
    # top-p (nucleus) set.
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Create a mask to exclude the tokens that their cumulative probabilities exceed the
    # threshold p.
    mask = probs_sum - probs_sort > p

    # Set the probabilities of the excluded tokens to 0.
    probs_sort[mask] = 0.0

    # Renormalize the distribution.
    # eg. [0.2, 0.2, 0.2, 0.2] / 0.8 -> [0.25, 0.25, 0.25, 0.25].
    # The `div_` method is an in-place operation.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample a token from the renormalized distribution.
    # The `multinomial` method samples a token index from the multinomial distribution.
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # Gather the sampled token index from the original index tensor.
    next_token = torch.gather(probs_idx, -1, next_token)

    # Return the sampled token index.
    return next_token
