# SPDX-License-Identifier: Apache-2.0

from vllm import LLM

from vllm_ft.util import build_request_items, get_speculative_config, make_arg_parser


def run():
    parser = make_arg_parser(
        "Simple vLLM text generation.",
        default_prompt_source="hardcoded",
        default_num_requests=3,
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        speculative_config=get_speculative_config(args),
    )

    request_items = build_request_items(args, llm.get_tokenizer())
    prompts = [req.prompt for req, _ in request_items]
    # Use the sampling params from the first item.
    sampling_params = request_items[0][1]

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    run()
