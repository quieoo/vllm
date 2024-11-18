from vllm import LLM, SamplingParams
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",
                        help="Specify where the model is",
                        required=True)
    
    args = parser.parse_args()

    PROMPT_TEMPLATE = "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"  # noqa: E501
    system_message = "You are a friendly chatbot who always responds in the style of a pirate."  # noqa: E501
    # Sample prompts.
    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]
    prompts = [
        PROMPT_TEMPLATE.format(system_message=system_message, prompt=prompt)
        for prompt in prompts
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    llm = LLM(model=args.model_dir, enforce_eager=True)


    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


