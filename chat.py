import fire
from llama import Llama
from pprint import pprint
from typing import Optional


def main(
    ckpt_dir: str = 'llama-2-7b-chat',
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [[]]
    user_input = ""

    print()
    while True:

        user_input = input("User: ")
        dialogs[0].append({"role": "user", "content": user_input})

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        generation = results[0]["generation"]
        content = generation["content"]
        dialogs[0].append(generation)

        print()
        print("Assistant:", content)
        print()


if __name__ == "__main__":
    fire.Fire(main)
