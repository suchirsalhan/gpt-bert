from tokenizers import Tokenizer
import torch
from tqdm import tqdm
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the preprocessed combined.txt")
    parser.add_argument("--tokenizer_file", type=Path, required=True, help="Path to the tokenizer JSON file")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save tokenized output (.bin)")
    return parser.parse_args()


def tokenize_text(tokenizer, text):
    text = text.strip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return torch.tensor(ids, dtype=torch.int16)


def tokenize_file(input_file, output_file, tokenizer):
    tokenized_documents = []
    n_subwords = 0

    with input_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Tokenizing")):
            if not line.strip():
                continue
            tokenized = tokenize_text(tokenizer, line)
            tokenized_documents.append(tokenized)
            n_subwords += len(tokenized)

            if i == 0:
                print("\nExample document:")
                print(line.strip())
                print("Tokens:")
                print(" ".join(tokenizer.id_to_token(tok.item()) for tok in tokenized), "\n")

    torch.save(tokenized_documents, output_file)
    print(f"\nSaved {len(tokenized_documents)} documents with {n_subwords} total subwords to {output_file}")


if __name__ == "__main__":
    args = parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    tokenize_file(args.input_file, args.output_file, tokenizer)
