"""Augment dataset with <unk> characters."""
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_file",
    help="path to file to be unk",
    required=True
)
parser.add_argument(
    "-c",
    "--count",
    help="number of words in vocabulary to retain",
    required=True
)
parser.add_argument(
    "-o",
    "--output_file",
    help="path to output file",
    required=True
)

args = parser.parse_args()

lines = [line.strip().split() for line in codecs.open(
    args.input_file,
    'r',
    encoding='utf-8'
)]

vocab = {}

for line in lines:
    for word in line:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

trimmed_vocab = set(sorted(
    vocab,
    key=lambda x: vocab[x],
    reverse=True
)[:int(args.count)])

unked_lines = [
    [x.lower() if x in trimmed_vocab else '<unk>' for x in line]
    for line in lines
]

f = codecs.open(args.output_file, 'w', encoding='utf-8')

for line in unked_lines:
    f.write(' '.join(line) + '\n')
f.close()
