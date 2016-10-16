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

print 'Reading file ...'
lines = [line.strip().lower().split() for line in open(
    args.input_file,
    'r',
)]

vocab = {}

print 'Creating vocabulary ...'
for ind, line in enumerate(lines):
    if ind % 100000 == 0:
        print 'Finished %d out of %d lines ' % (ind, len(lines))
    for word in line:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

print 'Trimming vocabulary ...'
trimmed_vocab = set(sorted(
    vocab,
    key=lambda x: vocab[x],
    reverse=True
)[:int(args.count)])

print '%d items in trimmed vocabulary ' % (len(trimmed_vocab))

print 'Replacing with unks ...'
unked_lines = [
    [x if x in trimmed_vocab else '<unk>' for x in line]
    for line in lines
]

f = open(args.output_file, 'w')

print 'Writing output file ...'
for ind, line in enumerate(unked_lines):
    if ind % 100000 == 0:
        print 'Finished %d out of %d lines ' % (ind, len(unked_lines))
    f.write(' '.join(line) + '\n')
f.close()
