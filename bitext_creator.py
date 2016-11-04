"""Create a bitext corpus like src ||| tgt."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-src",
    "--src",
    help="src file",
    required=True
)
parser.add_argument(
    "-tgt",
    "--tgt",
    help="tgt file",
    required=True
)
parser.add_argument(
    "-o",
    "--output_path",
    help="output path",
    required=True
)

args = parser.parse_args()

src_f = open(args.src, 'r')
tgt_f = open(args.tgt, 'r')
op_f = open(args.output_path, 'w')
for ind, (src, tgt) in enumerate(zip(src_f, tgt_f)):
    if ind % 100000 == 0:
        print 'Finished %d lines ' % (ind)
    line = src.strip() + ' ||| ' + tgt.strip()
    op_f.write(line + '\n')
op_f.close()
