"""Oracle."""
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-tb",
    "--train_bitext",
    help="path to train bitext",
    required=True
)
parser.add_argument(
    "-ta",
    "--train_alignments",
    help="path to train alignments",
    required=True
)
parser.add_argument(
    "-o",
    "--output_path",
    help="output path",
    required=True
)

args = parser.parse_args()

alignments = [
    line.strip().split() for line in open(args.train_alignments, 'r')
]

bitext = [line.strip().split(' ||| ') for line in open(args.train_bitext, 'r')]

src_sents = [x[0].split() for x in bitext]
tgt_sents = [x[1].split() for x in bitext]


def parse_alignment(alignment):
    """Parse the word alignments into dictionaries."""
    forward_alignment = {}
    backward_alignment = {}
    for alignment_pair in alignment:
        alignment_pair = [int(x) for x in alignment_pair.split('-')]
        if alignment_pair[0] not in forward_alignment:
            forward_alignment[alignment_pair[0]] = [alignment_pair[1]]
        else:
            forward_alignment[alignment_pair[0]].append(alignment_pair[1])
        if alignment_pair[1] not in backward_alignment:
            backward_alignment[alignment_pair[1]] = [alignment_pair[0]]
        else:
            backward_alignment[alignment_pair[1]].append(alignment_pair[0])
    return forward_alignment, backward_alignment


def get_not_in_target(alignment):
    """Get Words not aligned to a target word."""
    targets = [int(x.split('-')[0]) for x in alignment]
    return set(range(max(targets))) - set(targets)

actions = []
for ind, (alignment, english_sent, french_sent) in enumerate(zip(alignments, src_sents, tgt_sents)[:10]):

    if ind % 1000 == 0:
        print 'Finished %d out of %d ' % (ind, len(src_sents))
    indices = set()
    backward_alignment, forward_alignment = parse_alignment(alignment)
    not_in_target = get_not_in_target(alignment)
    target_start = 0
    source_start = 0
    actionset = []

    for ind, word in enumerate(english_sent):

        # If word in source doesn't have an alignment, SHIFT
        if ind not in forward_alignment:
            print '{: <2} | {: <20} | {: <2} | {: <30} | {: <30} '.format(ind, word, 'NS', ' '.join(english_sent[source_start:ind+1]), 'NULL')
            actionset.append([word, 'S', ' '.join(english_sent[source_start:ind+1]), ' '.join(english_sent[ind+1:]),  '$NONE$'])
            continue

        # Add all the alignments for this word to indices
        for item in forward_alignment[ind]:
            indices.add(item)
        action = 'T'
        max_ind = max(indices)

        # Check if all target indices of the max spanning source block alignment are contained if not, SHIFT
        for i in range(target_start, max_ind):
            if i in not_in_target:
                continue
            if i not in indices:
                action = 'S'
                break

        # Write the SHIFT action to the actionset
        if action == 'S':
            print '{: <2} | {: <20} | {: <2} | {: <30} | {: <30} | {: <10}  '.format(ind, word, action, ' '.join(english_sent[source_start:ind+1]), 'NULL', str(target_start) + ', ' + str(max_ind) + ' -> ' + ' '.join([str(x) for x in indices]))
            actionset.append([word, action, ' '.join(english_sent[source_start:ind+1]), ' '.join(english_sent[ind+1:]), '$NONE$'])

        # Write the TRANSLATE action to the actionset
        elif action == 'T':
            translation = french_sent[target_start:max_ind+1]
            curr_phrase_block = english_sent[source_start:ind+1]
            print '{: <2} | {: <20} | {: <2} | {: <30} | {: <30} | {: <10} '.format(ind, word, action, ' '.join(curr_phrase_block), ' '.join(translation), str(target_start)  + ' -> ' + str(max_ind))
            actionset.append([word, action, ' '.join(curr_phrase_block), ' '.join(english_sent[ind+1:]), ' '.join(translation) if translation != [] else '$NONE$'])
            target_start = max(max_ind + 1, target_start) # Max to handle the case where a word in the source maps backwards in the target
            source_start = ind + 1
            indices = set()
    actionset.insert(0, ' '.join(french_sent))
    actionset.insert(0, ' '.join(english_sent))
    #print ' '.join(french_sent)
    print '==============================================================================================='
    actions.append(actionset)
