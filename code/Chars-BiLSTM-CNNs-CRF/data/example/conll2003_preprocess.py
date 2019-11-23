from pathlib import Path

def inputpath(name):
    return 'eng.{}'.format(name)

def wordspath(name):
    return '{}.words.txt'.format(name)

def tagspath(name):
    return '{}.tags.txt'.format(name)

def process(inp, wp, tp):
    sentences = [[]]
    tags = [[]]
    with Path(inp).open() as rf:
        for line in rf.readlines():
            if line.strip() == '':
                sentences.append([])
                tags.append([])
            else:
                items = line.strip().split()
                sentences[-1].append(items[0])
                tags[-1].append(items[3])

    with Path(wp).open('w') as wf, Path(tp).open('w') as tf:
        for line in sentences:
            if line:
                wf.write(' '.join(line) + '\n')
        for line in tags:
            if line:
                tf.write(' '.join(line) + '\n')

if __name__ == '__main__':
    for ds in ['train', 'testa', 'testb']:
        process(inputpath(ds), wordspath(ds), tagspath(ds))