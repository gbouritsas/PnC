import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2ListOfListsOfLists2int(v):
    return [[[[] if c == ' ' else int(c) for c in vii.split(',')] for vii in vi.split(',,')] for vi in v.split(',,,')]


def str2ListOfLists2int(v):
    return [[[] if c == ' ' else int(c) for c in vi.split(',')] for vi in v.split(',,')]


def str2list2str(v):
    return [c for c in v.split(',')]

def str2list2int(v):
    return [-1 if c==' ' else int(c) for c in v.split(',')]


def str2list2float(v):
    return [float(c) for c in v.split(',')]


def str2list2bool(v):
    return [str2bool(c) for c in v.split(',')]

def ListOfListsOfint2str(v):
    string = ''
    for i,vi in enumerate(v):
        for j,c in enumerate(vi):
            if j==len(vi)-1:
                if i==len(v)-1:
                    string += str(c)
                else:
                    string += str(c)+',,'
            else:
                string += str(c)+','
    return string