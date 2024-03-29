import unicodedata
import re
from typing import List

from trie import Trie, TrieNode

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

NUM_SPECIAL_TOKENS = 5

VOCAB_TOKENS = ['<pad>', '<unk>', '<mask>', '<s>', '</s>', '|', '~', 'i', 'u', 'o', 'a', 'e', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 'l', 's', 't', 'v', 'y', 'w', 'z', 'bw', 'by', 'cw', 'cy', 'dw', 'fw', 'gw', 'hw', 'kw', 'jw', 'jy', 'ny', 'mw', 'my', 'nw', 'pw', 'py', 'rw', 'ry', 'sw', 'sy', 'tw', 'ty', 'vw', 'vy', 'zw', 'pf', 'ts', 'sh', 'shy', 'mp', 'mb', 'mf', 'mv', 'nc', 'nj', 'nk', 'ng', 'nt', 'nd', 'ns', 'nz', 'nny', 'nyw', 'byw', 'ryw', 'shw', 'tsw', 'pfy', 'mbw', 'mby', 'mfw', 'mpw', 'mpy', 'mvw', 'mvy', 'myw', 'ncw', 'ncy', 'nsh', 'ndw', 'ndy', 'njw', 'njy', 'nkw', 'ngw', 'nsw', 'nsy', 'ntw', 'nty', 'nzw', 'shyw', 'mbyw', 'mvyw', 'nshy', 'nshw', 'nshyw', 'bg', 'pfw', 'pfyw', 'vyw', 'njyw', 'x', 'q', ',', '.', '?', '!', '-', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\'']

SIL_ID = 5
BLANK_ID = 6

KINSPEAK_VOCAB = {v:i for i,v in enumerate(VOCAB_TOKENS)}
KINSPEAK_VOCAB_IDX = {i:v for i,v in enumerate(VOCAB_TOKENS)}

VOWELS = {'i', 'u', 'o', 'a', 'e'}

def next_syllab_options(dictionary:Trie, parent_node:TrieNode):
    if parent_node is None: # Non-word-state --> Return any char other than blank
        return [(k,dictionary.root.children[k]) if (k in dictionary.root.children) else (k,None) for k in KINSPEAK_VOCAB_IDX.keys() if k != BLANK_ID]
    if parent_node.is_end: # Return space or children
        return [(k,parent_node.children[k]) for k in parent_node.children.keys() if k != BLANK_ID] + [(k,None) for k in KINSPEAK_VOCAB_IDX.keys() if ((((k < KINSPEAK_VOCAB['i']) or (k > KINSPEAK_VOCAB['q'])) and (k != BLANK_ID)) and (not (k in parent_node.children.keys())))]
    else:
        return [(k,parent_node.children[k]) for k in parent_node.children.keys() if k != BLANK_ID] + ([] if (KINSPEAK_VOCAB['\''] in parent_node.children.keys()) else [(KINSPEAK_VOCAB['\''],None)])

def next_syllab_options_s2s(dictionary:Trie, parent_node:TrieNode):
    if parent_node is None: # Non-word-state --> Return any char other than blank
        return [(k,dictionary.root.children[k]) if (k in dictionary.root.children) else (k,None) for k in KINSPEAK_VOCAB_IDX.keys()]
    if parent_node.is_end: # Return non-words or word children
        return [(k,parent_node.children[k]) for k in parent_node.children.keys()] + [(k,None) if (k!=BLANK_ID) else (k,parent_node) for k in KINSPEAK_VOCAB_IDX.keys() if ((k < KINSPEAK_VOCAB['i']) or (k > KINSPEAK_VOCAB['q']))]
    else:
        return [(k,parent_node.children[k]) for k in parent_node.children.keys()] + [(KINSPEAK_VOCAB['\''],None)]

def build_kinyarwanda_dictionary_trie(filename='/home/nzeyi/KINLP/data/KinTokens.tsv') -> Trie:
    trie = Trie()
    lines = []
    with open(filename, 'rb') as f:
        for i, line in enumerate(f,1):
            try:
                lines.append(line.decode('utf-8').rstrip('\n'))
            except UnicodeDecodeError as e:
                pass
    for line in lines:
        tks = line.split('\t')
        if len(tks) == 8:
            token = tks[0].lower()
            totalCount = int(tks[2])
            totalDocuments = int(tks[3])
            if (totalCount >= 2) and (totalDocuments >= 2):
                seq = text_to_id_sequence(token)
                trie.insert(seq)
    return trie

def normalize_text(string, encoding="utf-8") -> str:
    string = string.decode(encoding) if isinstance(string, type(b'')) else string
    string = string.replace('`','\'')
    string = string.replace("'", "\'")
    string = string.replace("‘", "\'")
    string = string.replace("’", "\'")
    string = string.replace("‚", "\'")
    string = string.replace("‛", "\'")
    string = string.replace(u"æ", u"ae").replace(u"Æ", u"AE")
    string = string.replace(u"œ", u"oe").replace(u"Œ", u"OE")
    return unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode("ascii").lower()

def append_new(seq, val):
    if len(seq) > 0:
        if (seq[-1] == val):
            seq.append(BLANK_ID)
    seq.append(val)
def process_cons(cons, seq):
    if cons in KINSPEAK_VOCAB:
        append_new(seq,KINSPEAK_VOCAB[cons])
    else:
        for c in cons:
            if c in KINSPEAK_VOCAB:
                append_new(seq, KINSPEAK_VOCAB[c])

def text_to_id_sequence(text) -> List[int]:
    seq = []
    txt = normalize_text(text)
    txt = re.sub(r"\s+", '|', txt).strip()
    start = 0
    end = 0
    while(end < len(txt)):
        if(txt[end] in VOWELS) or (txt[end] == '|'):
            if(end > start):
                process_cons(txt[start:end], seq)
            append_new(seq, KINSPEAK_VOCAB[txt[end]])
            end += 1
            start = end
        else:
            end += 1
    if (end > start):
        process_cons(txt[start:end], seq)
    return seq

def id_sequence_to_text(seq) -> str:
    return ''.join([' ' if (id == SIL_ID) else KINSPEAK_VOCAB_IDX[id] for id in seq if id!=BLANK_ID])

def syllbe_vocab_size() -> int:
    return len(VOCAB_TOKENS)

def syllbe_vocab_units() -> List[str]:
    return VOCAB_TOKENS

CONSONANTS = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 'l', 's', 't', 'v', 'y', 'w', 'z', 'bw', 'by', 'cw', 'cy', 'dw', 'fw', 'gw', 'hw', 'kw', 'jw', 'jy', 'ny', 'mw', 'my', 'nw', 'pw', 'py', 'rw', 'ry', 'sw', 'sy', 'tw', 'ty', 'vw', 'vy', 'zw', 'pf', 'ts', 'sh', 'shy', 'mp', 'mb', 'mf', 'mv', 'nc', 'nj', 'nk', 'ng', 'nt', 'nd', 'ns', 'nz', 'nny', 'nyw', 'byw', 'ryw', 'shw', 'tsw', 'pfy', 'mbw', 'mby', 'mfw', 'mpw', 'mpy', 'mvw', 'mvy', 'myw', 'ncw', 'ncy', 'nsh', 'ndw', 'ndy', 'njw', 'njy', 'nkw', 'ngw', 'nsw', 'nsy', 'ntw', 'nty', 'nzw', 'shyw', 'mbyw', 'mvyw', 'nshy', 'nshw', 'nshyw', 'bg', 'pfw', 'pfyw', 'vyw', 'njyw'}

def has_valid_kinyarwanda_orthography(text) -> bool:
    if text is None:
        return False
    seq = text_to_id_sequence(text)
    if len(seq) == 0:
        return False
    prev = seq[0]
    for next in seq[1:-1]:
        if (KINSPEAK_VOCAB_IDX[prev] in VOWELS) and (KINSPEAK_VOCAB_IDX[next] in VOWELS):
            return False
        if (KINSPEAK_VOCAB_IDX[prev] in CONSONANTS) and (KINSPEAK_VOCAB_IDX[next] in CONSONANTS):
            return False
        if not ((KINSPEAK_VOCAB_IDX[prev] in VOWELS) or (KINSPEAK_VOCAB_IDX[prev] in CONSONANTS) or (KINSPEAK_VOCAB_IDX[prev] == '\'')):
            return False
        prev = next
    next = seq[-1]
    if (KINSPEAK_VOCAB_IDX[prev] in CONSONANTS) and (KINSPEAK_VOCAB_IDX[next] in CONSONANTS):
        return False
    if not (KINSPEAK_VOCAB_IDX[next] in VOWELS):
        return False
    if len(seq) > 1:
        if not ((KINSPEAK_VOCAB_IDX[prev] in VOWELS) or (KINSPEAK_VOCAB_IDX[prev] in CONSONANTS) or (KINSPEAK_VOCAB_IDX[prev] == '\'')):
            return False
    if len(seq) == 1:
        return (text =='u') or (text == 'i')
    return True

if __name__ == '__main__':
    assert len(VOCAB_TOKENS) == len(KINSPEAK_VOCAB), "Vocab length mismatch"
    assert len(VOCAB_TOKENS) == len(KINSPEAK_VOCAB_IDX), "Vocab length mismatch"

    tests = ['abana', 'terimbere',
             'Miel ok',
             'Gourmand grande ?',
             '5 Xaviersss wacu twese',
             'Ishyano riratugwiriye ',
             "Nta munsi w’ubusa wigeze wirenza muri uyu mwaka Abanyarwanda batibajije ku myitwarire ya Perezida wa RDC, kubera uburyo batunguwe mu kanya nk’ako guhumbya."]
    for test in tests:
        seq = text_to_id_sequence(test)
        print(test, [KINSPEAK_VOCAB_IDX[id] for id in seq], id_sequence_to_text(seq))
    file = "/mnt/NVM/KINLP/data/kinspeak/jw_speech/samples/new_txt/77327059-53be-42ee-8ea1-21e1b2c30999_299_203471_209245_3171_3273.txt"
    f = open(file, 'r')
    test = f.read()
    f.close()
    seq = text_to_id_sequence(test)
    print(test, '\n', [KINSPEAK_VOCAB_IDX[id] for id in seq], '\n', id_sequence_to_text(seq))
    print()
    test_kinya = ['test', 'aban', 'Imana', 'serupyipyinyurimpyisi', 'twendE', 'Ishakwe', '123', 'y\'Imana', 'u', 'a', 'i', 'ok', 'hose', 'alexandre', 'koxo']
    for t in test_kinya:
        print(t,':', has_valid_kinyarwanda_orthography(t))

    text = "Igirire ikizere kandi uhore wisanzuye Abagore benshi banezezwa no kubona umugabo wihagazeho kandi wifitiye ikizere n' ubwo bamwe muri bo bigira nk' aho bitabafashe ho ."
    print(id_sequence_to_text(text_to_id_sequence(text)))

    text = "Twaramubajije tuti “ubwo se niduhura n’abasirikare ba leta, ntibatekereza ko natwe turi abasirikare?” "
    print(id_sequence_to_text(text_to_id_sequence(text)))
