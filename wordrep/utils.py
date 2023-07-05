from collections import defaultdict
from typing import Dict, List, Union
from spacy.language import Language
from spacy.tokens import Token
from simalign import SentenceAligner
import spacy
from torch import align_tensors
from tqdm import tqdm

IS_OPEN = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']
IS_CLOSE = ['ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ']
IS_OTHER =['PUNCT', 'SYM', 'X']
MODELS = {
        'en': "en_core_web_lg",
        'fr': "fr_core_news_lg",
        }
ALIGN_METHODS = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

class BaseRepDetector():

    def __init__(self, lang: str) -> None:
        if lang in MODELS:
            self.nlp = spacy.load(MODELS[lang], disable=['parser','ner'])
        else:
            raise KeyError(f"available langs are {MODELS.keys()} but get {lang}")
    
    def _tokenize(self, sent):
        return self.nlp(sent)

    def _detect(self, tokens: List[Token], account, account_lem) -> Dict[str, List[int]]:
        if account == 'open':
            acc = IS_OPEN
        if account == 'clopen':
            acc = IS_OPEN + IS_CLOSE
        if account == 'all':
            acc = IS_OPEN + IS_CLOSE + IS_OTHER

        d = defaultdict(list)
 
        for ix, tok in enumerate(tokens):
            if tok.pos_ in acc:
                k = tok.lemma_ if account_lem else tok.text
                d[k].append(ix)

        return {k: v for k, v in d.items() if  len(v) > 1}

    def _visualize(self, tokens: List[Token], rep:Dict[str, List[int]]):
        """visualize word rep
        """
        toks = list(map(lambda x: x.text, tokens))
        if rep:
            for v in rep.values():
                for idx in v:
                    toks[idx] = f"\033[4m{toks[idx]}\033[0m"
            sent_marked = ' '.join(toks)
            print(sent_marked)
        else:
            print('[info]No reps')

    def detect_corpus(self, corpus, idx=None, acc='open', acc_lem=False, vis=True):
        with open(corpus, encoding='utf-8') as f:
            reps = []
            lines = f.readlines()
        iterable = enumerate(lines)
        if idx != None:
            lines = [lines[i] for i in idx]
            iterable = zip(idx, lines)
        for i, l in iterable:
            toks = self._tokenize(l)
            rep = self._detect(toks, account=acc, account_lem=acc_lem)
            if rep:
                reps.append((i, rep))
                if vis:
                    self._visualize(toks, rep)
        return reps

class AlignRepDetector():
    def __init__(self, src_lang: str, tgt_lang: str, align_method='i') -> None:
        self.src_detector = BaseRepDetector(src_lang)
        self.tgt_detector = BaseRepDetector(tgt_lang)
        self.align_model =  SentenceAligner(matching_methods=align_method)
        self.align_method = align_method

    def _tokenize(self, src_sent, tgt_sent):

        src_tokens = self.src_detector._tokenize(src_sent)
        tgt_tokens = self.tgt_detector._tokenize(tgt_sent)

        return src_tokens, tgt_tokens

    def _detect(self, src_sent, tgt_sent, account, vis=False):
        tgt_tokens = self.tgt_detector._tokenize(tgt_sent)
        tgt_rep = self.tgt_detector._detect(tgt_tokens, account)
        
        # early break for efficiency
        if tgt_rep == {}:
            return {}, {}, {}, [], []

        src_tokens = self.src_detector._tokenize(src_sent)
        src_rep = self.src_detector._detect(src_tokens, account='all')
        src = [token.text for token in src_tokens]
        tgt = [token.text for token in tgt_tokens]


        align = self.align_model.get_word_aligns(src, tgt)[ALIGN_METHODS[self.align_method]]

        # closure 
        # src_domain = set([a[0] for a in align])
        # tgt_domain = set([a[1] for a in align])

        # for i in set(range(len(src))).difference(src_domain):
        #     align.append((i,'*'))
        # for j in set(range(len(tgt))).difference(tgt_domain):
        #     align.append(('*',j))

        src_to_tgt = defaultdict(list)
        for (id1, id2) in align:
            src_to_tgt[id1].append(id2)

        tgt_to_src = defaultdict(list)
        for (id1, id2) in align:
            tgt_to_src[id2].append(id1)

        true_rep = {k: set(v) for k,v in tgt_rep.items()}

        for src_tok, src_ids in src_rep.items():
            for src_id in src_ids:
                for tgt_tok in tgt_rep.keys():
                    true_rep[tgt_tok] = true_rep[tgt_tok].difference(set(src_to_tgt[src_id]))
        
        true_rep = {k: tgt_rep[k] for k, v in true_rep.items() if v != set()}

        preimage = []
        for tgt_idx in true_rep.values():
            for tgt_id in tgt_idx:
                preimage.append(tgt_to_src[tgt_id])
        preimage = [i for i in preimage if len(i) > 0]

        return true_rep, preimage, src_rep, src_tokens, tgt_tokens

    def _visualize(self, tokens: List[Token], rep:Union[Dict[str, List[int]], List[List[int]]]):
        """visualize word rep
        """
        toks = list(map(lambda x: x.text, tokens))
        if isinstance(rep,dict):
            r = rep.values()
        elif isinstance(rep, list):
            r = rep
        if len(r) == 0:
            print('[info]No reps')
            return
        for v in r:
            for idx in v:
                toks[idx] = f"\033[4m{toks[idx]}\033[0m"
        sent_marked = ' '.join(toks)
        print(sent_marked)

    def detect(self, src_sent, tgt_sent, account='open', vis=False):
        rep, preimage, src_rep, src_tokens, tgt_tokens = self._detect(src_sent, tgt_sent, account=account)
        if vis:
            if preimage:
                self._visualize(src_tokens, preimage)
            else:
                print(src_sent)
                self._visualize(tgt_tokens, rep)
        return rep, preimage, src_rep
    
    def detect_corpus(self, src_corpus, tgt_corpus, account='open'):
        reps = []
        with open(src_corpus, 'r', encoding='utf-8') as src_f:
            with open(tgt_corpus, 'r', encoding='utf-8') as tgt_f:
                    src_sents = src_f.readlines()
                    tgt_sents = tgt_f.readlines()
                    assert len(src_sents) == len(tgt_sents), f"{src_corpus}({len(src_sents)}) and {tgt_corpus}({len(tgt_sents)}) do not have same number of lines"
                    for i in tqdm(range(len(src_sents))):
                        src_sent = src_sents[i]
                        tgt_sent = tgt_sents[i]
                        rep, preimage, _ = self.detect(src_sent, tgt_sent, account)
                        if rep:
                            reps.append((i, rep, preimage))
        return reps
 
    def visualize_corpus(self, src_corpus, tgt_corpus, reps, ref_corpus=None):
        with open(src_corpus, 'r', encoding='utf-8') as src_f:
            with open(tgt_corpus, 'r', encoding='utf-8') as tgt_f:
                    src_sents = src_f.readlines()
                    tgt_sents = tgt_f.readlines()
                    assert len(src_sents) == len(tgt_sents), f"{src_corpus}({len(src_sents)}) and {tgt_corpus}({len(tgt_sents)}) do not have same number of lines"
                    if ref_corpus:
                        with open(ref_corpus, 'r', encoding='utf-8') as ref_f:
                            ref_sents = ref_f.readlines()
                            assert len(src_sents) == len(ref_sents), f"{src_corpus}({len(src_sents)}) and {ref_corpus}({len(ref_sents)}) do not have same number of lines"
                    for (i, rep, preimage) in reps:
                        src_sent = src_sents[i]
                        tgt_sent = tgt_sents[i]
                        src_tokens, tgt_tokens = self._tokenize(src_sent, tgt_sent)
                        print(f"line {i}--------")
                        if preimage:
                            self._visualize(src_tokens, preimage)
                        else:
                            print(src_sent)
                        self._visualize(tgt_tokens, rep)
                        if ref_corpus:
                            ref_sent = ref_sents[i]
                            print(ref_sent)