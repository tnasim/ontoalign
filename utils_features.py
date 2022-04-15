from itertools import product
from re import finditer

import ngram
from fuzzycomp import fuzzycomp
from gensim.models import KeyedVectors
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from nltk.corpus import wordnet
from py_stringmatching.similarity_measure.affine import Affine
from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.cosine import Cosine
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.editex import Editex
from py_stringmatching.similarity_measure.generalized_jaccard import \
    GeneralizedJaccard
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.needleman_wunsch import \
    NeedlemanWunsch
from py_stringmatching.similarity_measure.overlap_coefficient import \
    OverlapCoefficient
from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
from py_stringmatching.similarity_measure.partial_token_sort import \
    PartialTokenSort
from py_stringmatching.similarity_measure.ratio import Ratio
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.similarity_measure.soundex import Soundex
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.token_sort import TokenSort
from py_stringmatching.similarity_measure.tversky_index import TverskyIndex
from tqdm import tqdm

af = Affine()
me = MongeElkan()
nw = NeedlemanWunsch()
sw = SmithWaterman()
bd = BagDistance()
cos = Cosine()
pr = PartialRatio()
sf = SoftTfIdf()
edx = Editex()
gj = GeneralizedJaccard()
jw = JaroWinkler()
lev = Levenshtein()
dice = Dice()
jac = Jaccard()
jaro = Jaro()
pts = PartialTokenSort()
rat = Ratio()
sound = Soundex()
tfidf = TfIdf()
ts = TokenSort()
tv_ind = TverskyIndex()
over_coef = OverlapCoefficient()

# It's long

# ---------------------------------------------------------------------------------
# Gensim word2vec Pre-trained Models:
# ---------------------------------------------------------------------------------
# https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
# ---------------------------------------------------------------------------------
# conceptnet-numberbatch-17-06-300 (1917247 records)
# fasttext-wiki-news-subwords-300 (999999 records)
# glove-twitter-100 (1193514 records)
# glove-twitter-200 (1193514 records)
# glove-twitter-25 (1193514 records)
# glove-twitter-50 (1193514 records)
# glove-wiki-gigaword-100 (400000 records)
# glove-wiki-gigaword-200 (400000 records)
# glove-wiki-gigaword-300 (400000 records)
# glove-wiki-gigaword-50 (400000 records)
# word2vec-google-news-300 (3000000 records)
# word2vec-ruscorpora-300 (184973 records)

print('Loading word2vec model...')
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model = api.load("glove-wiki-gigaword-300")
# model = api.load("conceptnet-numberbatch-17-06-300")
model = api.load("glove-wiki-gigaword-300")

# model1 = api.load("glove-wiki-gigaword-300")
# model2 = api.load("word2vec-google-news-300")
# model3 = api.load("glove-twitter-200")
# model4 = api.load("fasttext-wiki-news-subwords-300")
# model5 = api.load("conceptnet-numberbatch-17-06-300")

# corpus = api.load('wiki-english-20171001')
# model = Word2Vec(corpus)

print('Word2vec models are loaded.')


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                       identifier)
    return [m.group(0) for m in matches]


def get_word2vec_sim(row_set1, row_set2, model):
    sum_sim = 0
    N = max(len(row_set1), len(row_set2))

    for w1 in row_set1:
        maxSim = 0
        for w2 in row_set2:
            try:
                sim = model.wv.similarity(w1, w2)
            except:
                sim = 0

            if sim > maxSim:
                maxSim = sim
        sum_sim = sum_sim + maxSim

    sum_sim = sum_sim / N

    return sum_sim


def get_words(text):
    if '_' in text:
        row_set = text.split('_')
    else:
        if '-' in text:
            row_set = text.split('-')
        else:
            row_set = camel_case_split(text)

    row_set = [x.lower() for x in row_set]
    return row_set


def calculate_features(dataset, string_type):
    ngrams1 = []
    ngrams2 = []
    ngrams3 = []
    ngrams4 = []
    dices = []
    jaccards = []
    jaros = []
    lcs = []
    mes = []
    sws = []
    afs = []
    bds = []
    coses = []
    prs = []
    sfs = []
    edxs = []
    gjs = []
    jws = []
    lws = []
    ptss = []
    rats = []
    sounds = []
    tfidfs = []
    tss = []
    tvs = []
    ovs = []
    nws = []
    wordnet_sims = []
    w2vec_sims1 = []
    w2vec_sims2 = []
    w2vec_sims3 = []
    w2vec_sims4 = []
    w2vec_sims5 = []

    if string_type == 'Entity':
        index = 2
    elif string_type == 'Parent':
        index = 4
    elif string_type == 'Path':
        index = 6

    for key, row in tqdm(dataset.iterrows()):

        string1 = row[index]
        string2 = row[index + 1]

        ngrams1.append(ngram.NGram.compare(string1, string2, N=1))
        ngrams2.append(ngram.NGram.compare(string1, string2, N=2))
        ngrams3.append(ngram.NGram.compare(string1, string2, N=3))
        ngrams4.append(ngram.NGram.compare(string1, string2, N=4))
        lws.append(lev.get_sim_score(string1, string2))
        jaros.append(jaro.get_sim_score(string1, string2))
        lcs.append(2 * fuzzycomp.lcs_length(string1, string2) / (
                len(string1) + len(string2)))
        nws.append(nw.get_raw_score(string1, string2))
        sws.append(sw.get_raw_score(string1, string2))
        afs.append(af.get_raw_score(string1, string2))
        bds.append(bd.get_sim_score(string1, string2))
        prs.append(pr.get_sim_score(string1, string2))
        edxs.append(edx.get_sim_score(string1, string2))
        ptss.append(pts.get_sim_score(string1, string2))
        rats.append(rat.get_sim_score(string1, string2))
        sounds.append(sound.get_sim_score(string1, string2))
        tss.append(ts.get_sim_score(string1, string2))
        jws.append(jw.get_sim_score(string1, string2))

        row_set1 = get_words(string1)
        row_set2 = get_words(string2)

        mes.append(me.get_raw_score(row_set1, row_set2))
        coses.append(cos.get_sim_score(row_set1, row_set2))
        sfs.append(sf.get_raw_score(row_set1, row_set2))
        gjs.append(gj.get_sim_score(row_set1, row_set2))
        tfidfs.append(tfidf.get_sim_score(row_set1, row_set2))
        tvs.append(tv_ind.get_sim_score(row_set1, row_set2))
        ovs.append(over_coef.get_sim_score(row_set1, row_set2))
        dices.append(dice.get_sim_score(row_set1, row_set2))
        jaccards.append(jac.get_sim_score(row_set1, row_set2))

        # >>> t = wn.synsets('fly', wn.VERB)[0]
        # >>> s = wn.synsets('say', wn.VERB)[0]
        # >>> print(s.shortest_path_distance(t))
        # None
        # >>> print(s.path_similarity(t, simulate_root=False))
        # None
        # >>> print(s.lch_similarity(t, simulate_root=False))
        # None
        # >>> print(s.wup_similarity(t, simulate_root=False))
        allsyns1 = set(ss for word in row_set1 for ss in wordnet.synsets(word))
        allsyns2 = set(ss for word in row_set2 for ss in wordnet.synsets(word))

        best = [wordnet.wup_similarity(s1, s2) for s1, s2 in
                product(allsyns1, allsyns2)]
        if len(best) > 0:
            wordnet_sims.append(best[0])
        else:
            wordnet_sims.append(0)
        
        w2vec_sims1.append(get_word2vec_sim(row_set1, row_set2, model))
        # w2vec_sims1.append(get_word2vec_sim(row_set1, row_set2, model1))
        # w2vec_sims2.append(get_word2vec_sim(row_set1, row_set2, model2))
        # w2vec_sims3.append(get_word2vec_sim(row_set1, row_set2, model3))
        # w2vec_sims4.append(get_word2vec_sim(row_set1, row_set2, model4))
        # w2vec_sims5.append(get_word2vec_sim(row_set1, row_set2, model5))

    dataset['Ngram1' + '_' + string_type] = ngrams1
    dataset['Ngram2' + '_' + string_type] = ngrams2
    dataset['Ngram3' + '_' + string_type] = ngrams3
    dataset['Ngram4' + '_' + string_type] = ngrams4
    dataset['Dice' + '_' + string_type] = dices
    dataset['Jaccard' + '_' + string_type] = jaccards
    dataset['Jaro' + '_' + string_type] = jaros
    dataset['Longest_com_sub' + '_' + string_type] = lcs
    dataset['Monge-Elkan' + '_' + string_type] = mes
    dataset['SmithWaterman' + '_' + string_type] = sws
    dataset['AffineGap' + '_' + string_type] = afs
    dataset['BagDistance' + '_' + string_type] = bds
    dataset['Cosine_similarity' + '_' + string_type] = coses
    dataset['PartialRatio' + '_' + string_type] = prs
    dataset['Soft_TFIDF' + '_' + string_type] = sfs
    dataset['Editex' + '_' + string_type] = edxs
    dataset['GeneralizedJaccard' + '_' + string_type] = gjs
    dataset['JaroWinkler' + '_' + string_type] = jws
    dataset['Levenshtein' + '_' + string_type] = lws
    dataset['PartialTokenSort' + '_' + string_type] = ptss
    dataset['Ratio' + '_' + string_type] = rats
    dataset['Soundex' + '_' + string_type] = sounds
    dataset['TFIDF' + '_' + string_type] = tfidfs
    dataset['TokenSort' + '_' + string_type] = tss
    dataset['TverskyIndex' + '_' + string_type] = tvs
    dataset['OverlapCoef' + '_' + string_type] = ovs
    dataset['Needleman-Wunsch' + '_' + string_type] = nws
    dataset['Wordnet_sim' + '_' + string_type] = wordnet_sims
    dataset['Word2vec_sim1' + '_' + string_type] = w2vec_sims1
    # dataset['Word2vec_sim1' + '_' + string_type] = w2vec_sims1
    # dataset['Word2vec_sim2' + '_' + string_type] = w2vec_sims2
    # dataset['Word2vec_sim3' + '_' + string_type] = w2vec_sims3
    # dataset['Word2vec_sim4' + '_' + string_type] = w2vec_sims4
    # dataset['Word2vec_sim5' + '_' + string_type] = w2vec_sims5

    return dataset
