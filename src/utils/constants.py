import gensim.downloader

from datasets.BIOS_dataset import BIOSDataset
from datasets.XING_dataset import XINGDataset
from src.fairness_methods.fairness_method_CIFRank import CIFRank
from src.fairness_methods.fairness_method_FAIR import FAIRRanking
from src.fairness_methods.fairness_method_LFR import LearningFairRepresentations
from src.fairness_methods.fairness_method_iFair import iFairRanking
from src.fairness_methods.fairness_method import FairnessMethod

word2vec_google_news = gensim.downloader.load('word2vec-google-news-300')


fairness_method_mapping = {
    'CIFRank': CIFRank,
    'IFAIR': iFairRanking,
    'LFR': LearningFairRepresentations,
    'FAIR': FAIRRanking,
    'no_method': FairnessMethod
}

dataset_mapping = {
    'BIOS': BIOSDataset,
    'XING': XINGDataset,
}