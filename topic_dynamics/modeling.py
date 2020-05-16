"""
Topic modeling related functionality.
"""
import os
from typing import Tuple

import artm

from .analyzing import save_metadata


def create_batches(directory: str, name: str) -> Tuple[artm.BatchVectorizer, artm.Dictionary]:
    """
    Create the batches and the dictionary from the dataset. Both of them are saved.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: BatchVectorizer and Dictionary.
    """
    print("Creating the batches and the dictionary of the data.")
    batch_vectorizer = artm.BatchVectorizer(
        data_path=directory, data_format="bow_uci", collection_name=name,
        target_folder=os.path.abspath(os.path.join(directory, "batches")))
    dictionary = batch_vectorizer.dictionary
    if not os.path.exists(os.path.abspath(os.path.join(directory, "dictionary.dict"))):
        dictionary.save(os.path.abspath(os.path.join(directory, "dictionary")))
    return batch_vectorizer, dictionary


def define_model(n_topics: int, dictionary: artm.Dictionary, sparse_theta: float,
                 sparse_phi: float, decorrelator_phi: float) -> artm.artm_model.ARTM:
    """
    Define the ARTM model.
    :param n_topics: number of topics.
    :param dictionary: batch vectorizer dictionary.
    :param sparse_theta: sparse theta parameter.
    :param sparse_phi: sparse phi Parameter.
    :param decorrelator_phi: decorellator phi Parameter.
    :return: ARTM model.
    """
    print("Defining the model.")
    topic_names = ["topic_{}".format(i) for i in range(1, n_topics + 1)]
    model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,
                           scores=[artm.PerplexityScore(name="PerplexityScore",
                                                        dictionary=dictionary),
                                   artm.SparsityPhiScore(name="SparsityPhiScore"),
                                   artm.SparsityThetaScore(name="SparsityThetaScore"),
                                   artm.TopicKernelScore(name="TopicKernelScore",
                                                         probability_mass_threshold=0.3),
                                   artm.TopTokensScore(name="TopTokensScore", num_tokens=15)],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name="SparseTheta",
                                                                           tau=sparse_theta),
                                         artm.SmoothSparsePhiRegularizer(name="SparsePhi",
                                                                         tau=sparse_phi),
                                         artm.DecorrelatorPhiRegularizer(name="DecorrelatorPhi",
                                                                         tau=decorrelator_phi)])
    return model_artm


def train_model(model: artm.artm_model.ARTM, n_doc_iter: int, n_col_iter: int,
                dictionary: artm.Dictionary, batch_vectorizer: artm.BatchVectorizer) -> None:
    """
    Train the ARTM model.
    :param model: the trained model.
    :param n_doc_iter: number of document passes.
    :param n_col_iter: number of collection passes.
    :param dictionary: Batch Vectorizer dictionary.
    :param batch_vectorizer: Batch Vectorizer.
    :return: None.
    """
    print("Training the model.")
    model.num_document_passes = n_doc_iter
    model.initialize(dictionary=dictionary)
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=n_col_iter)


def model_topics(output_dir: str, n_topics: int, sparse_theta: float, sparse_phi: float,
                 decorrelator_phi: float, n_doc_iter: int, n_col_iter: int, n_files: int,
                 diffs: bool, batch_size: int = 0) -> None:
    """
    Take the input, create the batches, train the model with the given parameters,
    and saves all metadata.
    :param output_dir: the output directory.
    :param n_topics: number of topics.
    :param sparse_theta: sparse theta parameter.
    :param sparse_phi: sparse phi parameter.
    :param decorrelator_phi: decorellator phi parameter.
    :param n_doc_iter: number of document passes.
    :param n_col_iter: number of collection passes.
    :param n_files: number of the most topical files to be saved for each topic.
    :param batch_size: the number of topics that will be saved together for matrices.
                       0 means that the all topics are saved together.
    :param diffs: True if the topics are modeled on diffs,
                  False if they are modeled on full files.
    :return: None.
    """
    if diffs:
        name = "diffs_dataset"
        tokens_file = os.path.abspath(os.path.join(output_dir, "diffs_tokens.txt"))
        slices_file = os.path.abspath(os.path.join(output_dir, "diffs_slices.txt"))
    else:
        name = "dataset"
        tokens_file = os.path.abspath(os.path.join(output_dir, "tokens.txt"))
        slices_file = os.path.abspath(os.path.join(output_dir, "slices.txt"))
    if batch_size == 0:
        batch_size = n_topics
    batch_vectorizer, dictionary = create_batches(output_dir, name)
    model = define_model(n_topics, dictionary, sparse_theta, sparse_phi, decorrelator_phi)
    train_model(model, n_doc_iter, n_col_iter, dictionary, batch_vectorizer)
    results_dir = os.path.abspath(os.path.join(output_dir, "results"))
    save_metadata(model, results_dir, n_files, batch_size, tokens_file, slices_file)
    print("Topic modeling finished.")
