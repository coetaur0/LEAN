"""
Preprocess the SNLI dataset and word embeddings to be used by the LEAN model.
"""
# Aurelien Coet, 2018.

import os
import pickle
import argparse
import fnmatch
import json

from lean.data import Preprocessor


def preprocess_SNLI_data(inputdir,
                         embeddings_file,
                         w2h_file,
                         lear_file,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         labeldict={},
                         bos=None,
                         eos=None):
    """
    Preprocess the data from the SNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Compute lexical entailment between the words in the premises and
    hypotheses.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        w2h_file: The path to the file containing pre-trained Word2Hyp word
            embeddings that must be used to compute lexical entailment
            between words in the data.
        lear_file: The path to the file containing pre-trained LEAR word
            embeddings that must be used to compute lexical entailment
            between words in the data.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_train.txt"):
            train_file = file
        elif fnmatch.fnmatch(file, "*_dev.txt"):
            dev_file = file
        elif fnmatch.fnmatch(file, "*_test.txt"):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                labeldict=labeldict,
                                bos=bos,
                                eos=eos)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("* Preparing the data...")
    prepared_data = preprocessor.prepare_data(data)

    print("* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(prepared_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)
    
    print("* Building Word2Hyp embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(w2h_file)
    with open(os.path.join(targetdir, "w2h_embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    print("* Building LEAR embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(lear_file)
    with open(os.path.join(targetdir, "lear_embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    default_config = "../../config/preprocessing/snli_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the SNLI dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing SNLI"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess_SNLI_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
        os.path.normpath(os.path.join(script_dir, config["w2h_file"])),
        os.path.normpath(os.path.join(script_dir, config["lear_file"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"]
    )
