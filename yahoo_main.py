import sys
import argparse
from helper import read_data
import tensorflow as tf
from model.ael_model import ael_model as ael_seq2seq
from model.pre_train_model import pre_train_model as pre_train_seq2seq
from model.rl_model import rl_model as rl_train_seq2seq


def main():
    """
    Main function of the project.

    Usage:
    python yahoo_main.py [mode] [model} [-d --data_folder]

    modes: train, infer and beamsearcheval
    models: pre, ael and rl
    data_folder: Folder where data from pre-process step is stored
    """

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="modus: train, beam_search_eval, val_data_out")
    parser.add_argument("model", help="pre, ael or rl")
    parser.add_argument("attention", help="bahdanau or luong")
    parser.add_argument("num_epochs", help="number of epochs for training")
    parser.add_argument("-t", "--train_mode",
                        help="for training with ael or rl: 'discriminator' for pre-train discriminator and 'adversarial' for adversarial training.")
    parser.add_argument("-d", "--data_folder", help="Specify location of data")
    args = parser.parse_args()

    # Load vocab and bpe dict
    _, vocab_list = read_data.load_vocab_and_dict(args.data_folder)

    # Map word index to word text
    rev_word_dict = {index: word for index, word in enumerate(vocab_list)}

    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Load model which is specified in the arguments
    if args.model == "pre":

        # Load Pre-Train-Model
        seq2seq = pre_train_seq2seq(len(vocab_list), max_seq_len=50, sess=sess, data_path=args.data_folder,
                                    mode=args.mode, attention=args.attention)

        # Load one of different modes train, beam_search_eval or val_data_out
        if args.mode == "train":
            seq2seq.train(sess, epochs=int(args.num_epochs), rev_word_index=rev_word_dict, save_weights=True)
        elif args.mode == "beam_search_eval":
            seq2seq.runbeamsearch(sess, 50)
        elif args.mode == "val_data_out":
            seq2seq.predict_all_validation_data(sess, rev_word_dict)
    elif args.model == "ael":

        # Load AEL-Model
        seq2seq = ael_seq2seq(len(vocab_list), max_seq_len=50, sess=sess, data_path=args.data_folder, mode=args.modus)

        # Load mode train or val_data_out
        if args.mode == "train":
            seq2seq.train(sess, train_mode=args.train_mode, epochs=int(args.num_epochs), rev_word_index=rev_word_dict,
                          save_weights=True)
        elif args.mode == "val_data_out":
            seq2seq.predict_all_validation_data(sess, rev_word_dict)
    elif args.model == "rl":

        # Load RL-Model
        seq2seq = rl_train_seq2seq(len(vocab_list), max_seq_len=50, sess=sess, data_path=args.data_folder,
                                   mode=args.mode)

        # Load mode train or val_data_out
        if args.mode == "train":
            seq2seq.train(sess, train_mode=args.train_mode, epochs=int(args.num_epochs), rev_word_index=rev_word_dict,
                          save_weights=True)
        elif args.mode == "val_data_out":
            seq2seq.predict_all_validation_data(sess, rev_word_dict)
    else:
        print(
            "Model unknown. Please use pre (pre-train-model), ael (approximative embedding layer model) or rl (reinforcement-learning model)")
        sys.exit()

if __name__ == "__main__":
    main()
