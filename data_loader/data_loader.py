import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset, Dataset
import sys
from helper import read_data
import re

class data_loader:
    def __init__(self, max_seq_len, batch_size, data_path):
        """
        Init function for defining class variables
        :param max_seq_len: Maximum sequence length for filtering texts and padding
        :param batch_size: batch size for defining output size of every iterator step
        :param data_path: location of preprocessed data
        """

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.data_path = data_path

    def split_string(self, x):
        """
        This function receives a string of numbers and splits the numbers on every whitespace. After that, the numbers
        are casted to int
        :param x: Sequence of numbers as string
        :return: Numbers as tensor as int
        """
        return tf.cast(tf.string_to_number(tf.string_split([x]).values), dtype=tf.int32)

    def split_multi_string(self, x):
        """
        Splits a list of strings on whitespaces and casts them to int
        :param x: list of sequences
        :return: list of int tensors
        """
        # Split best answers on comma
        n_best_answers = tf.string_split([x], delimiter=",")

        # Reformat data to sparse tensor
        n_best_answers = tf.SparseTensorValue(indices=n_best_answers.indices,
                                              values=tf.string_split(n_best_answers.values),
                                              dense_shape=n_best_answers.dense_shape)

        # Get data as sparse tensor
        up_ba = n_best_answers.values

        # Sparse tensor to dense Tensor with padding '0'
        up_ba = tf.sparse_to_dense(up_ba.indices, (up_ba.dense_shape[0], up_ba.dense_shape[1] + 1), up_ba.values, '0')

        # If n_best_answer not empty, convert every answer to int
        up_ba = tf.cond(tf.greater(tf.size(up_ba), 0),
                        lambda: tf.map_fn(lambda s: tf.string_to_number(s, out_type=tf.int32), up_ba, dtype=tf.int32),
                        lambda: tf.string_to_number(up_ba, out_type=tf.int32))

        # Get length of all sequences
        seq_len = tf.argmin(tf.to_int32(tf.not_equal(up_ba, 0)), axis=1)

        # Create filter mask
        idx = tf.less_equal(seq_len, self.max_seq_len - 1)

        # Filter sequences and length
        up_ba = tf.boolean_mask(up_ba, idx)
        up_ba = up_ba[:, 0:tf.cond(tf.greater(tf.size(up_ba), 0), lambda: self.max_seq_len, lambda: 0)]
        up_ba = tf.pad(up_ba, [[0, 0], [0, self.max_seq_len - tf.shape(up_ba)[1]]])
        seq_len = tf.boolean_mask(seq_len, idx) + 1

        # Make datasets
        sequences_nba = Dataset.from_tensors(up_ba)
        len_nba = Dataset.from_tensors(seq_len)

        return Dataset.zip((sequences_nba, len_nba))

    def filter_by_sequence_length(self, sub, len_sub, cont, len_cont, nba, nba_len):
        """
        Filters data by a defined max sequence length

        :param sub: subject sequences
        :param len_sub: length of subject sequences
        :param cont: content sequences
        :param len_cont: length of content sequences
        :param nba: n best answers sequences
        :param nba_len: length of n best answers
        :return: filtered data
        """
        filter_len_sub = tf.less_equal(len_sub, self.max_seq_len)
        filter_len_cont = tf.logical_and(filter_len_sub, tf.less_equal(len_cont, self.max_seq_len))
        filter_len_ans = tf.logical_and(filter_len_cont, tf.greater(tf.size(nba), 0))
        return filter_len_ans

    def process_pad(self, sub, len_sub, cont, len_cont, target_input, target_output, len_nba):
        """
        Prccesses a padding on subject and content
        :param sub: subject sequences
        :param len_sub: length of subject sequences
        :param cont: content sequences
        :param len_cont: length of content sequences
        :param target_input: target input sequences
        :param target_output: target output sequences
        :param len_nba: length of target input/output
        :return:
        """
        sub = tf.reshape(tf.pad(tf.reshape(sub, [1, -1]), [[0, 0], [0, self.max_seq_len]]), [-1])[:self.max_seq_len]
        cont = tf.reshape(tf.pad(tf.reshape(cont, [1, -1]), [[0, 0], [0, self.max_seq_len]]), [-1])[:self.max_seq_len]
        return sub, len_sub, cont, len_cont, target_input, target_output, len_nba

    def get_seq_len_and_join_ba(self, sub, cont, nba_and_len):
        """
        For getting length of subject sequences, content sequences and answer sequences
        :param sub: subject sequences
        :param cont: content sequences
        :param nba_and_len: n best answer ans length
        :return: Dataset with sequences and length
        """

        # Seperate n best answers and length
        nba = nba_and_len[0]
        nba_len = nba_and_len[1]

        # Count sequence length of subject and content
        len_sub = Dataset.from_tensors(tf.size(sub))
        len_cont = Dataset.from_tensors(tf.size(cont))

        # Make dataset from tensors
        sub = Dataset.from_tensors(sub)
        cont = Dataset.from_tensors(cont)
        nba = Dataset.from_tensors(nba)
        nba_len = Dataset.from_tensors(nba_len)

        return Dataset.zip((sub, len_sub, cont, len_cont, nba, nba_len))

    def process_target(self, sub, len_sub, cont, len_cont, nba, nba_len):
        """
        Processes target sequence with setting GO-Symbol in front of target input sequence and EOS-Symbol at the end
        of target output sequence
        :param sub: subject sequences
        :param len_sub: length of subject sequences
        :param cont: content sequences
        :param len_cont: length of content sequences
        :param nba: n best answers sequences
        :param nba_len: length of n best answers
        :return: Dataset with processed target input and output
        """

        def set_eos(nba):
            """
            Receives one answer out of all answers and sets an EOS-Symbol at the end
            :param nba: One answer out of n best answers
            :return: Answer with following EOS-Symbol
            """
            len_nba = tf.argmin(nba, output_type=tf.int32)
            nba = tf.concat([nba[:len_nba], [3], nba[len_nba + 1:]], axis=0)
            return nba

        # Set number 2 (GO) at the beginning of the sequences
        target_input = tf.concat([tf.fill((tf.size(nba_len), 1), 2), nba[:, :-1]], axis=1)

        # Set number 3 (EOS) at the end of the sequences
        target_output = tf.map_fn(set_eos, nba, dtype=tf.int32)

        # Convert function input back to dataset
        sub = Dataset.from_tensors(sub)
        len_sub = Dataset.from_tensors(len_sub)
        cont = Dataset.from_tensors(cont)
        len_cont = Dataset.from_tensors(len_cont)
        target_input = Dataset.from_tensors(target_input)
        target_output = Dataset.from_tensors(target_output)
        len_nba = Dataset.from_tensors(nba_len)

        return Dataset.zip((sub, len_sub, cont, len_cont, target_input, target_output, len_nba))

    def input_fn(self, num_threads=1):
        """
        Receives sequences from defined data path and processes it to feed it into the seq2seq system.
        :param num_threads: Number of parallel operations
        :return: training iterator for iterating over training data and validation iterator for iterating over
        validation data
        """

        # Get subject sequences from file
        sequences_subject = TextLineDataset(self.data_path + "sequences_subject.txt")
        sequences_subject = sequences_subject.map(self.split_string, num_threads=num_threads)

        # Get content sequences from file
        sequences_content = TextLineDataset(self.data_path + "sequences_content.txt")
        sequences_content = sequences_content.map(self.split_string, num_threads=num_threads)

        # Get n best answer sequences from file
        sequences_n_best_answers = TextLineDataset(self.data_path + "sequences_n_best_answers.txt")
        sequences_n_best_answers = sequences_n_best_answers.flat_map(self.split_multi_string)

        # Merge sequences into dataset
        all_data = Dataset.zip((sequences_subject, sequences_content, sequences_n_best_answers))

        # Get length for all sequences
        all_data = all_data.flat_map(self.get_seq_len_and_join_ba)

        # Filter sequence by maximum sequence length
        all_data = all_data.filter(self.filter_by_sequence_length)

        # Process target sequences by setting GO and EOS symbols
        all_data = all_data.flat_map(self.process_target)

        # Pad all sequences to a fixed length
        all_data = all_data.map(self.process_pad)

        # Count of validation data. In the thesis, the value 10000 was used. To work with little data count, the value
        # is set to 100 for now
        n_val_data = 100

        # Make validation data by defining count, repeat data after count
        validation_data = all_data.take(n_val_data).repeat()

        # Process a padding for validation data
        validation_data = validation_data.padded_batch(self.batch_size, padded_shapes=(
            [None], [], [None], [], [None, self.max_seq_len], [None, self.max_seq_len], [None]))

        # Make training data by defining count, repeat data after count
        all_data = all_data.skip(n_val_data)
        all_data = all_data.repeat()

        # Shuffle training data after 10000 iterations
        all_data = all_data.shuffle(10000)

        # Process a padding for training data
        all_data = all_data.padded_batch(self.batch_size, padded_shapes=(
            [None], [], [None], [], [None, self.max_seq_len], [None, self.max_seq_len], [None]))

        # Make iterators for iterating over training and validation data
        training_iterator = all_data.make_one_shot_iterator()
        validation_iterator = validation_data.make_one_shot_iterator()

        return training_iterator, validation_iterator
