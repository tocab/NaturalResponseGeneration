import os.path
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import TrainingHelper, dynamic_decode, BahdanauAttention, BasicDecoder, \
    AttentionWrapper, BeamSearchDecoder, tile_batch, LuongAttention
from tensorflow.python.layers import core as layers_core
from helper import read_data
import sys
import re
import pickle
from metrics import metrics
from data_loader import data_loader


class pre_train_model():
    def __init__(self, vocab_length, max_seq_len, data_path, sess, encoder_hidden_units=512, decoder_hidden_units=1024,
                 attention_size=512, embedding_size=300, batch_size=64, start_token=2, end_token=3, mode="train",
                 attention="bahdanau", beam_width=1, initialize_and_load=True, reuse=False):

        # Define class variables
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.embedding_size = embedding_size
        self.attention_size = attention_size if attention == "bahdanau" else attention_size * 2
        self.vocab_len = vocab_length
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.end_token = end_token
        self.data_path = data_path
        self.attention = attention
        self.beam_width = beam_width
        self.generator_weights_path = data_path + "/generator_weights_pre/"

        # Initialize Variable for word embeddings
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            self.embeddings_english = tf.get_variable("embeddings",
                                                      initializer=tf.random_uniform([vocab_length, embedding_size],
                                                                                    -1.0, 1.0),
                                                      dtype=tf.float32)

        # Define layer size of encoder and decoder
        encoder_layers = 2
        decoder_layers = 2

        # Initialize encoder forward and backward cell
        self.encoder_cell_forward = MultiRNNCell([LSTMCell(self.encoder_hidden_units) for _ in range(encoder_layers)])
        self.encoder_cell_backward = MultiRNNCell([LSTMCell(self.encoder_hidden_units) for _ in range(encoder_layers)])

        # Initialize decoder cell
        self.decoder_cell = MultiRNNCell([LSTMCell(self.decoder_hidden_units) for _ in range(decoder_layers)])

        # Initialize dense layer
        self.projection_layer = layers_core.Dense(self.vocab_len, use_bias=True)

        # Load train or infer mode
        if mode == "train" or mode == "val_data_out":

            # Load train and validation examples
            self.train_examples, self.validation_examples = self.initialize_train_val_examples(self.max_seq_len,
                                                                                               self.batch_size,
                                                                                               self.data_path)

            # Unpack validation examples and save a class variable for printing them in training process
            self.subject_val, _, self.content_val, _, _, target_output, _ = self.validation_examples
            self.target_output_val = target_output[:, 0, :]

            # Build train graph with train examples as input
            target_output, self.final_outputs, final_seq_len, self.generator_params, self.decoder_cell_attn, self.attn_zero_state, self.beam_outputs, self.beam_out_len = self.build_train_graph(
                self.train_examples)

            # Build MLE modul for train graph
            self.build_mle_modul(target_output, self.final_outputs.rnn_output)

            # Build validation graph with validation examples and beam size. Get back metrics and outputs
            self.avg_score, self.greedy_score, self.extreme_score, self.val_seq_len, self.validation_outputs = self.build_validation_graph(
                self.validation_examples, beam_width=self.beam_width)

            # If variables should be initialized and weights should be loaded
            if initialize_and_load:

                # Initialize variables
                sess.run(tf.global_variables_initializer())

                # Load weights
                self.saver_generator = tf.train.Saver(self.generator_params)
                if weights_are_valid(self.generator_weights_path):
                    self.saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_path))
                    print("weights loaded.")

        if mode == "infer":
            # There is only one input at inference time, also redefine start token
            self.batch_size = 1
            self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)

            # Define class variables
            self.placeholders, self.infer_output = self.build_infer_graph(reuse=reuse, beam_width=beam_width)
            self.subject_ph = self.placeholders[0]
            self.content_ph = self.placeholders[1]
            self.len_subject_ph = self.placeholders[2]
            self.len_content_ph = self.placeholders[3]

            # Initialize variables
            sess.run(tf.global_variables_initializer())

        # Create weight paths if not exist, create it
        if not os.path.exists(os.path.dirname(self.generator_weights_path)):
            os.makedirs(os.path.dirname(self.generator_weights_path))

    def build_infer_graph(self, beam_width=1, reuse=False):
        """
        Build graph for infering unseen data from the graph
        :param beam_width: Define beam width
        :return: Placeholders for inputing data and outputs of beam search
        """

        # Placeholders for Encoder
        subject_ph = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='subject')
        content_ph = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='content')
        len_subject_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='sub_len')
        len_content_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='cont_len')

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject_ph, len_subject_ph, content_ph, len_content_ph])[0]

        # Also concat length by adding them
        len_both = len_subject_ph + len_content_ph

        # Build initial graph with concatted subject and content and sequence length. Get back decoder cell and
        # attention zero state
        decoder_cell, attn_zero_state = self.build_initial_graph(sub_cont_concat_op, len_both, beam_width=beam_width,
                                                                 reuse=reuse)

        # Build beam search decoder
        decoder = BeamSearchDecoder(decoder_cell, self.embeddings_english, self.start_token, self.end_token,
                                    attn_zero_state, beam_width, output_layer=self.projection_layer)

        # Define variable scope train decoder to initialize the decoder with dynamic decode
        with tf.variable_scope("train_decoder", reuse=reuse):
            outputs, _, _ = dynamic_decode(decoder, output_time_major=False, maximum_iterations=self.max_seq_len)

        # Transform beam outputs for readable output
        beam_outputs = tf.transpose(outputs.predicted_ids, [2, 0, 1])

        return [subject_ph, content_ph, len_subject_ph, len_content_ph], beam_outputs

    def build_validation_graph(self, validation_data, beam_width=1):
        """
        Build same graph as training graph with validation data so test metrics with unseen data
        :param validation_data: Validation data input
        :param beam_width: beam width parameter
        :return:
        """

        # Unpack subject, content and answers and corresponding length
        subject, len_subject, content, len_content, target_input, target_output, len_target = validation_data

        # Choose best answer per question
        target_output = target_output[:, 0, :]
        len_target = tf.reshape(tf.to_int32(len_target[:, 0]), [-1])

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject, len_subject, content, len_content])[0]
        len_both = len_subject + len_content

        # Build initial graph with concatted subject and content and sequence length. Get back decoder cell and
        # attention zero state
        decoder_cell, attn_zero_state = self.build_initial_graph(sub_cont_concat_op, len_both, reuse=True,
                                                                 beam_width=beam_width)

        # Build beam search decoder
        decoder = BeamSearchDecoder(decoder_cell, self.embeddings_english, self.start_token, self.end_token,
                                    attn_zero_state, beam_width, output_layer=self.projection_layer)

        # Define variable scope train decoder to initialize the decoder with dynamic decode. Reuse variables from scope
        # because it has already been defined for train graph
        with tf.variable_scope("train_decoder", reuse=True):
            outputs, _, val_seq_len = dynamic_decode(decoder, output_time_major=False,
                                                     maximum_iterations=self.max_seq_len)

        # Take only first output of beam search
        validation_outputs = tf.transpose(outputs.predicted_ids, [2, 0, 1])
        validation_outputs = tf.reshape(validation_outputs[0, :, :], [self.batch_size, -1])
        val_seq_len = tf.transpose(val_seq_len)
        val_seq_len = tf.reshape(val_seq_len[0, :], [-1])

        # Calculate metric scores
        avg_score, greedy_score, extreme_score = self.metrics_module(validation_outputs, val_seq_len,
                                                                     target_output, len_target)

        return avg_score, greedy_score, extreme_score, val_seq_len, validation_outputs

    def metrics_module(self, predicted_output_ids, beam_len_output, target_output, len_target):
        """
        Calculate metrics with predicted outputs and real outputs
        :param predicted_output_ids: Predicted outputs
        :param beam_len_output: Predicted outputs length
        :param target_output: Real output
        :param len_target: Real output length
        :return:
        """

        # Calculate embedding average score
        avg_score = metrics.embedding_average_score(predicted_output_ids, beam_len_output, target_output,
                                                    len_target, self.embeddings_english)
        avg_score = tf.reduce_mean(avg_score)

        # Calculate greedy embedding score
        greedy_score = metrics.greedy_embedding_matching(predicted_output_ids, beam_len_output, target_output,
                                                         len_target, self.embeddings_english)
        greedy_score = tf.reduce_mean(greedy_score)

        # Calculate vector extrema score
        extreme_score = metrics.vector_extrema(predicted_output_ids, beam_len_output, target_output, len_target,
                                               self.embeddings_english)
        extreme_score = tf.reduce_mean(extreme_score)

        return avg_score, greedy_score, extreme_score

    def answer_question(self, sess, question, content, bpe_dict, vocab_list):

        rev_word_dict = {index: word for index, word in enumerate(vocab_list)}
        questions = [question, content]

        questions_processed, input_length = read_data.process_questions(questions, bpe_dict, vocab_list)

        fd = {self.subject_ph: [questions_processed[0]],
              self.content_ph: [questions_processed[1]],
              self.len_subject_ph: [input_length[0]],
              self.len_content_ph: [input_length[1]]}

        preds = sess.run(self.infer_output, feed_dict=fd)

        candidates = []
        for i in range(len(preds)):
            candidate = " ".join([rev_word_dict[key] if key in rev_word_dict else "<ERROR>" for key in preds[i][0]])
            candidate = candidate.replace("@@ ", "")
            candidate = re.sub(r"<EOS>.+", "", candidate)
            candidates.append(candidate)

        # candidates = "<br \>".join(candidates)
        candidates = candidates[0]
        return candidates

    def train(self, sess, epochs, rev_word_index, save_weights=False):
        """
        Train function for starting training process of the pre train model
        :param sess: pre-defined session
        :param epochs: Count of training epochs
        :param rev_word_index: Reverse word index to map output ids to corresponding vocabulary
        :param save_weights: If true, weights are saved to hard disk
        :return:
        """

        # List for saving metrics during training process
        save_metrics = []

        print('Starting training for pre train model')
        # read the data from queue shuffled
        for i in range(epochs):

            # Run train operation every epoch to update weights with defined loss
            sess.run(self.mle_train_op)

            # Print scores every 100 epochs and append them to list
            if i % 100 == 0:
                greedy_score, avg_score, extreme_score, loss = sess.run(
                    [self.greedy_score, self.avg_score, self.extreme_score, self.loss])
                print(i, greedy_score, avg_score, extreme_score, loss)
                save_metrics.append([greedy_score, avg_score, extreme_score, loss])

            # Save metrics every 10000 epochs
            if i % 10000 == 0:
                pickle.dump(save_metrics, open(self.data_path + "metrics_" + str(i) + ".p", "wb"))

            # Print example outputs every 1000 epochs and save weights
            if i % 1000 == 0:
                self.print_val_output(5, rev_word_index, sess)
                if save_weights:
                    self.saver_generator.save(sess, self.generator_weights_path + "model.ckpt", global_step=i)

    def initialize_train_val_examples(self, max_seq_len, batch_size, data_path):
        """
        Initialize iterators for train and validation data and fetch them for iterating over epochs
        :param max_seq_len: Maximum sequence length to filter data in data loader
        :param batch_size: Batch size for data loader
        :param data_path: Data path where pre processed subject, content and answer data is located
        :return:
        """

        # Create data loader
        data_loader_1 = data_loader.data_loader(max_seq_len, batch_size, data_path)

        # Receive training and validation iterator
        training_iterator, validation_iterator = data_loader_1.input_fn()

        # Receive train and validation examples to give them into the graph
        train_examples = training_iterator.get_next()
        validation_examples = validation_iterator.get_next()

        return train_examples, validation_examples

    def predict_all_validation_data(self, sess, rev_word_index):
        """
        Function to predict answers for all validation data
        :param sess: pre-defined session
        :param rev_word_index: Reverse word index to map output ids to corresponding vocabulary
        :return:
        """

        # Define count of validation data and the count of iterations based on it
        n_val_data = 10000
        iterations = int(round(n_val_data / self.batch_size))

        print("Start predicting all", n_val_data, "data points from validation data in", iterations, "iterations")

        # Print all validation data outputs
        for i in range(iterations):
            self.print_val_output(self.batch_size, rev_word_index, sess)

    def runbeamsearch(self, sess, max_beam_width):
        """
        Function to evaluate different beam width parameters
        :param sess: pre-defined session
        :param max_beam_width: Maximum beam width where to stop evaluation
        :return:
        """

        # List for saving metrics when evaluating beam size
        save_metrics = []

        print('Start evaluating beam search')
        # Get training and validation examples
        train_examples, validation_examples = self.initialize_train_val_examples(self.max_seq_len,
                                                                                 self.batch_size,
                                                                                 self.data_path)

        # Build train graph with training examples for receiving generator params
        _, _, _, generator_params, _, _, _, _ = self.build_train_graph(train_examples)

        # With generator params, build a new validation graph with increasing beam size for every epoch
        for i in range(1, max_beam_width + 1):

            # Build validation graph with validation examples and beam size for this epoch
            avg_score, greedy_score, extreme_score, _, self.validation_outputs = self.build_validation_graph(
                validation_examples,
                beam_width=i)

            # Initialize graph
            sess.run(tf.global_variables_initializer())

            # Load saved generator weights if exist
            saver_generator = tf.train.Saver(generator_params)
            if os.path.isdir(self.generator_weights_path):
                saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_path))

            # Get metrics for validation examples with this epoch
            avg_score1, greedy_score1, extreme_score1 = sess.run([avg_score, greedy_score, extreme_score])

            # Print and save metrics
            print("beam size", i, "avg:", avg_score1, "greedy:", greedy_score1, "xtreme:", extreme_score1)
            save_metrics.append([greedy_score1, avg_score1, extreme_score1])
            pickle.dump(save_metrics, open(self.data_path + "metrics_beam_" + str(i) + ".p", "wb"))

    def build_train_graph(self, train_examples):
        """
        Building train graph with train examples
        :param train_examples: Examples from train data
        :return: Predicted outputs, parameters of generator, decoder cell, attention zero state
        """

        # Unpack subject, content and answers and corresponding length
        subject, len_subject, content, len_content, target_input, target_output, len_target = train_examples

        # Choose best answer per question
        target_input = target_input[:, 0, :]
        target_output = target_output[:, 0, :]
        len_target = tf.to_int32(len_target[:, 0])

        # Look up word vectors for decoder input
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_english, target_input)

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject, len_subject, content, len_content])[0]
        len_both = len_subject + len_content

        # Load inital graph twice, one for train and another for output with beam decoder
        decoder_cell, attn_zero_state = self.build_initial_graph(sub_cont_concat_op, len_both)
        decoder_cell_beam, attn_zero_state_beam = self.build_initial_graph(sub_cont_concat_op, len_both, reuse=True,
                                                                           beam_width=self.beam_width)

        # Make train decoder
        helper = TrainingHelper(decoder_inputs_embedded, len_target, time_major=False)
        decoder = BasicDecoder(decoder_cell, helper, attn_zero_state, output_layer=self.projection_layer)

        # Make beam search decoder
        beam_search_decoder = BeamSearchDecoder(decoder_cell_beam, self.embeddings_english, self.start_token,
                                                self.end_token,
                                                attn_zero_state_beam, self.beam_width,
                                                output_layer=self.projection_layer)

        # Define variable scope train decoder to initialize the train decoder and beam search decoder
        # with dynamic decode
        with tf.variable_scope("train_decoder"):
            final_outputs, final_state, final_seq_len = dynamic_decode(decoder, output_time_major=False)
        with tf.variable_scope("train_decoder", reuse=True):
            beam_outputs, _, beam_out_len = dynamic_decode(beam_search_decoder, output_time_major=False,
                                                           maximum_iterations=self.max_seq_len)

        # Output of train decoder
        final_outputs_max_len = tf.shape(final_outputs.sample_id)[1]
        target_output = target_output[:, :final_outputs_max_len]

        # Output of beam search decoder
        beam_outputs = tf.transpose(beam_outputs.predicted_ids, [2, 0, 1])
        beam_outputs = tf.reshape(beam_outputs[0, :, :], [self.batch_size, -1])
        beam_out_len = tf.transpose(beam_out_len)
        beam_out_len = tf.reshape(beam_out_len[0, :], [-1])

        # Get generator parameters
        generator_params = [param for param in tf.trainable_variables() if "discriminator" not in param.name]

        return target_output, final_outputs, final_seq_len, generator_params, decoder_cell, attn_zero_state, beam_outputs, beam_out_len

    def build_mle_modul(self, dec_output, rnn_output):
        """
        Modul for train the system with cross entropy
        :param dec_output: real output values
        :param rnn_output: rnn output of generated data
        :return:
        """

        # Cross entropy Loss between real and predicted output
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=dec_output,
            logits=rnn_output
        )

        # Mask Padding by weighting padding values with zero
        target_weights = tf.cast(tf.greater(dec_output, 0), tf.float32)

        # Multiply masked padding to loss and reduce sum. Divide with batch size to receive final loss
        self.loss = tf.reduce_sum(self.cross_entropy * target_weights) / self.batch_size

        # Train with adam optimizer
        with tf.variable_scope("Adam_solver"):
            self.mle_train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def build_initial_graph(self, encoder_input, len_both, beam_width=1, reuse=False):
        """
        Building initial graph with input for encoder and a fixed beam width
        :param encoder_input: Input that will be processed from encoder
        :param len_both: length of input
        :param beam_width: beam width
        :param reuse: If this graph already exists and should be reused, like in validation graph
        :return: decoder cell and attention zero state
        """
        # look up embeddings for input sequence
        encoder_subject_embedded = tf.nn.embedding_lookup(self.embeddings_english, encoder_input)

        # Define variable scope for LSTM Encoder
        with tf.variable_scope("LSTM_Encoder_subject", reuse=reuse):

            # Create a bidirectional lstm with encoder forward cell and encoder backward cell defined as class variables
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_forward,
                                                                     self.encoder_cell_backward,
                                                                     inputs=encoder_subject_embedded,
                                                                     sequence_length=len_both,
                                                                     dtype=tf.float32,
                                                                     time_major=False)

            # Concat outputs and states of forward and backward lstm
            outputs = tf.concat(outputs, 2)

            # Unpack forward state and backward state vom output states
            forward_states, backward_states = output_states

            # List vor c states (lstm cell state) and h states (lstm hidden state)
            c_states = []
            h_states = []

            # Rearrange state to give them into decoder: concat forward and backward c and h state
            for i, state in enumerate(forward_states):
                c_forward = state[0]
                c_backward = backward_states[i][0]

                c_state = tf.concat([c_forward, c_backward], 1)
                c_states.append(c_state)

                h_forward = state[1]
                h_backward = backward_states[i][1]

                h_state = tf.concat([h_forward, h_backward], 1)
                h_states.append(h_state)

            # List for saving states as tuple
            state_tuples = []

            # Saving states as LSTMStateTuple
            for i, c_state in enumerate(c_states):
                state_tuple = LSTMStateTuple(c_state, h_states[i])
                state_tuples.append(state_tuple)

            # Cast list to tuple
            state_tuples = tuple(state_tuples)

        # multiply rnn output if beam search is used
        outputs = tile_batch(outputs, beam_width)
        len_both = tile_batch(len_both, beam_width)
        encoder_final_state = tile_batch(state_tuples, beam_width)

        # Choose luong or bahdanau attention
        if self.attention == "luong":
            AttentionBuilder = LuongAttention
        elif self.attention == "bahdanau":
            AttentionBuilder = BahdanauAttention
        else:
            print("Attention mechanism not found.")
            sys.exit()

        # Define variable scope for attention mechanism
        with tf.variable_scope("Attention", reuse=reuse):

            # Create an attention mechanism
            attention_mechanism = AttentionBuilder(self.attention_size, outputs, len_both)

            # Create Attention wrapper with decoder cell
            decoder_cell = AttentionWrapper(self.decoder_cell, attention_mechanism, self.attention_size)

        # Create zero state of decoder cell with specified batch size and beam width
        attn_zero_state = decoder_cell.zero_state(batch_size=self.batch_size * beam_width, dtype=tf.float32)

        # Set cell state to final decoder cell state
        attn_zero_state = attn_zero_state.clone(cell_state=encoder_final_state)

        return decoder_cell, attn_zero_state

    def concat_seqs(self, x):
        """
        Concatting two sequences with given length
        :param x: Packed variables sequence 1, sequence length 1, sequence 2, sequence length 2
        :return: Concatted sequence
        """

        # Unpack variables
        seqs1 = x[0]
        seq_len1 = x[1]
        seqs2 = x[2]
        seq_len2 = x[3]

        # Concat sequences
        concatting = tf.reshape(tf.concat([seqs1[:seq_len1], seqs2[:seq_len2]], axis=0), [1, -1])

        # Pad new sequence with maximum sequence length
        padding = tf.pad(concatting, [[0, 0], [0, self.max_seq_len * 3]])

        # Reshape sequence
        padding = tf.reshape(padding[:, :self.max_seq_len * 3], [-1])

        return [padding, 0, 0, 0]

    def print_val_output(self, count, rev_word_index, sess):
        """
        For printing validation output
        :param count: Deciding how many validation data points will be printed
        :param rev_word_index: Reverse word index to map output ids to corresponding vocabulary
        :param sess: pre-defined session
        :return:
        """

        # Start session and receive validation subjects, validation content, real output answers and predicted answers
        subject_val, content_val, tar_val, val_output = sess.run(
            [self.subject_val, self.content_val, self.target_output_val, self.validation_outputs])

        print("sample:")
        # Print samples
        for j in range(count):
            # Map numbers to words in reverse word index dict
            question_text = " ".join(
                [rev_word_index[key] if key in rev_word_index else "" for key in subject_val[j]])
            content_text = " ".join(
                [rev_word_index[key] if key in rev_word_index else "" for key in content_val[j]])
            answer_text = " ".join(
                [rev_word_index[key] if key in rev_word_index else "" for key in val_output[j]])
            gold_answer_text = " ".join(
                [rev_word_index[key] if key in rev_word_index else "" for key in tar_val[j]])

            # Replace gaps that are produces by bpe
            answer_text = answer_text.replace("@@ ", "")
            question_text = question_text.replace("@@ ", "")
            content_text = content_text.replace("@@ ", "")
            gold_answer_text = gold_answer_text.replace("@@ ", "")

            # Delete <PAD>
            answer_text = answer_text.replace("<PAD>", "")
            question_text = question_text.replace("<PAD>", "")
            content_text = content_text.replace("<PAD>", "")
            gold_answer_text = gold_answer_text.replace("<PAD>", "")

            # Print data
            print("Question:", question_text)
            print("Content:", content_text)
            print("Answer:", answer_text)
            print("Gold Answer:", gold_answer_text)
            print("--")


def weights_are_valid(data_path):
    """
    Checks if the weight path exists and if valid files are saved
    :param data_path: Data path where pre processed subject, content and answer data is located
    :return: Boolean which is True when saved weights are valid
    """

    # If path exists
    if os.path.isdir(data_path):

        # list all files in that path
        files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

        # If no files are found, return False
        if len(files) == 0:
            return False

        # If "checkpoint" file not found, return False
        if "checkpoint" not in files:
            return False

        # Search for files data, index and meta
        data_files = [file for file in files if "data" in file]
        index_files = [file for file in files if "index" in file]
        meta_files = [file for file in files if "meta" in file]

        # If one of the files not found in the path, return False
        if len(data_files) == 0 or len(index_files) == 0 or len(meta_files) == 0:
            return False

        return True

    return False
