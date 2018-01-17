import os.path
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import pickle
from model import pre_train_model


class ael_model():
    def __init__(self, vocab_length, max_seq_len, data_path, sess, encoder_hidden_units=512, decoder_hidden_units=1024,
                 attention_size=512, embedding_size=300, batch_size=64, start_token=2, end_token=3, mode="train",
                 attention="bahdanau", beam_width=1, reuse=False):

        # Class variables
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.data_path = data_path
        self.vocab_len = vocab_length

        # Load pre train model
        pre_trained_model = pre_train_model.pre_train_model(vocab_length, max_seq_len, data_path, sess,
                                                            encoder_hidden_units, decoder_hidden_units,
                                                            attention_size, embedding_size, batch_size,
                                                            start_token, end_token, mode, attention, beam_width,
                                                            initialize_and_load=False, reuse=reuse)

        # Load data from pre model
        self.embeddings_english = pre_trained_model.embeddings_english
        self.projection_layer = pre_trained_model.projection_layer
        self.start_token = pre_trained_model.start_token
        self.print_val_output = pre_trained_model.print_val_output
        self.predict_all_validation_data = pre_trained_model.predict_all_validation_data
        self.answer_question = pre_trained_model.answer_question

        # Transpose embeddings for future calculations
        self.transposed_emb1 = tf.transpose(self.embeddings_english)
        self.transposed_emb2 = tf.transpose(self.embeddings_english)

        # Specify weights path
        self.generator_weights_pre = pre_trained_model.generator_weights_path
        self.generator_weights_ael = data_path + "/generator_weights_ael/"
        self.discriminator_weights_ael = data_path + "/discriminator_weights_ael/"

        # Cells and Layers for discriminator
        encoder_layers = 2
        self.d_encoder_cell_fw = MultiRNNCell([LSTMCell(encoder_hidden_units) for _ in range(encoder_layers)])
        self.d_encoder_cell_bw = MultiRNNCell([LSTMCell(encoder_hidden_units) for _ in range(encoder_layers)])
        discriminator_layers = 2
        self.discriminator_cell_fw = MultiRNNCell([LSTMCell(512) for _ in range(discriminator_layers)])
        self.discriminator_cell_bw = MultiRNNCell([LSTMCell(512) for _ in range(discriminator_layers)])

        # Load train or infer mode
        if mode == "train" or mode == "val_data_out":

            # Load additional data from pre train model
            self.attention_decoder_cell = pre_trained_model.decoder_cell_attn
            self.attention_zero_state = pre_trained_model.attn_zero_state
            self.greedy_score = pre_trained_model.greedy_score
            self.avg_score = pre_trained_model.avg_score
            self.extreme_score = pre_trained_model.extreme_score
            self.subject_val = pre_trained_model.subject_val
            self.content_val = pre_trained_model.content_val
            self.target_output_val = pre_trained_model.target_output_val
            self.validation_outputs = pre_trained_model.validation_outputs
            train_examples = pre_trained_model.train_examples
            validation_examples = pre_trained_model.validation_examples
            generator_params = pre_trained_model.generator_params

            # Build discriminator for train data
            discriminator_params, self.D_loss, self.D_solver, self.G_loss, self.G_solver, _ = self.build_discriminator_modul(
                train_examples, generator_params)

            # Build additional discriminator for validation data
            _, self.D_loss_val, _, self.G_loss_val, _, _ = self.build_discriminator_modul(validation_examples,
                                                                                          generator_params,
                                                                                          reuse=True)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load generator weights
            self.saver_generator = tf.train.Saver(generator_params, max_to_keep=0)
            if pre_train_model.weights_are_valid(self.generator_weights_ael):
                self.saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_ael))
                print("AEL generator weights loaded.")
            elif pre_train_model.weights_are_valid(self.generator_weights_pre):
                self.saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_pre))
                print("Pre-trained generator weights loaded.")

            # Load discriminator weights
            self.saver_discriminator = tf.train.Saver(discriminator_params, max_to_keep=0)
            if pre_train_model.weights_are_valid(self.discriminator_weights_ael):
                self.saver_discriminator.restore(sess, tf.train.latest_checkpoint(self.discriminator_weights_ael))
                print("RL discriminator weights loaded.")

        if mode == "infer":
            # There is only one input at inference time, also redefine start token
            self.batch_size = 1
            self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)

            # Define class variables
            self.placeholders, self.infer_output = pre_trained_model.build_infer_graph(reuse=reuse, beam_width=beam_width)
            self.subject_ph = self.placeholders[0]
            self.content_ph = self.placeholders[1]
            self.len_subject_ph = self.placeholders[2]
            self.len_content_ph = self.placeholders[3]

            # Initialize variables
            sess.run(tf.global_variables_initializer())

        # Create weight paths if not exist
        if not os.path.exists(os.path.dirname(self.generator_weights_ael)):
            os.makedirs(os.path.dirname(self.generator_weights_ael))
        if not os.path.exists(os.path.dirname(self.discriminator_weights_ael)):
            os.makedirs(os.path.dirname(self.discriminator_weights_ael))

    def train(self, sess, train_mode, epochs, rev_word_index, save_weights=False):
        """
        Train function for training discriminator and generator
        :param sess: pre-defined session
        :param train_mode: specifies if discriminator is pre-trained or if generator and discriminator are trained both
        :param epochs: Count of training epochs
        :param rev_word_index: Reverse word index to map output ids to corresponding vocabulary
        :param save_weights: If true, weights are saved to hard disk
        :return:
        """

        # Pre training of discriminator
        if train_mode == "discriminator":
            # For saving metrics for discriminator pre training
            discriminator_stats = []

            print("discriminator pre-train...")

            # Pre train discriminator for count of epochs
            for i in range(epochs):

                # Run session by running D_solver which is the training operation for discriminator. Get also the loss
                # for training data end validation data
                _, dloss, dloss_val = sess.run([self.D_solver, self.D_loss, self.D_loss_val])
                print(i, dloss, dloss_val)
                discriminator_stats.append([dloss, dloss_val])

                # Save weights all five iterations. Also save metrics file
                if i % 5 == 0:
                    pickle.dump(discriminator_stats, open(self.data_path + "dloss_ael_pre_" + str(i) + ".p", "wb"))
                    if save_weights:
                        self.saver_discriminator.save(sess, self.discriminator_weights_ael + "model.ckpt",
                                                      global_step=i)

        # Train mode adversarial for train generator with discriminator
        if train_mode == "adversarial":
            # For saving metrics for adversarial training
            ael_adversarial_stats = []
            print("adversarial training...")

            # Adversarial training for count of epochs
            for i in range(epochs):

                # Run Session by running first D_solver which trains the discriminator, and then G_solver which trains
                # the generator. Also get train loss for discriminator and generator training.
                _, _, dloss, gloss = sess.run([self.D_solver, self.G_solver, self.D_loss, self.G_loss])

                # Save and print stats every epoch
                if i % 1 == 0:
                    greedy_score, avg_score, extreme_score = sess.run(
                        [self.greedy_score, self.avg_score, self.extreme_score])
                    print(i, "greedy:", greedy_score,
                          "avg:", avg_score,
                          "xtreme:", extreme_score,
                          "dloss:", dloss,
                          "gloss:", gloss)
                    ael_adversarial_stats.append([greedy_score, avg_score, extreme_score, dloss, gloss])

                # Save weights for discriminator and generator and save stats file every ten iterations
                if i % 10 == 0:
                    pickle.dump(ael_adversarial_stats, open(self.data_path + "metrics_ael_adv_" + str(i) + ".p", "wb"))
                    self.print_val_output(64, rev_word_index, sess)

                    if save_weights:
                        self.saver_generator.save(sess, self.generator_weights_ael + "model.ckpt", global_step=i)
                        self.saver_discriminator.save(sess, self.discriminator_weights_ael + "model.ckpt",
                                                      global_step=i)

    def build_discriminator_modul(self, train_examples, gen_params, reuse=False):
        """
        Build discriminator with train examples and generator parameters
        :param train_examples: Examples out of train set
        :param gen_params: generator parameters
        :param reuse: If weights already exist, they can be reused with this option
        :return: discriminator_params, D_loss, D_solver, G_loss, G_solver, probabilities for data being fake
        """

        # Unpack subject, content and answers and corresponding length
        subject, len_subject, content, len_content, target_input, target_output, len_target = train_examples

        # Choose best answer per question
        target_output = target_output[:, 0, :]
        len_target = tf.reshape(tf.to_int32(len_target[:, 0]), [-1])

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject, len_subject, content, len_content])[0]
        len_both = len_subject + len_content

        # look up start symbol in word embeddings
        start_symbol_emb = tf.nn.embedding_lookup(self.embeddings_english, self.start_token)

        # Tensor array to save predicted approximative embeddings
        predictive_embs = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.max_seq_len, dynamic_size=False,
                                                       infer_shape=True)

        # Tensor array to save predicted IDs
        predictive_ids = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.max_seq_len, dynamic_size=False,
                                                      infer_shape=True)

        # Tensor array for saving embeddings that have been looked up during generation process
        looked_embs = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.max_seq_len, dynamic_size=False,
                                                   infer_shape=True)

        # Function that will be looped for generating process
        def ael_decoder(i, decoder_input, decoder_state, predictive_embs, predictive_ids, looked_embs):
            # reuse variables for decoder in scope train_decoder/decoder
            with tf.variable_scope("train_decoder", reuse=True):
                with tf.variable_scope("decoder", reuse=True):
                    # Do RNN step, get output of RNN and hidden state
                    decoder_output, decoder_state_new = self.attention_decoder_cell(decoder_input, decoder_state)

                    # Get probabilities for words by using dense layer
                    weights = tf.layers.dense(inputs=decoder_output, units=self.vocab_len, reuse=True,
                                              activation=tf.nn.softmax)

            # Separate weights and process approximate embedding calculation in two steps
            weighted_embs1 = tf.map_fn(self.weight_embedding1, weights[:32, :])
            weighted_embs2 = tf.map_fn(self.weight_embedding2, weights[32:, :])

            # Concat results for approximation process
            weighted_embs = tf.concat([weighted_embs1, weighted_embs2], axis=0)

            # Also receive IDs for the probabilities that were calculated for words
            argmax_weights = tf.stop_gradient(tf.to_int32(tf.argmax(weights, axis=1)))

            # Save predicted approximative embeddings in tensor array
            predictive_embs = predictive_embs.write(i, weighted_embs)

            # Save predicted IDs in tensor array
            predictive_ids = predictive_ids.write(i, argmax_weights)

            # Look up word embeddings for predicted IDs for next RNN step
            lookuped_embs = tf.nn.embedding_lookup(self.embeddings_english, argmax_weights)

            # Save looked up word embeddings in tensor array
            looked_embs = looked_embs.write(i, lookuped_embs)

            return i + 1, lookuped_embs, decoder_state_new, predictive_embs, predictive_ids, looked_embs

        # Start RNN loop
        _, _, _, predictive_embs, predictive_ids, looked_embs = control_flow_ops.while_loop(
            cond=lambda i, _2, _3, _4, _5, _6: i < self.max_seq_len,
            body=ael_decoder,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), start_symbol_emb, self.attention_zero_state, predictive_embs,
                predictive_ids,
                looked_embs))

        # Transform tensor array to tensor and transpose
        predictive_embs = predictive_embs.stack()
        predictive_embs = tf.transpose(predictive_embs, [1, 0, 2])

        # Transform tensor array to tensor and transpose
        predictive_ids = predictive_ids.stack()
        predictive_ids = tf.transpose(predictive_ids, [1, 0])

        # Transform tensor array to tensor and transpose
        looked_embs = looked_embs.stack()
        looked_embs = tf.transpose(looked_embs, [1, 0, 2])

        # Get sequence length of generated sequences
        seq_len_gen = tf.map_fn(self.get_seq_len, predictive_ids, back_prop=False)

        # Look up real answers word embeddings
        target_output_emb = tf.nn.embedding_lookup(self.embeddings_english, target_output)

        # Look up concatted subject and content word embeddings
        sub_cont_emb = tf.nn.embedding_lookup(self.embeddings_english, sub_cont_concat_op)

        # Encode subject and content with discriminators encoder
        with tf.variable_scope("discriminator_encoder", reuse=reuse):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.d_encoder_cell_fw,
                                                                     self.d_encoder_cell_bw,
                                                                     inputs=sub_cont_emb,
                                                                     sequence_length=len_both,
                                                                     dtype=tf.float32,
                                                                     time_major=False)

            # Concat forward and backward states
            forward_states, backward_states = output_states

        # Use discriminator to classify real and fake answers
        with tf.variable_scope("discriminator", reuse=reuse):
            # First make predictions for fake data
            output_fake, _ = tf.nn.bidirectional_dynamic_rnn(self.discriminator_cell_fw,
                                                             self.discriminator_cell_bw,
                                                             inputs=predictive_embs,
                                                             initial_state_fw=forward_states,
                                                             initial_state_bw=backward_states,
                                                             sequence_length=seq_len_gen,
                                                             time_major=False)

            # Concat outputs and states of forward and backward lstm for fake data
            output_fake = tf.concat(output_fake, 2)
            output_fake = tf.map_fn(self.take_last_output, [output_fake, seq_len_gen])[0]

            # Dense layer which shows probability that a sequence is classified as real or fake
            dense_output_fake = tf.layers.dense(inputs=output_fake, units=2)

            # Second make predictions for real data
            output_real, _ = tf.nn.bidirectional_dynamic_rnn(self.discriminator_cell_fw,
                                                             self.discriminator_cell_bw,
                                                             inputs=target_output_emb,
                                                             initial_state_fw=forward_states,
                                                             initial_state_bw=backward_states,
                                                             sequence_length=len_target,
                                                             time_major=False)

            # Concat outputs and states of forward and backward lstm for real data
            output_real = tf.concat(output_real, 2)
            output_real = tf.map_fn(self.take_last_output, [output_real, len_target])[0]

            # Dense layer which shows probability that a sequence is classified as real or fake
            dense_output_real = tf.layers.dense(inputs=output_real, units=2, reuse=True)

        # Cross entropy loss for fake data
        cross_entropy_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([self.batch_size], dtype=tf.int32),
            logits=dense_output_fake
        )

        # Cross entropy loss for real data
        cross_entropy_real = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([self.batch_size], dtype=tf.int32),
            logits=dense_output_real
        )

        # Final discriminator loss by adding loss for real and fake data
        D_loss = tf.reduce_mean(cross_entropy_fake) + tf.reduce_mean(cross_entropy_real)

        # Cross entropy loss for generator
        cross_entropy_fool = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([self.batch_size], dtype=tf.int32),
            logits=dense_output_fake
        )

        # Final generator loss
        G_loss = tf.reduce_mean(cross_entropy_fool)

        # Variables of discriminator
        discriminator_params = [param for param in tf.trainable_variables() if "discriminator" in param.name]

        # Use Adam solver for train discriminator
        with tf.variable_scope("solvers", reuse=reuse):
            D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=discriminator_params)

        # Parameters that are trained for generator
        train_params = [param for param in gen_params if
                        "decoder" in param.name and "dense" not in param.name or "memory" in param.name]

        # Use gradient descent for train generator
        G_solver = tf.train.GradientDescentOptimizer(0.01).minimize(G_loss, var_list=train_params)

        return discriminator_params, D_loss, D_solver, G_loss, G_solver, tf.nn.softmax(dense_output_fake)

    def concat_seqs(self, x):
        """
        Concatting two sequences with given length
        :param x: Packed variables sequence 1, sequence length 1, sequence 2, sequence length 2
        :return: Concatted sequence
        """
        seqs1 = x[0]
        seq_len1 = x[1]
        seqs2 = x[2]
        seq_len2 = x[3]
        concatting = tf.reshape(tf.concat([seqs1[:seq_len1], seqs2[:seq_len2]], axis=0), [1, -1])
        padding = tf.pad(concatting, [[0, 0], [0, self.max_seq_len * 3]])
        padding = tf.reshape(padding[:, :self.max_seq_len * 3], [-1])
        return [padding, 0, 0, 0]

    def weight_embedding1(self, weights):
        """
        One of two word vector multiplications for processing in parallel
        :param weights: probabilities for token in vocabulary
        :return: approximative word vectors
        """
        weighted_matrix = tf.multiply(self.transposed_emb1, weights)
        reduced_matrix = tf.reduce_sum(weighted_matrix, axis=1)
        return reduced_matrix

    def weight_embedding2(self, weights):
        """
        Two of two word vector multiplications for processing in parallel
        :param weights: probabilities for token in vocabulary
        :return: approximative word vectors
        """
        weighted_matrix = tf.multiply(self.transposed_emb2, weights)
        reduced_matrix = tf.reduce_sum(weighted_matrix, axis=1)
        return reduced_matrix

    def get_seq_len(self, seq):
        # Get sequence length of one sequence
        seq_len = tf.to_int32(tf.argmax(tf.to_int32(tf.equal(seq, 3)))) + 1
        return seq_len

    def take_last_output(self, x):
        # gives back last output of a sequence given the sequence length
        output = x[0]
        seq_len = x[1]
        return [output[seq_len - 1], 0]
