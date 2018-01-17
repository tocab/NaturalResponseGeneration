import os.path
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import pickle
from model import pre_train_model
import sys


class rl_model():
    def __init__(self, vocab_length, max_seq_len, data_path, sess, encoder_hidden_units=512, decoder_hidden_units=1024,
                 attention_size=512, embedding_size=300, batch_size=64, start_token=2, end_token=3, mode="train",
                 attention="bahdanau", beam_width=1, reuse=False):

        # Class variables
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.data_path = data_path

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

        # Specify weights path
        self.generator_weights_pre = pre_trained_model.generator_weights_path
        self.generator_weights_rl = data_path + "/generator_weights_rl/"
        self.discriminator_weights_rl = data_path + "/discriminator_weights_rl/"

        # Cells and Layers for discriminator
        encoder_layers = 2
        self.d_encoder_cell_fw = MultiRNNCell([LSTMCell(encoder_hidden_units) for _ in range(encoder_layers)])
        self.d_encoder_cell_bw = MultiRNNCell([LSTMCell(encoder_hidden_units) for _ in range(encoder_layers)])
        discriminator_layers = 2
        self.discriminator_cell_fw = MultiRNNCell([LSTMCell(512) for _ in range(discriminator_layers)])
        self.discriminator_cell_bw = MultiRNNCell([LSTMCell(512) for _ in range(discriminator_layers)])

        # Load train or infer mode
        if mode == "train" or mode == "val_data_out":

            # Load data from pre train model
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
            beam_output = pre_trained_model.beam_outputs
            beam_out_len = pre_trained_model.beam_out_len
            val_output = pre_trained_model.validation_outputs
            val_seq_len = pre_trained_model.val_seq_len
            generator_params = pre_trained_model.generator_params
            final_outputs = pre_trained_model.final_outputs

            # Build discriminator for train data
            discriminator_params, self.D_pretrain_loss, self.D_solver, _ = self.build_discriminator(train_examples,
                                                                                                    beam_output,
                                                                                                    beam_out_len)
            # Build additional discriminator for validation data
            _, self.D_validation_loss, _, self.output_discriminator = self.build_discriminator(validation_examples,
                                                                                               val_output,
                                                                                               val_seq_len,
                                                                                               reuse=True)

            # Build Reinforcement module with train examples and predicted output
            _ = self.build_rl_module(train_examples, final_outputs, generator_params, beam_output)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load generator weights
            self.saver_generator = tf.train.Saver(generator_params, max_to_keep=0)
            if pre_train_model.weights_are_valid(self.generator_weights_rl):
                self.saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_rl))
                print("RL generator weights loaded.")
            elif pre_train_model.weights_are_valid(self.generator_weights_pre):
                self.saver_generator.restore(sess, tf.train.latest_checkpoint(self.generator_weights_pre))
                print("Pre-trained generator weights loaded.")

            # Load discriminator weights
            self.saver_discriminator = tf.train.Saver(discriminator_params, max_to_keep=0)
            if pre_train_model.weights_are_valid(self.discriminator_weights_rl):
                self.saver_discriminator.restore(sess, tf.train.latest_checkpoint(self.discriminator_weights_rl))
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
        if not os.path.exists(os.path.dirname(self.generator_weights_rl)):
            os.makedirs(os.path.dirname(self.generator_weights_rl))
        if not os.path.exists(os.path.dirname(self.discriminator_weights_rl)):
            os.makedirs(os.path.dirname(self.discriminator_weights_rl))

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

        if train_mode == "discriminator":
            # For saving metrics for discriminator pre training
            discriminator_stats = []

            print("discriminator pre-training ...")

            # Pre train discriminator for count of epochs
            for i in range(epochs):

                # Run session by running D_pretrain_loss which is the training operation for discriminator. Get also
                # the loss for training data end validation data
                d_loss_train, _, d_loss_val = sess.run([self.D_pretrain_loss, self.D_solver, self.D_validation_loss])

                # Save and print loss all five iterations
                if i % 5 == 0:
                    discriminator_stats.append([d_loss_train, d_loss_val])
                    print(i, d_loss_train, d_loss_val)

                # Save metrics file all 1000 iterations
                if i % 1000 == 0:
                    pickle.dump(discriminator_stats, open(self.data_path + "metrics_rl_discriminator_" + str(i) + ".p", "wb"))

                # Save weights all 1000 iterations
                if save_weights and i % 1000 == 0:
                    self.saver_discriminator.save(sess, self.discriminator_weights_rl + "model.ckpt", global_step=i)

        # Train mode adversarial for train generator with discriminator
        elif train_mode == "adversarial":
            # For saving metrics for adversarial training
            ael_adversarial_stats = []

            print("adversarial training...")

            # Adversarial training for count of epochs
            for i in range(epochs):

                # Run Session by running first D_solver which trains the discriminator, and then rl_gen_train_op which
                # trains the generator. Also get train loss for discriminator and generator training.
                _, _, dloss, gloss = sess.run([self.D_solver, self.rl_gen_train_op, self.D_pretrain_loss, self.loss_rl])

                # Save and print metrics all ten iterations
                if i % 10 == 0:
                    greedy_score, avg_score, extreme_score = sess.run(
                        [self.greedy_score, self.avg_score, self.extreme_score])
                    print(i, greedy_score, avg_score, extreme_score, dloss, gloss)
                    ael_adversarial_stats.append([greedy_score, avg_score, extreme_score, dloss, gloss])

                # Save metrics file all 1000 iterations
                if i % 1000 == 0:
                    pickle.dump(ael_adversarial_stats, open(self.data_path + "metrics_rl_adv_" + str(i) + ".p", "wb"))

                # Print validation data and save weights all 500 iterations
                if i % 500 == 0:
                    self.print_val_output(5, rev_word_index, sess)
                    if save_weights:
                        self.saver_generator.save(sess, self.generator_weights_rl + "model.ckpt", global_step=i)
                        self.saver_discriminator.save(sess, self.discriminator_weights_rl + "model.ckpt", global_step=i)
        else:
            print("Mode unknown. Please choose between 'discriminator' and 'adversarial'")
            sys.exit()

    def build_discriminator(self, train_examples, beam_outputs, beam_out_len, reuse=False):
        """
        Build discriminator with train examples and beam outputs of generator
        :param train_examples: Examples out of train set
        :param beam_outputs: Generated examples of beam search
        :param beam_out_len: Length of generated output
        :param reuse: If weights already exist, they can be reused with this option
        :return:
        """

        # Unpack subject, content and answers and corresponding length
        subject, len_subject, content, len_content, target_input, target_output, len_target = train_examples

        # Choose best answer per question
        target_output = target_output[:, 0, :]
        len_target = len_target[:, 0]

        # Look up real answers word embeddings
        target_output_emb = tf.nn.embedding_lookup(self.embeddings_english, target_output)

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject, len_subject, content, len_content])[0]
        len_both = len_subject + len_content

        # Look up predicted answers word embeddings
        generated_examples = tf.nn.embedding_lookup(self.embeddings_english, beam_outputs)

        # Encode subject and content with discriminators encoder
        sub_cont_emb = tf.nn.embedding_lookup(self.embeddings_english, sub_cont_concat_op)
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
                                                             inputs=generated_examples,
                                                             sequence_length=beam_out_len,
                                                             initial_state_fw=forward_states,
                                                             initial_state_bw=backward_states,
                                                             time_major=False)

            # Concat outputs and states of forward and backward lstm
            output_fake = tf.concat(output_fake, 2)

            # Take output of last time step
            cat_idx = tf.stack([tf.range(0, tf.shape(output_fake)[0]), beam_out_len - 1], axis=1)
            output_fake = tf.gather_nd(output_fake, cat_idx)

            # Dense layer which shows probability that a sequence is classified as real or fake
            dense_output_fake = tf.layers.dense(inputs=output_fake, units=2, reuse=reuse)

            # Second make predictions for real data
            output_real, _ = tf.nn.bidirectional_dynamic_rnn(self.discriminator_cell_fw,
                                                             self.discriminator_cell_bw,
                                                             inputs=target_output_emb,
                                                             sequence_length=len_target,
                                                             initial_state_fw=forward_states,
                                                             initial_state_bw=backward_states,
                                                             time_major=False)
            # Concat outputs and states of forward and backward lstm
            output_real = tf.concat(output_real, 2)

            # Take output of last time step
            cat_idx = tf.stack([tf.range(0, tf.shape(output_real)[0]), tf.to_int32(len_target - 1)], axis=1)
            output_real = tf.gather_nd(output_real, cat_idx)

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
        D_pretrain_loss = tf.reduce_mean(cross_entropy_fake) + tf.reduce_mean(cross_entropy_real)

        # Variables of discriminator
        discriminator_params = [param for param in tf.trainable_variables() if "discriminator" in param.name]

        # Use Adam solver for train discriminator
        with tf.variable_scope("discriminator", reuse=reuse):
            D_solver = tf.train.AdamOptimizer().minimize(D_pretrain_loss, var_list=discriminator_params)

        return discriminator_params, D_pretrain_loss, D_solver, tf.nn.softmax(dense_output_fake)

    def build_rl_module(self, train_examples, final_outputs, generator_params, beam_outputs):
        """
        Train generator with reinforcement learning
        :param train_examples: Examples out of train set
        :param final_outputs: final outputs of RNN for calculating pre train loss
        :param generator_params: Parameter of generator
        :param beam_outputs: Generated examples of beam search
        :return:
        """

        # Unpack subject, content and answers and corresponding length
        subject, len_subject, content, len_content, target_input, target_output, len_target = train_examples

        # Choose best answer per question
        target_output = target_output[:, 0, :]

        # Concat subject and content to feed it into encoder
        sub_cont_concat_op = tf.map_fn(self.concat_seqs, [subject, len_subject, content, len_content])[0]
        len_both = len_subject + len_content

        # Get maximum sequence length of predicted sequences
        seq_len_final_out = tf.shape(beam_outputs)[1]

        # Look up beam outputs and process a transpose for putting them into RNN
        processed_x = tf.transpose(tf.nn.embedding_lookup(self.embeddings_english, beam_outputs),
                                   perm=[1, 0, 2])

        # Tensor array for saving process x data
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=seq_len_final_out, name="ta_emb_x", clear_after_read=False)
        ta_emb_x = ta_emb_x.unstack(processed_x)

        # Tensor array for beam output IDs
        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len_final_out, name="ta_x",
                                            clear_after_read=False)
        ta_x = ta_x.unstack(tf.transpose(beam_outputs, perm=[1, 0]))

        # Tensor array for saving the generated sequences with monte carlo method
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len_final_out ** 2,
                                             dynamic_size=False, infer_shape=True, name="ta_gen_x")

        # Loop for every RNN step
        def outer_loop(i, count, gen_x):

            # Set given num = i for inner loop
            given_num = i

            # When current index i < given_num, use the provided tokens as the input at each time step
            def _g_recurrence_1(j, x_t, h_tm1, given_num, gen_x, count):
                # Decoder cell step, receive output and hidden state
                output, h_t = self.attention_decoder_cell(x_t, h_tm1)
                # Read token at position j from tensor array
                token_to_write = ta_x.read(j)
                # Read token embeddings at position j from tensor array for next time step
                x_tp1 = ta_emb_x.read(j)
                # Write token to tensor array
                gen_x = gen_x.write(count, token_to_write)

                return j + 1, x_tp1, h_t, given_num, gen_x, count + 1

            # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
            def _g_recurrence_2(j, x_t, h_tm1, given_num, gen_x, count):
                # Decoder cell step, receive output and hidden state
                output, h_t = self.attention_decoder_cell(x_t, h_tm1)
                # Calculate probabilities for every token in vocabulary
                o_t = self.projection_layer(output)
                # Calculate log probabilities
                log_prob = tf.log(tf.nn.softmax(o_t))
                # Get next token by taking one random token out of multinomial distribution of probabilities
                next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                # Look up word vectors for next tokens
                x_tp1 = tf.nn.embedding_lookup(self.embeddings_english, next_token)
                # Write token to tensor array
                gen_x = gen_x.write(count, next_token)
                return j + 1, x_tp1, h_t, given_num, gen_x, count + 1

            # First inner loop defines the fixed token for monte carlo search
            j, x_t, h_tm1, given_num, gen_x, count = control_flow_ops.while_loop(
                cond=lambda j, _2, _3, given_num, _5, _6: j < given_num,
                body=_g_recurrence_1,
                loop_vars=(tf.constant(0, dtype=tf.int32),
                           tf.nn.embedding_lookup(self.embeddings_english, self.start_token),
                           self.attention_zero_state,
                           given_num,
                           gen_x,
                           count))

            # Second inner loop defines monte carlo search with the previous selected fixed token
            _, _, _, _, gen_x, count = control_flow_ops.while_loop(
                cond=lambda j, _2, _3, _4, _5, _6: j < seq_len_final_out,
                body=_g_recurrence_2,
                loop_vars=(
                    j, tf.nn.embedding_lookup(self.embeddings_english, self.start_token), h_tm1, given_num, gen_x,
                    count))

            return i + 1, count, gen_x

        # Overall loop for calculating all sequences with monte carlo
        _, _, gen_x = control_flow_ops.while_loop(
            cond=lambda i, count, _2: i < seq_len_final_out + 1,
            body=outer_loop,
            loop_vars=(tf.constant(1, dtype=tf.int32), tf.constant(0, dtype=tf.int32), gen_x))

        # Transform tensor array to tensor and reshape it
        gen_x = gen_x.stack()
        gen_x = tf.transpose(gen_x, perm=[1, 0])
        gen_x = tf.reshape(gen_x, [seq_len_final_out * self.batch_size, seq_len_final_out])

        # Get seq length of generated sequences.
        seq_lens = tf.map_fn(self.get_seq_len, gen_x)

        # Multiply subject-content-concatination for having the same count as the examples of generated examples by
        # generation with monte carlo
        sub_cont_concat_op = tf.tile(sub_cont_concat_op, [seq_len_final_out, 1])
        len_both = tf.reshape(len_both, [self.batch_size, 1])
        len_both = tf.tile(len_both, [1, seq_len_final_out])
        len_both = tf.reshape(len_both, [-1])

        # Look up word vectors subject-content and generated examples
        sub_cont_emb = tf.nn.embedding_lookup(self.embeddings_english, sub_cont_concat_op)
        gen_x_embs = tf.nn.embedding_lookup(self.embeddings_english, gen_x)

        # Create batches to process data in a parallel loop
        sub_cont_emb_batches = tf.reshape(sub_cont_emb, [seq_len_final_out, self.batch_size, self.max_seq_len * 3,
                                                         self.embedding_size])
        len_both_batches = tf.reshape(len_both, [seq_len_final_out, self.batch_size])
        gen_x_embs_batches = tf.reshape(gen_x_embs,
                                        [seq_len_final_out, self.batch_size, seq_len_final_out, self.embedding_size])
        seq_lens_batches = tf.reshape(seq_lens, [seq_len_final_out, self.batch_size])

        # Get discriminator rewards by applying function discriminator_reward
        dense_output = tf.map_fn(self.discriminator_reward,
                                 [sub_cont_emb_batches, len_both_batches, gen_x_embs_batches, seq_lens_batches],
                                 parallel_iterations=1, back_prop=False)[0]

        # Reshape rewards for multiplication with pre train loss
        dense_output = tf.reshape(dense_output, [seq_len_final_out * self.batch_size, 2])
        rewards = tf.reshape(tf.nn.softmax(dense_output)[:, 1], [self.batch_size, seq_len_final_out])

        # Cross entropy Loss
        seq_len_fo = tf.shape(final_outputs.sample_id)[1]
        target_output = target_output[:, :seq_len_fo]

        # Find minimum of shapes seq_len_final_out and rewards
        min_shape = tf.minimum(seq_len_fo, tf.shape(rewards)[1])

        # Calculate pre train loss with rnn outputs and multiply it with reward values of discrimnator
        cross_rl = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output,
            logits=final_outputs.rnn_output
        )[:, :min_shape] * rewards[:, :min_shape]

        # Mask Padding by weighting padding values with zero
        target_weights = tf.cast(tf.greater(target_output, 0), tf.float32)[:, :min_shape]

        # Multiply masked padding to loss and reduce sum. Divide with batch size to receive final loss
        self.loss_rl = tf.reduce_sum(cross_rl * target_weights) / self.batch_size

        # Train with adam optimizer
        self.rl_gen_train_op = tf.train.AdamOptimizer().minimize(self.loss_rl, var_list=generator_params)

        return "discrims"

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

    def get_seq_len(self, seq):
        # Get sequence length of one sequence
        seq_len = tf.to_int32(tf.argmax(tf.to_int32(tf.equal(seq, 3)))) + 1
        return seq_len

    def take_last_output(self, x):
        # gives back last output of a sequence given the sequence length
        output = x[0]
        seq_len = x[1]
        return [output[seq_len - 1], 0]

    def discriminator_reward(self, x):
        """
        Discriminator reward step. Discriminator receives real and fake data in batches and returns a probabilities for
        fake data for being fake or real
        :param x: Contains subject and content embeddings, length of subject-content, generated sequences and length
        of generated sequences
        :return: probabilities that a sequence is real or fake
        """

        # Unpack values
        sub_cont_emb = x[0]
        len_both = x[1]
        gen_x_embs = x[2]
        seq_lens = x[3]

        # Encode subject and content with discriminators encoder
        with tf.variable_scope("discriminator_encoder", reuse=True):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(self.d_encoder_cell_fw,
                                                                     self.d_encoder_cell_bw,
                                                                     inputs=sub_cont_emb,
                                                                     sequence_length=len_both,
                                                                     dtype=tf.float32,
                                                                     time_major=False)

            # Concat forward and backward state
            forward_states, backward_states = output_states

        # Use discriminator to classify real and fake answers
        with tf.variable_scope("discriminator", reuse=True):
            # First make predictions for generated data
            output, _ = tf.nn.bidirectional_dynamic_rnn(self.discriminator_cell_fw,
                                                        self.discriminator_cell_bw,
                                                        inputs=gen_x_embs,
                                                        sequence_length=seq_lens,
                                                        initial_state_fw=forward_states,
                                                        initial_state_bw=backward_states,
                                                        time_major=False)

            # Concat outputs and states of forward and backward lstm for generated data
            output = tf.concat(output, 2)
            # Take output of last RNN step
            output = tf.map_fn(self.take_last_output, [output, seq_lens])[0]
            # Dense layer which shows probability that a sequence is classified as real or fake
            dense_output = tf.layers.dense(inputs=output, units=2, reuse=True)

        return [dense_output, 0, 0, 0]
