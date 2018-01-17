import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


def vector_extrema(pred, len_pred, gold, len_gold, embeddings):
    """
    Vector extrema metrics computes the value of every single dimension in the word vectors which has the greatest
    difference from zero.
    :param pred: predicted sequences
    :param len_pred: length of predicted sequences
    :param gold: real sequences
    :param len_gold: length of real sequences
    :param embeddings: word embeddings
    :return: score for every real-prediction-pair
    """

    # Get maximum sequence length by determine maximum shape of gold and predicted data
    max_seq_len = tf.maximum(tf.shape(pred)[1], tf.shape(gold)[1])

    # Get batch size by determine shape of predicted data
    batch_size = tf.shape(pred)[0]

    # Pad pred and/or gold sequences if the length of them is lower then maximum sequence length
    sequences1 = tf.pad(pred, [[0, 0], [0, max_seq_len - tf.shape(pred)[1]]])
    sequences2 = tf.pad(gold, [[0, 0], [0, max_seq_len - tf.shape(gold)[1]]])

    # look up word embeddings for gold and predicted sequences
    seq1_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences1))
    seq2_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences2))

    # Initialize tensor array for following calculation loop
    scores = tensor_array_ops.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False, infer_shape=True)

    def _calc_loop(i, score_array):
        """
        Loop for iterating over every single batch and finding the score
        :param i: loop counter
        :param score_array: score array to save the scores
        :return: loop counter i for next iteration and score_array
        """

        # Find minimum and maximum value for every dimension in predictions
        seq1_max = tf.reshape(tf.reduce_max(seq1_embs[i, :len_pred[i]], axis=0), [1, -1])
        seq1_min = tf.reshape(tf.reduce_min(seq1_embs[i, :len_pred[i]], axis=0), [1, -1])

        # Find the maximum absolute value in min and max data
        seq1_mask = tf.greater(tf.abs(seq1_max), tf.abs(seq1_min))

        # Select values by multiplying mask to determined minimum and maximum values
        seq1_max = tf.multiply(seq1_max, tf.cast(seq1_mask, tf.float32))
        seq1_min = tf.multiply(seq1_min, tf.cast(tf.logical_not(seq1_mask), tf.float32))

        # Add vectors for finding final sequence representation for predictions
        seq1 = seq1_max + seq1_min

        # Find minimum and maximum value for every dimension in real data
        seq2_max = tf.reshape(tf.reduce_max(seq2_embs[i, :len_gold[i]], axis=0), [1, -1])
        seq2_min = tf.reshape(tf.reduce_min(seq2_embs[i, :len_gold[i]], axis=0), [1, -1])

        # Find the maximum absolute value in min and max data
        seq2_mask = tf.greater(tf.abs(seq2_max), tf.abs(seq2_min))

        # Select values by multiplying mask to determined minimum and maximum values
        seq2_max = tf.multiply(seq2_max, tf.cast(seq2_mask, tf.float32))
        seq2_min = tf.multiply(seq2_min, tf.cast(tf.logical_not(seq2_mask), tf.float32))

        # Add vectors for finding final sequence representation for real data
        seq2 = seq2_max + seq2_min

        # Get score by applying cosine similarity on representation of prediction and of real data
        cs = cosine_similarity(seq1, seq2)

        # Save score in score array
        score_array = score_array.write(i, cs)

        return i + 1, score_array

    # Loop over every single batch. Start loop with i=0 and empty scores array. Filled scores array is returned.
    _, scores = control_flow_ops.while_loop(
        cond=lambda i, _1: i < batch_size,
        body=_calc_loop,
        loop_vars=(tf.constant(0, dtype=tf.int32), scores))

    # Flatten scores
    scores = tf.reshape(scores.stack(), [-1])

    return scores


def embedding_average_score(pred, len_pred, gold, len_gold, embeddings):
    """
    Computes a score for every prediction gold pair by calculating the mean of the prediction word vectors and the gold
    word vectors and comparing them with cosine similarity
    :param pred: predicted sequences
    :param len_pred: length of predicted sequences
    :param gold: real sequences
    :param len_gold: length of real sequences
    :param embeddings: word embeddings
    :return: score for every real-prediction-pair
    """

    # Get maximum sequence length by determine maximum shape of gold and predicted data
    max_seq_len = tf.maximum(tf.shape(pred)[1], tf.shape(gold)[1])

    # Get batch size by determine shape of predicted data
    batch_size = tf.shape(pred)[0]

    # Pad pred and/or gold sequences if the length of them is lower then maximum sequence length
    sequences1 = tf.pad(pred, [[0, 0], [0, max_seq_len - tf.shape(pred)[1]]])
    sequences2 = tf.pad(gold, [[0, 0], [0, max_seq_len - tf.shape(gold)[1]]])

    # look up word embeddings for gold and predicted sequences
    seq1_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences1))
    seq2_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences2))

    # Initialize tensor array for following calculation loop
    scores = tensor_array_ops.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False, infer_shape=True)

    def _calc_loop(i, score_array):
        """
        Loop for iterating over every single batch and finding the score
        :param i: loop counter
        :param score_array: score array to save the scores
        :return: loop counter i for next iteration and score_array
        """

        # Select one batch of seq1
        seq1_meaned = seq1_embs[i, :len_pred[i]]

        # Calculate mean over seq1
        seq1_meaned = tf.reduce_mean(seq1_meaned, axis=0)

        # Select one batch of seq2
        seq1_meaned = tf.reshape(seq1_meaned, [1, -1])

        # Calculate mean over seq2
        seq2_meaned = tf.reshape(tf.reduce_mean(seq2_embs[i, :len_gold[i]], axis=0), [1, -1])

        # Calculate cosine similarity over meaned seq1 and seq2
        cos_sim = cosine_similarity(seq1_meaned, seq2_meaned)

        # Save score in score array
        score_array = score_array.write(i, cos_sim)

        return i + 1, score_array

    # Loop over every single batch. Start loop with i=0 and empty scores array. Filled scores array is returned.
    _, scores = control_flow_ops.while_loop(
        cond=lambda i, _1: i < batch_size,
        body=_calc_loop,
        loop_vars=(tf.constant(0, dtype=tf.int32), scores))

    # Flatten scores
    scores = tf.reshape(scores.stack(), [-1])

    return scores


def greedy_embedding_matching(sequences1, seq_len1, sequences2, seq_len2, embeddings):
    """
    Calculates greedy embedding score for two sequences. The score is calculated two times because it is asymetric.
    The final score is the mean over the two calculation steps.
    :param sequences1: first sequence batch
    :param seq_len1: length of first sequences
    :param sequences2: second sequence batch
    :param seq_len2: length of second sequences
    :param embeddings: word embeddings
    :return:
    """

    # Do two calculations steps, first with sequences1 on first position and second with sequences2 on first position
    step_one = greedy_embedding_matching_step(sequences1, seq_len1, sequences2, seq_len2, embeddings)
    step_two = greedy_embedding_matching_step(sequences2, seq_len2, sequences1, seq_len1, embeddings)

    # Concat the results and calculate the mean
    scores = tf.reshape(tf.concat([step_one, step_two], axis=0), [2, -1])
    scores = tf.reduce_mean(scores, axis=0)

    return scores


def greedy_embedding_matching_step(sequences1, seq_len1, sequences2, seq_len2, embeddings):
    """
    Calculating the greedy score of two sequences
    :param sequences1: first sequence batch
    :param seq_len1: length of first sequences
    :param sequences2: second sequence batch
    :param seq_len2: length of second sequences
    :param embeddings: word embeddings
    :return:
    """

    # Get maximum sequence length by determine maximum shape of sequences1 and sequences2
    max_seq_len = tf.maximum(tf.shape(sequences1)[1], tf.shape(sequences2)[1])

    # Get batch size by determine shape of sequences1
    batch_size = tf.shape(sequences1)[0]

    # Pad sequences1 and/or sequences2 if the length of them is lower then maximum sequence length
    sequences1 = tf.pad(sequences1, [[0, 0], [0, max_seq_len - tf.shape(sequences1)[1]]])
    sequences2 = tf.pad(sequences2, [[0, 0], [0, max_seq_len - tf.shape(sequences2)[1]]])

    # look up word embeddings for sequences1 and sequences2
    seq1_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences1))
    seq2_embs = tf.nn.embedding_lookup(embeddings, tf.nn.relu(sequences2))

    # Initialize tensor array for following calculation loop
    scores = tensor_array_ops.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False, infer_shape=True)

    def _calc_loop(i, score_array):
        """
        Loop for iterating over every single batch and finding the score
        :param i: loop counter
        :param score_array: score array to save the scores
        :return: loop counter i for next iteration and score_array
        """

        # Multiply sequence seq1 to seq_len2 to find the closest embeddings to every word
        seq1_tiled = tf.tile(seq1_embs[i, :seq_len1[i]], [1, seq_len2[i]])

        # Reshape seq1 and seq2
        seq1_unrolled = tf.reshape(seq1_tiled, [seq_len1[i] * seq_len2[i], -1])
        seq2_unrolled = tf.tile(seq2_embs[i, :seq_len2[i]], [seq_len1[i], 1])

        # Put seq1 and seq2 into cosine similarity
        cos_sim = tf.reshape(cosine_similarity(seq1_unrolled, seq2_unrolled), [seq_len1[i], -1])

        # Find words with max cosine similarity
        max_cos_sim = tf.reduce_max(cos_sim, axis=1)

        # Calculate mean over all words with max cosine similarity
        mean_cos_sim = tf.reduce_mean(max_cos_sim)

        # Save score into score array
        score_array = score_array.write(i, mean_cos_sim)

        return i + 1, score_array

    # Loop over every single batch. Start loop with i=0 and empty scores array. Filled scores array is returned.
    _, scores = control_flow_ops.while_loop(
        cond=lambda i, _1: i < batch_size,
        body=_calc_loop,
        loop_vars=(tf.constant(0, dtype=tf.int32), scores))

    # Turn tensor array into tensor
    scores = scores.stack()

    return scores


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    :param vec1: Vector1
    :param vec2: Vector2
    :return: Returns cosine similarity score
    """

    # Normalize vector 1 and 2
    normalize1 = tf.nn.l2_normalize(vec1, 1)
    normalize2 = tf.nn.l2_normalize(vec2, 1)

    # Calculate cosine similarity
    cos_similarity = tf.reduce_sum(tf.multiply(normalize1, normalize2), axis=1)

    return cos_similarity
