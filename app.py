import argparse
from flask import Flask
from flask_restplus import Resource, Api, fields
import tensorflow as tf

from model.pre_train_model import pre_train_model
from model.rl_model import rl_model
from model.ael_model import ael_model
from helper import read_data

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_folder", help="Specify location of data")
args = parser.parse_args()

# Config and initialize session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Load bpe dictionary and vocabulary
bpe_dict, vocab_list = read_data.load_vocab_and_dict(args.data_folder)

# Initialize all four models
pre_model = pre_train_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
                            mode="infer")
pre_model_beam = pre_train_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
                                 mode="infer", beam_width=50, reuse=True)
# Not this one ...
# ael_model_init = ael_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
#                                  mode="infer", reuse=True)
rl_model_init = rl_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
                         mode="infer", reuse=True)

# Define weights path of the models
pre_weights_path = args.data_folder + "generator_weights_pre/"
ael_weights_path = args.data_folder + "generator_weights_ael/"
rl_weights_path = args.data_folder + "generator_weights_rl/"

# Initialize safer
saver_generator = tf.train.Saver(tf.trainable_variables())

app = Flask(__name__)
api = Api(app)

question = api.model('question', {
    'question': fields.String(required=True, description='Question to ask')
})

answer = api.model('answer', {
    'pre_train_model': fields.String(required=True, description='Question to ask'),
    'beam_model': fields.String(required=True, description='Question to ask'),
    'reinforcement_learning_model': fields.String(required=True, description='Question to ask')
})


@api.route('/request_answer')
class QuestionAnswering(Resource):

    @api.doc('ask_question')
    @api.expect(question)
    @api.marshal_with(answer, code=201)
    def post(self):
        '''Request an answer for a question'''
        question = api.payload['question']

        # Restore weights for pre train model and generate answer without and with beam search
        saver_generator.restore(sess, tf.train.latest_checkpoint(pre_weights_path))
        preds_pre_train = pre_model.answer_question(sess, question, '', bpe_dict, vocab_list)
        preds_beam = pre_model_beam.answer_question(sess, question, '', bpe_dict, vocab_list)

        # Restore weights for rl model and generate answers
        saver_generator.restore(sess, tf.train.latest_checkpoint(rl_weights_path))
        preds_rl = rl_model_init.answer_question(sess, question, '', bpe_dict, vocab_list)

        response = {
            'pre_train_model': preds_pre_train,
            'beam_model': preds_beam,
            'reinforcement_learning_model': preds_rl
        }

        return response, 201


if __name__ == '__main__':
    app.run()
