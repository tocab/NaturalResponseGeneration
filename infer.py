from bottle import get, post, request, run, route
from helper import read_data
import tensorflow as tf
from model.pre_train_model import pre_train_model
from model.rl_model import rl_model
from model.ael_model import ael_model
import argparse

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
ael_model_init = ael_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
                                 mode="infer", reuse=True)
rl_model_init = rl_model(len(vocab_list), max_seq_len=50, batch_size=1, sess=sess, data_path=args.data_folder,
                                 mode="infer", reuse=True)

# Define weights path of the models
pre_weights_path = args.data_folder + "/generator_weights_pre/"
ael_weights_path = args.data_folder + "/generator_weights_ael/"
rl_weights_path = args.data_folder + "/generator_weights_rl/"

# Initialize safer
saver_generator = tf.train.Saver(tf.trainable_variables())


# Define start page
# link: http://localhost:8080/infer
@get('/infer')
def login():
    return '''
        <form action="/infer" method="post">
            Subject: <input name="subject" type="text" size="40"/> <br />
            Content: <input name="content" type="text" size="40"/> <br />
            <input value="Submit" type="submit" />
        </form>
    '''


# Define page after posting a question
@post('/infer')  # or @route('/login', method='POST')
def do_login():
    # Get subject and content
    subject = request.forms.get('subject')
    content = request.forms.get('content')

    # Restore weights for pre train model and generate answer without and with beam search
    saver_generator.restore(sess, tf.train.latest_checkpoint(pre_weights_path))
    preds_pre_train = pre_model.answer_question(sess, subject, content, bpe_dict, vocab_list)
    preds_beam = pre_model_beam.answer_question(sess, subject, content, bpe_dict, vocab_list)

    # Restore weights for ael model and generate answers
    saver_generator.restore(sess, tf.train.latest_checkpoint(ael_weights_path))
    preds_ael = ael_model_init.answer_question(sess, subject, content, bpe_dict, vocab_list)

    # Restore weights for rl model and generate answers
    saver_generator.restore(sess, tf.train.latest_checkpoint(rl_weights_path))
    preds_rl = rl_model_init.answer_question(sess, subject, content, bpe_dict, vocab_list)
    return '''
        <form action="/infer" method="post">
            Subject: <input name="subject" type="text" value="{}" size="40" /> <br />
            Content: <input name="content" type="text" value="{}" size="40" /> <br />
            <input value="Submit" type="submit" />
        </form>
        <br />
        <table>
        <tr><td><b>Pre-Train-Model:</b></td> <td>{}</td></tr>
        <tr><td><b>Beam-Search-Model:</b></td> <td>{}</td></tr>
        <tr><td><b>RL-Model:</b></td> <td>{}</td></tr>
        <tr><td><b>AEL-Model:</b></td> <td>{}</td></tr>
        </table>
    '''.format(subject, content, preds_pre_train, preds_beam, preds_rl, preds_ael)


# Run server
run(host='0.0.0.0', port=8080, debug=True)
