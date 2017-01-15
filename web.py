import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from flask import Flask, request
app = Flask(__name__)

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("author", "Kleef", "Part of the name of the author")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


author = "M.L."


def web(cnn, sess, vocab_processor):
    sess.run(tf.initialize_all_variables())
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", author))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Checkpoint restored.")
    else:
        print("No checkpoint found.")


    app = Flask(__name__)

    @app.route("/")
    def hello():
        return "Hello World!"

    @app.route("/score", methods=['GET', 'POST'])
    def return_score():
        input = request.args.get('text')
        input = [input]

        input = [data_helpers.clean_str(sent) for sent in input]

        x = np.array(list(vocab_processor.fit_transform(input)))
        return str(dev_step(x)[0])

    def dev_step(x_batch):
        """
        Evaluates model on a dev set
        """

        y = np.array([1], dtype=np.float32)
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y,
            cnn.dropout_keep_prob: 1.0
        }
        scores = sess.run(
            [cnn.scores],
            feed_dict)

        return scores[0]

    app.run()


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    x_text, y = data_helpers.load_data_and_labels(author)
    y = np.array(y, dtype=np.float32)
    # Build vocabulary
    listwords = [len(x.split(" ")) for x in x_text]
    max_document_length = max(listwords)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_document_length,
            num_classes=1,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        web(cnn,sess,vocab_processor)




