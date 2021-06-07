import sys
import time
from batch_data_utils import *
from batch_model import SummaRuNNer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.set_printoptions(threshold=sys.maxsize)
# Parameters
# ==================================================
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
vocab = Vocab(params.max_vocab_size, emb_dim=50, dataset_path='data/',
              glove_path='glove/glove.6B.50d.txt',
              vocab_path='data_files/vocab.txt', lookup_path='data_files/lookup.pkl')

dg = DataGenerator('data/', params.max_inp_seq_len, params.max_out_seq_len, vocab, use_pgen=False,
                   use_sample=False)
current_time = str(time.time())
log_dir = 'logs/' + current_time


def train():
    f = open('output.txt', 'w')
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            Summa = SummaRuNNer(vocab.size(), vocab._dim, vocab.wvecs)
            writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
            train_loss_summary = tf.compat.v1.summary.scalar('train_loss', Summa.loss)
            streaming_loss, streaming_loss_update = tf.contrib.metrics.streaming_mean(Summa.loss)
            streaming_loss_scalar = tf.summary.scalar('validation_loss', streaming_loss)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_params = tf.compat.v1.trainable_variables()
            train_op = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.1).minimize(Summa.loss,
                                                                                        var_list=train_params)
            # train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1).minimize(Summa.loss, var_list=train_params)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # create log dirs
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", current_time))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            min_eval_loss = float('Inf')
            min_train_loss = float('Inf')
            val_step = 0
            step = 0
            for epoch in range(FLAGS.num_epochs):
                while True:
                    (enc_in, dec_in) = dg.get_batch()
                    step += 1
                    feed_dict = {
                        Summa.x: enc_in,
                        Summa.y: dec_in,
                    }
                    if enc_in.shape[0] != params.batch_size:
                        break
                    # sess.run(train_op, feed_dict)
                    [_, train_loss, train_summary] = sess.run([train_op, Summa.loss, train_loss_summary], feed_dict)

                    print('Epoch: ' + str(epoch) + ' Batch: ' + str(step) + ' loss: ' + str(train_loss))
                    f.write('Epoch: ' + str(epoch) + ' Batch: ' + str(step) + ' loss: ' + str(train_loss) + '\n')
                    f.flush()
                    writer.add_summary(train_summary, step)

                    if step % FLAGS.checkpoint_every == 0 and step != 0:  # fix
                        eval_loss = 0
                        val_step += 1
                        val_dg = DataGenerator('data/', params.max_inp_seq_len,
                                               params.max_out_seq_len, vocab,
                                               use_pgen=False, use_sample=False)
                        val_batches = 0
                        while True:
                            (enc_in, dec_in) = val_dg.get_batch(split='val')
                            if enc_in.shape[0] != params.batch_size:
                                break
                            val_batches += 1
                            feed_dict = {
                                Summa.x: enc_in,
                                Summa.y: dec_in,
                            }
                            [val_loss, _] = sess.run([Summa.loss, streaming_loss_update], feed_dict)
                            eval_loss += val_loss

                            if val_dg.ptr == 0:  # finished epoch of validation data
                                break
                        streaming_summ = sess.run(streaming_loss_scalar)
                        writer.add_summary(streaming_summ, val_step)
                        print('Validation Step: ' + str(val_step) + ' Loss in validation: ' + str(
                            eval_loss / val_batches))
                        f.write('Validation Step: ' + str(val_step) + ' Loss in validation: ' + str(
                            eval_loss / val_batches) + '\n')
                        f.flush()
                        if eval_loss < min_eval_loss:
                            min_eval_loss = eval_loss
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                    if dg.ptr == 0:  # finished epoch
                        break
    f.close()


if __name__ == "__main__":
    train()
