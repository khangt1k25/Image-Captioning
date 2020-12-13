import tensorflow as tf
import numpy as np
from time import time
from data import data_loader
from models import Encoder, Decoder, Attention
from utils import loss_function
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt


# Training epoch
@tf.function
def train_step(encoder, decoder, optimizer, tokenizer, loss_object, img_tensor, target):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(loss_object, target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = loss / int(target.shape[1])

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss



# Validate to return hypothesis, target for calculating BLEU 
def validate(encoder, decoder, optimizer, tokenizer, img_tensor, target):
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]] * target.shape[0], 1)
    features = encoder(img_tensor)
    hypo = [[tokenizer.word_index["<start>"]] * target.shape[0]]
    for i in range(1, target.shape[1]):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        pre = tf.math.argmax(predictions, axis=1)
        hypo.append(pre.numpy())
        dec_input = tf.expand_dims(pre, 1)
    hypo = np.array(hypo)
    hypo = hypo.T
    hypo = tokenizer.sequences_to_texts(hypo.tolist())
    target = tokenizer.sequences_to_texts(target.numpy().tolist())
    hypo = [[ele for ele in hyp.split(" ") if ele not in ["<pad>","<end>"]] for hyp in hypo]
    target = [[ele for ele in tar.split(" ") if ele not in ["<pad>","<end>"]] for tar in target]
    target = [[tar] for tar in target]
    return hypo, target


@tf.function
def run(EPOCHS):

    ## Load data and init 
    loader = data_loader(
        features_shape=2048,
        attention_features_shape=64,
        batch_size=64,
        buffer_size=1000,
        top_k=5000,
    )
    dataset_train = loader.load_dataset("train")
    dataset_test = loader.load_dataset("test")
    tokenizer = loader.tokenizer

    embedding_dim = 200
    encoder_dim = embedding_dim
    units = 512
    vocab_size = loader.top_k + 1
    num_steps = len(loader.train_path) // loader.batch_size

    ## Load model
    embedding_matrix = np.load("/content/drive/My Drive/datasets/embeddingmatrix.npy")
    encoder = Encoder(encoder_dim)
    decoder = Decoder(embedding_dim, vocab_size, units, embedding_matrix)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    ## Load checkpoint
    checkpoint_path = "/content/drive/My Drive/datasets/modelcheckpoint/embedding"
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # Running
    BLEU_1, BLEU_2, BLEU_3, BLEU_4 = [], [], [], []
    loss_plot = []
    for epoch in range(start_epoch, EPOCHS):
        start = time()
        total_loss = 0
        print("---Training---Epoch {}".format(epoch))
        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(
                encoder, decoder, optimizer, tokenizer, loss_object, img_tensor, target
            )
            total_loss += t_loss

        loss_plot.append(total_loss / num_steps)

        if epoch % 10 == 0:
            ckpt_manager.save()
            print("---Testing---Epoch {}".format(epoch))
            bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0
            for (batch, (img_tensor, target)) in enumerate(dataset_test):
                hypotheses, references = validate(
                    encoder, decoder, optimizer, tokenizer, img_tensor, target
                )
                bleu_1 += corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
                bleu_2 += corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
                bleu_3 += corpus_bleu(
                    references, hypotheses, weights=(0.33, 0.33, 0.33, 0)
                )
                bleu_4 += corpus_bleu(
                    references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
                )
                if batch == 5:
                    break
            bleu_1, bleu_2, bleu_3, bleu_4 = (
                bleu_1 / (batch + 1),
                bleu_2 / (batch + 1),
                bleu_3 / (batch + 1),
                bleu_4 / (batch + 1),
            )
            print("Bleu_1: {}".format(bleu_1))
            print("Bleu_2: {}".format(bleu_2))
            print("Bleu_3: {}".format(bleu_3))
            print("Bleu_4: {}".format(bleu_4))
            BLEU_1.append(bleu_1)
            BLEU_2.append(bleu_2)
            BLEU_3.append(bleu_3)
            BLEU_4.append(bleu_4)
            print("Epoch {} Loss {:.6f}".format(epoch, total_loss / num_steps))
            print("Time taken for 1 epoch {} sec\n".format(time() - start))

    return BLEU_1, BLEU_2, BLEU_3, BLEU_4, loss_plot


if __name__ == "__main__":
    b1, b2, b3, b4, loss = run(5)
    plt.plot(loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss plot")
    plt.savefig("/content/drive/My Drive/datasets/Flickr8k/Flickr8k_stats/loss.png")
    plt.show()
