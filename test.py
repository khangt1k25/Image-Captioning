import numpy as np
import tensorflow as tf
from data import modelCNN, preprocess_img_path, data_loader
from models import Encoder, Decoder
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from train import validate
import math


if __name__ == "__main__":
    
    loader = data_loader(
        features_shape=2048,
        attention_features_shape=64,
        batch_size=256,
        buffer_size=1000,
        top_k=5000
    )
    
    dataset_train = loader.load_dataset("train")
    dataset_test = loader.load_dataset("test")
    tokenizer = loader.tokenizer
    
    # loading model
    embedding_matrix = np.load("./content/drive/My Drive/datasets/embeddingmatrix.npy")
    encoder = Encoder(encoder_dim = 200)
    decoder = Decoder(embedding_dim = 200, vocab_size = loader.top_k + 1, units = 512, embedding_matrix = embedding_matrix)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    checkpoint_path = "./content/drive/My Drive/datasets/modelcheckpoint/lstm"
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)


    print(start_epoch)
    # # tesing
    bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0
    for (batch, (img_tensor, target)) in enumerate(dataset_train):
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
    bleu = bleu_1 + bleu_2 + bleu_3 + bleu_4
    bleu = bleu/4
    print("Bleu_1: {}".format(bleu_1))
    print("Bleu_2: {}".format(bleu_2))
    print("Bleu_3: {}".format(bleu_3))
    print("Bleu_4: {}".format(bleu_4))
    print("Bleu  : {}".format(bleu))





        
    