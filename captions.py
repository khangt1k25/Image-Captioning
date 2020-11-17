import numpy as np
import tensorflow as tf
from data import modelCNN, preprocess_img, data_loader
from models import Encoder, Decoder
from PIL import Image
import matplotlib.pyplot as plt


def evaluate(encoder, decoder, tokenizer, max_length, attention_features_shape, image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(preprocess_img(image)[0], 0)
    img_tensor_val = modelCNN(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
    )

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == "<end>":
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[: len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())

    plt.savefig(
        "/content/drive/My Drive/datasets/Flickr8k/Flickr8k_Dataset/Flickr8k_stats/res.png"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    loader = data_loader(
        features_shape=2048,
        attention_features_shape=64,
        batch_size=256,
        buffer_size=1000,
        top_k=5000,
    )
    rid = np.random.randint(0, len(loader.name_test))
    image_id = loader.name_test[rid]
    image_path = (
        "/content/drive/My Drive/datasets/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/"
        + image_id
    )
    real_caption = " ".join(
        [loader.tokenizer.index_word[i] for i in loader.cap_test[rid] if i not in [0]]
    )
    # loading model
    encoder = Encoder(200)
    decoder = Decoder(embedding_dim=200, vocab_size=loader.top_k + 1, units=512)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    checkpoint_path = "/content/drive/My Drive/datasets/modelcheckpoint/train"
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # tesing
    result, attention_plot = evaluate(
        encoder,
        decoder,
        loader.tokenizer,
        loader.max_length,
        loader.attention_features_shape,
        image_id,
    )
    print("Real Caption:", real_caption)
    print("Prediction Caption:", " ".join(result))