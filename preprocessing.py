from data import data_loader

if __name__ == "__main__":
    loader = data_loader(2048, 64, 64, 1000, 5000)
    loader.save_npy()
    loader.save_embedding_matrix(200)



