from image_processing import load_image, save_image

data = load_image("/Users/tl9ie/research-2016/image-data/full-cropped", expand=True)
save_image(data, "bird_full_expanded.pkl.gz")
