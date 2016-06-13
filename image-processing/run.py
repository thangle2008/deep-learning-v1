from image_processing import load_image, save_image

data = load_image("/Users/tl9ie/research-2016/image-data/wholedataset", dim=140, expand_train=True, mode="RGB")
save_image(data, "bird_full_expanded_no_cropped_no_empty_140_rgb.pkl.gz")
