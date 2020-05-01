import pickle

with open("../logs/video_file_dict.pkl", "rb") as fp:
    file_dict = pickle.load(fp)

print(file_dict["innmztffzd.mp4"])
