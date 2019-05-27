import os, shutil

metadata_path = "..\\datasets\\PhysNetReal\\all_metadata.txt"
split_path = "..\\datasets\\PhysNetReal\\split_only_stable.txt"

stable = set()
scene_count = 0
with open(metadata_path, "r") as metadata:
    for scene in metadata:
        scene_count += 1
        scene_att = scene.split()
        idx = scene_att[0]
        fall = scene_att[18]
        if not int(fall):
            stable.add(idx)

stable_scene_count = len(stable)
num_train = round(stable_scene_count * 0.8)
num_val = round(stable_scene_count * 0.1)
num_test = stable_scene_count - num_train - num_val

f= open(split_path, "w+")

for scene in range(scene_count):
    if scene in stable:
        # TODO: split is 1, 2 or 3
        pass
    else:
        split = 0
    # f.write(str(scene) + "\t" + str(split))
# f.close()