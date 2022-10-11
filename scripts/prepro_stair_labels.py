import json
from tqdm import tqdm

results = {"images" : [], "dataset": "coco"}
for split in ["train", "val"]:
    print(f"Preprocess for {split} ... ")
    dataset = None
    if split == "train":
        with open("STAIR-captions/stair_captions_v1.2_train_tokenized.json", 'r') as f:
            dataset = json.load(f)
    else:
        with open("STAIR-captions/stair_captions_v1.2_val_tokenized.json", 'r') as f:
            dataset = json.load(f)

    captions = {}
    for anno in dataset["annotations"]:
        img_id = anno["image_id"]
        captions.setdefault(img_id,[])
        captions[img_id].append(anno)

    imgs = {}
    for img in dataset["images"]:
        imgs[img["id"]] = img
    
    L = len(captions.items())
    for i, (img_id, annos) in enumerate(tqdm(captions.items())):
        res = {}
        img = imgs[img_id]

        sentences = []
        for anno in annos:
            s = {}
            tokens = anno["tokenized_caption"].split()
            s["tokens"] = tokens
            s["raw"] = anno["caption"]
            s["imgid"] = i
            s["sentid"] = anno["id"]
            sentences.append(s)

        res["filepath"] = f"{split}2014"
        res["sentids"] = list(map(lambda x:x["id"],annos))
        res["filename"] = img["file_name"]
        res["imgid"] = i
        res["split"] = split
        res["sentences"] = sentences
        res["cocoid"] = img["id"]

        if split == "val" and i > L//2:
            res["split"] = "test"

        results["images"].append(res)

print("Write ...")
with open("data/stair_coco.json","w") as f:
    output = json.dumps(results,indent=4)
    f.write(output)
