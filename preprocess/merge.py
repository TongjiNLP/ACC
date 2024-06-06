import json
import os
import argparse

def merge_json(json_files:list,dst):
    new_json={}
    for file in json_files:
        with open(file,"r",encoding="utf-8") as f:
            data=json.load(f)
            new_json.update(data)

    print(f"data number: {len(new_json.keys())}")
    
    with open(dst,"w",encoding="utf-8",newline="") as f:
        json.dump(new_json,f,indent=4,ensure_ascii=False)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--prediction_dir",type=str,default="../predictions/MSQA_predictions")
    parser.add_argument("--dst",type=str,default="../data/MSQA_merge_prediction.json")

    args=parser.parse_args()

    merge_json(
        json_files=[os.path.join(args.prediction_dir,each) for each in args.prediction_dir],
        dst=args.dst
    )