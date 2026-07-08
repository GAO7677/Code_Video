import json
import glob
import os
def generate_json_file(video_dir,output_path):
    mp4_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    data_list = []
    for video in mp4_files:
        if "mask" in video:
            continue
        data = {
            "path": video,
            "caption": "The video shows rigid body motion.",
        }
        data_list.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        for d in data_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print(f"* Save {len(data_list)} data to {output_path}")


if __name__ == "__main__":

    # change your setting here
    video_dir = "./data/example_videos"
    output_path = "./data/data.jsonl"

    # generate
    generate_json_file(video_dir=video_dir,output_path=output_path)
