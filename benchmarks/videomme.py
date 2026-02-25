import json
import os
import pandas as pd
import re

class VideoMME:
    def __init__(self, config):
        self.video_dir = config['video_dir']
        self.raw_anno_path = config['anno_path']
        self.subset = config['subset']
        self.subtitle_dir = config['subtitle_dir'] if 'subtitle_dir' in config else None

        df = pd.read_parquet(self.raw_anno_path)
        if self.subset == "long w/o subs" or self.subset == "long w subs":
            data = df[df['duration']=='long'].to_json(orient="records")
        elif self.subset == "short w/o subs" or self.subset == "short w subs":
            data = df[df['duration']=='short'].to_json(orient="records")
        elif self.subset == "medium w/o subs" or self.subset == "medium w subs":
            data = df[df['duration']=='medium'].to_json(orient="records")
        else:
            data = df.to_json(orient="records")
        data = json.loads(data)
        self.rows = []
        row_id = 0
        for i in range(len(data)):
            video_id, ques, opts, ans = data[i]['videoID'], data[i]['question'],data[i]['options'], data[i]['answer']


            # option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
            # option = "\n".join([f"{opt}" for _, opt in enumerate(opts)])
            # question = ques + "\n" + option
            # post_prompt = "The best answer is:"
            # full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

            option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
            option = "\n".join([f"{opt}" for _, opt in enumerate(opts)])
            # full_prompt = option_prompt + "\n" + ques + "\n" + option + "\nThe best answer is:"
            full_prompt = option_prompt + "\n" + ques + "\n" + option +  "\nAnswer with the option's letter from the given choices directly."

            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

            if self.subset == "long w subs" or self.subset == "full w subs":
                subtitle_path = os.path.join(self.subtitle_dir, data[i]["videoID"] + ".srt")
                if os.path.exists(subtitle_path):
                    subtitle = open(subtitle_path).readlines()
                else:
                    subtitle = []
                subtitles_prompt = "This video's subtitles are listed below: \n"
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
                full_prompt = subtitles_prompt + subtitle_text + "\n" + full_prompt

            self.rows.append({
                "id" : row_id,
                "video_path" : video_path,
                "ques" : ques,
                "opts" : [opt[2:] for opt in opts],
                "gt_ans" : ans,
                "full_prompt" : full_prompt,
                "duration_type" : data[i]['duration'],
            })
            row_id+=1

    def metrics(self, result_path):
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['duration_type']
        res = {}
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                if item['ans'][0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                res[task_type][1]+=1
        m = []
        total, count = 0,0
        for k, v in res.items():
            count+=v[0]
            total+=v[1]
            m.append({
                "type" : k,
                "true" : v[0],
                "total" : v[1],
                "accuracy" : v[0]/v[1] if v[1]>0 else 0,
            })
        if self.subset == "full w/o subs" or self.subset == "full w subs":
            m.append({
                "type" : "overall",
                "true" : count,
                "total" : total,
                "accuracy" : count/total if total>0 else 0,
            })
        return m 