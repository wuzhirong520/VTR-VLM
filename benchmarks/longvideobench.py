import json
import os
import pandas as pd

class LongVideoBench:
    def __init__(self, config):
        self.video_dir = config['video_dir']
        self.raw_anno_path = config['anno_path']
        self.subset = config['subset']
        self.subtitle_dir = config['subtitle_dir'] if 'subtitle_dir' in config else None

        df = pd.read_parquet(self.raw_anno_path)
        data = df.to_json(orient="records")
        data = json.loads(data)
        self.rows = []
        row_id = 0
        for i in range(len(data)):
            video_path = os.path.join(self.video_dir, data[i]['video_path'])
            opts = []
            for oi in range(5):
                o = data[i][f'option{oi}']
                if o != "N/A":
                    opts.append(o)
            full_prompt = data[i]["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(opts)])
            if self.subset == "val w subs":
                subtitle_path = os.path.join(self.subtitle_dir, data[i]['subtitle_path'])
                with open(subtitle_path, "r") as f:
                    subtitles = json.load(f)
                    sub_str = "This video's subtitles are listed below: \n"
                    for subtitle in subtitles:
                        if "timestamp" in subtitle:
                            subtitle_text = subtitle["text"]
                        else:
                            subtitle_text = subtitle["line"]
                        sub_str += subtitle_text + "\n"
                    sub_str += "\nSelect the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"
                    full_prompt = sub_str + full_prompt
            self.rows.append({
                "id" : row_id,
                "video_path" : video_path,
                "duration_group" : data[i]['duration_group'],
                "ques" : data[i]['question'],
                "opts" : opts,
                "gt_ans" : chr(data[i]['correct_choice'] + ord('A')),
                "question_category" : data[i]['question_category'],
                "topic_category" : data[i]['topic_category'],
                "full_prompt" : full_prompt + "\n" + "Answer with the option's letter from the given choices directly.\n",
            })
            row_id+=1
    
    def metrics(self, result_path):
        type_map = {}
        for i in range(len(self.rows)):
            type_map[self.rows[i]['id']] = self.rows[i]['duration_group']
        res = {}
        total, count = 0,0
        with open(result_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                task_type = type_map[item['QA']['id']]
                if task_type not in res:
                    res[task_type]=[0,0,0]
                if len(item['ans']) > 0  and item['ans'][0]==item['QA']['gt_ans'][0]:
                    res[task_type][0]+=1
                    count+=1
                total+=1
                res[task_type][1]+=1
                res[task_type][2]=res[task_type][0]/res[task_type][1]
        res['overall'] = {
            "true_count" : count,
            "total_count" : total,
            "accuracy" : count/total,
        }
        res = {k:v for k,v in sorted(res.items(), key=lambda x:str(x[0]))}
        return res