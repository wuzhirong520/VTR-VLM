from .model import VLM_Model
from openai import Client
import time
import imageio.v2 as imageio
import base64
from PIL import Image
from io import BytesIO
import tempfile

def get_new_size(frame, max_resolution):
    height, width = frame.shape[:2]
    if height > max_resolution or width > max_resolution:
        scale = min(max_resolution / height, max_resolution / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        return new_height, new_width
    return height, width

class API_VLM_Model(VLM_Model):
    def __init__(self, model_path, extra_model_config):
        self.model_path = model_path
        self.client = Client(api_key=extra_model_config['api_key'], base_url=extra_model_config['base_url'])
        self.support_video_url = extra_model_config['support_video_url']
        self.max_resolution = extra_model_config['max_resolution']
        self.fps = 2
        self.max_retry_times = 5
    
    def get_image_url(self, frame):
        buffer = BytesIO()
        image = Image.fromarray(frame)
        image = image.resize(get_new_size(frame, self.max_resolution), Image.BILINEAR)
        image.save(buffer, format="JPEG", quality=60, optimize=True)
        image_url = f"data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        # print(len(image_url)/1000/1000)
        return image_url

    def get_video_url(self, frames):
        if self.support_video_url:
            ffmpeg_params = [
                "-vf", "scale='min(720,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "30",
                "-pix_fmt", "yuv420p"
            ]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
                with imageio.get_writer(temp_file.name, format='FFMPEG', mode='I', fps=2, ffmpeg_params = ffmpeg_params) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                video_url = f"data:video/mp4;base64," + base64.b64encode(open(temp_file.name, "rb").read()).decode("utf-8")
            return video_url
        else:
            return [self.get_image_url(frame) for frame in frames]
    
    def infer(self, prompt, frames_info):
        retry_time = 0
        while retry_time < self.max_retry_times:
            try:
                if frames_info is None:
                    resp = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{
                                "content":[{ "text": prompt, "type": "text"}],
                                "role":"user"
                        }],
                    )
                else:
                    if self.support_video_url:
                        resp = self.client.chat.completions.create(
                            model=self.model_path,
                            messages=[{
                                    "content":[
                                        { "text": prompt, "type": "text"},
                                        { "video_url": {"url":frames_info['frames'], "fps": self.fps}, "type": "video_url"},
                                    ],
                                    "role":"user"
                            }],
                        )
                    else:
                        # import json
                        # with open("mytestgpt5.json", "w") as ff:
                        #     json.dump([{
                        #             "content":[
                        #                 { "text": prompt, "type": "text"},
                        #             ] + [{ "image_url": {"url" : url}, "type": "image_url"} for url in frames_info['frames']],
                        #             "role":"user"
                        #     }], ff, ensure_ascii=False, indent=4)
                        resp = self.client.chat.completions.create(
                            model=self.model_path,
                            messages=[{
                                    "content":[
                                        { "text": prompt, "type": "text"},
                                    ] + [{ "image_url": {"url" : url}, "type": "image_url"} for url in frames_info['frames']],
                                    "role":"user"
                            }],
                        )
                if resp.choices[0].message.content is None:
                    raise Exception("API return None")
                # print(resp.choices[0].message.content)
                return resp.choices[0].message.content
            except Exception as e:
                if str(e).find("'type': 'TooManyRequests'}}") < 0:
                    retry_time += 1
                else:
                    time.sleep(1)
                print("Error : ", e, f" Retry {retry_time}/{self.max_retry_times}")
        return ""