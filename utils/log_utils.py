from datetime import datetime

def log(*args, **kwargs):
    current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"【{formatted_time}】", end="")
    print(*args, **kwargs)