import sys
import os

import gdown


DIR = "./expert_datasets"

DEMOS = {
    "ant": [("25000", "1MIFuQvSE-T_K4E06h_ys7dI7Ds1sf-wG")],
    "hand": [("10000_v2", "1fj2JRTKcyzTXxwn804pi2ptkaFeYPFot")],
    "maze2d": [ ("100", "1QY5IusLHcqwD7UrHkzkQ3pselIU5zjPk")],
    "pick": [("10000_clip", "103W-o5btEbRez596HBT6O73n7VJyV5lY")],
    "push": [("10000_clip", "1k8GmSfYBTyKzKMrgRw9z0GjhDKrrX9lM")],
    "walker": [("5traj_processed", "1tSXGBPGgE92LqgzM0QsEri23DkHynCtX")],
}


if __name__ == "__main__":
    tasks = []
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = ["ant", "hand", "maze2d", "pick", "push", "walker"]

    os.makedirs(DIR, exist_ok=True)

    for task in tasks:
        for postfix, id in DEMOS[task]:
            url = "https://drive.google.com/uc?id=" + id
            target_path = "%s/%s_%s.pt" % (DIR, task, postfix)
            if os.path.exists(target_path):
                print("%s is already downloaded." % target_path)
            else:
                print("Downloading demo (%s_%s) from %s" % (task, postfix, url))
                gdown.download(url, target_path)
