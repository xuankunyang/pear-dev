"""Two-round API/vision/protocol check; generated shapes only, no robot hardware."""
import argparse
import json
import tempfile
import threading
from pathlib import Path
from PIL import Image, ImageDraw
from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput
from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.robot import BaseRobot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="gpt_62_sol")
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
config = BaseRobot.load_config(args.config, [])
models = ModelsConfig.from_station_config(config)
fmt = models.action_format(args.model)
state = {part: dict.fromkeys(names, 0.0) for part, names in fmt.schema.items()}
prompt = SystemPrompt.render(
    "This is a synthetic API integration test, not a robot task. All images are generated shapes. "
    "Briefly identify each named image's shape and color. In each turn, return exactly zero change "
    "for every arm and gripper quantity, ending with exactly one fenced json action. "
    "Never propose any nonzero movement.", fmt)
client = models.models[args.model].new_client(prompt)
try:
    print(json.dumps(client.check()), flush=True)
    with tempfile.TemporaryDirectory(prefix="pear-synthetic-") as d:
        images = []
        for cam in fmt.cameras:
            im = Image.new("RGB", (640,480), "white")
            draw = ImageDraw.Draw(im)
            if cam.role == "left":
                draw.polygon([(320,80),(180,360),(460,360)],fill="red")
            elif cam.role == "top":
                draw.ellipse((190,110,450,370),fill="green")
            elif cam.role == "right":
                draw.rectangle((140,150,500,330),fill="blue")
            else:
                raise ValueError(cam.role)
            p=Path(d)/f"{cam.role}.jpg"
            im.save(p)
            images.append(CameraImage(cam.name,p))
        for turn in (1,2):
            out=client.infer(ModelInput(json.dumps(state),tuple(images)),
                on_event=lambda *_:None,cancellation=threading.Event())
            action=ActionParser.parse(out.answer,fmt)
            assert all(v==0 for part in fmt.schema for v in getattr(action,part).model_dump().values())
            print(json.dumps({"turn":turn,"answer":out.answer,"usage":out.usage,
                "history_items":len(client.history)},ensure_ascii=False),flush=True)
    print("PASS: 2 turns, 3 synthetic views, complete history, valid zero actions; no hardware opened.",flush=True)
finally:
    client.close()
