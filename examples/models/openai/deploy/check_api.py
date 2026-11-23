"""Send saved station snapshots to an API model; never open robot hardware."""

import argparse
import json
import threading
from pathlib import Path

from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput
from pear.deploy.station.agent_loop.config import ModelsConfig
from pear.deploy.station.agent_loop.messages import SystemPrompt
from pear.deploy.station.control.robot import BaseRobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--model", default="gpt_astra")
    args = parser.parse_args()

    config = BaseRobot.load_config(args.config, [])
    models = ModelsConfig.from_station_config(config)
    models.require(args.model)
    action_format = models.action_format(args.model)
    camera_config = {camera.role: camera.serial for camera in config.robot.cameras}
    images = []
    for camera in action_format.cameras:
        path = args.snapshot_dir / f"{camera_config[camera.role]}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(CameraImage(camera.name, path))
    state = {part: dict.fromkeys(names, 0.0) for part, names in action_format.schema.items()}
    prompt = SystemPrompt.render(
        "This is an API integration check using saved images and a synthetic state, not a live task. "
        "Return zero change for every arm and gripper quantity, ending with exactly one fenced json action.",
        action_format,
    )
    client = models.models[args.model].new_client(prompt)
    try:
        print(json.dumps(client.check()), flush=True)
        output = client.infer(
            ModelInput(json.dumps(state), tuple(images)),
            on_event=lambda *_: None,
            cancellation=threading.Event(),
        )
        action = ActionParser.parse(output.answer, action_format)
        if any(value != 0 for part in action_format.schema for value in getattr(action, part).model_dump().values()):
            raise ValueError("Expected a zero-delta action for the static API check")
        print("Validated action:", action.model_dump_json())
        print("Usage:", json.dumps(output.usage))
        print("Static API check passed; no hardware was opened.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
