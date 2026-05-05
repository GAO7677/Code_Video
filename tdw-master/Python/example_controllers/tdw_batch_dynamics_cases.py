import sys
from pathlib import Path
from typing import Dict, List

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.output_data import Bounds, OutputData, Rigidbodies
from tdw.tdw_utils import TDWUtils


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
PORT = 1071


class DynamicsCaseRunner:
    def __init__(self) -> None:
        self.output_root = OUTPUT_ROOT
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _new_controller(self) -> Controller:
        return Controller(launch_build=False, port=PORT)

    def _terminate(self, controller: Controller) -> None:
        try:
            controller.communicate({"$type": "terminate"})
        except Exception:
            pass

    def rigid_drop(self) -> None:
        case_dir = self.output_root.joinpath("rigid_drop")
        controller = self._new_controller()
        try:
            cam = ThirdPersonCamera(avatar_id="a",
                                    position={"x": 2.6, "y": 1.8, "z": -2.8},
                                    look_at={"x": 0, "y": 0.8, "z": 0})
            capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
            controller.add_ons.extend([cam, capture])
            object_id = controller.get_unique_id()
            resp = controller.communicate([TDWUtils.create_empty_room(12, 12),
                                           controller.get_add_object(model_name="iron_box",
                                                                     object_id=object_id,
                                                                     position={"x": 0, "y": 2.5, "z": 0}),
                                           {"$type": "send_rigidbodies",
                                            "frequency": "always"}])
            for _ in range(180):
                sleeping = False
                for i in range(len(resp) - 1):
                    if OutputData.get_data_type_id(resp[i]) == "rigi":
                        rigidbodies = Rigidbodies(resp[i])
                        for j in range(rigidbodies.get_num()):
                            if rigidbodies.get_id(j) == object_id:
                                sleeping = rigidbodies.get_sleeping(j)
                                break
                if sleeping:
                    break
                resp = controller.communicate([])
            self._terminate(controller)
        finally:
            pass

    def object_on_table(self) -> None:
        case_dir = self.output_root.joinpath("object_on_table")
        controller = self._new_controller()
        try:
            cam = ThirdPersonCamera(avatar_id="a",
                                    position={"x": 2.478, "y": 1.602, "z": 1.412},
                                    look_at={"x": 0, "y": 0.5, "z": 0})
            capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
            controller.add_ons.extend([cam, capture])
            table_id = controller.get_unique_id()
            resp = controller.communicate([TDWUtils.create_empty_room(12, 12),
                                           controller.get_add_object(model_name="small_table_green_marble",
                                                                     position={"x": 0, "y": 0, "z": 0},
                                                                     object_id=table_id),
                                           {"$type": "send_bounds",
                                            "frequency": "once"}])
            top = {"x": 0, "y": 0.75, "z": 0}
            for i in range(len(resp) - 1):
                if OutputData.get_data_type_id(resp[i]) == "boun":
                    bounds = Bounds(resp[i])
                    for j in range(bounds.get_num()):
                        if bounds.get_id(j) == table_id:
                            top = TDWUtils.array_to_vector3(bounds.get_top(j))
                            break
            top["y"] += 0.05
            controller.communicate(controller.get_add_object(model_name="jug01",
                                                             position=top,
                                                             object_id=controller.get_unique_id()))
            for _ in range(90):
                controller.communicate([])
            self._terminate(controller)
        finally:
            pass

    def flex_soft_body(self) -> None:
        case_dir = self.output_root.joinpath("flex_soft_body")
        controller = self._new_controller()
        try:
            cam = ThirdPersonCamera(avatar_id="a",
                                    position={"x": 3.83, "y": 3.6, "z": -0.71},
                                    look_at={"x": 0, "y": 0, "z": 0})
            capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
            controller.add_ons.extend([cam, capture])
            cube_id = controller.get_unique_id()
            dumbbell_id = controller.get_unique_id()
            controller.communicate([TDWUtils.create_empty_room(12, 12),
                                    {"$type": "convexify_proc_gen_room"},
                                    {"$type": "create_flex_container"},
                                    controller.get_add_object(model_name="cube",
                                                              object_id=cube_id,
                                                              library="models_flex.json",
                                                              position={"x": 0, "y": 0, "z": 0}),
                                    {"$type": "set_flex_solid_actor",
                                     "id": cube_id,
                                     "mass_scale": 50,
                                     "particle_spacing": 0.125},
                                    {"$type": "assign_flex_container",
                                     "id": cube_id,
                                     "container_id": 0},
                                    controller.get_add_object(model_name="dumbbell",
                                                              object_id=dumbbell_id,
                                                              library="models_flex.json",
                                                              position={"x": 0.25, "y": 2, "z": 0}),
                                    {"$type": "set_flex_soft_actor",
                                     "id": dumbbell_id,
                                     "particle_spacing": 0.05,
                                     "cluster_stiffness": 0.5,
                                     "mass_scale": 1},
                                    {"$type": "assign_flex_container",
                                     "id": dumbbell_id,
                                     "container_id": 0}])
            for _ in range(180):
                controller.communicate([])
            self._terminate(controller)
        finally:
            pass

    def run(self) -> Dict[str, str]:
        cases: List = [self.rigid_drop, self.object_on_table, self.flex_soft_body]
        results: Dict[str, str] = {}
        for case in cases:
            case_name = case.__name__
            print(f"Running {case_name}")
            case()
            results[case_name] = str(self.output_root.joinpath(case_name).resolve())
        return results


if __name__ == "__main__":
    runner = DynamicsCaseRunner()
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
        if not hasattr(runner, case_name):
            raise ValueError(f"Unknown case: {case_name}")
        getattr(runner, case_name)()
        outputs = {case_name: str(runner.output_root.joinpath(case_name).resolve())}
    else:
        outputs = runner.run()
    print(outputs)
