from typing import Callable, Optional

import zmq

from tdw.controller import Controller
from tdw.output_data import OutputData, Version


class ManualBuildController(Controller):
    """
    A TDW controller that binds the REP socket before launching the build.
    This avoids the launch-order race where the build starts before the Python controller
    is ready to receive the initial handshake.
    """

    def __init__(self, port: int, build_callback: Optional[Callable[[], None]] = None):
        self.add_ons = []
        context = zmq.Context()
        self._context = context
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        if build_callback is not None:
            build_callback()
        # Wait for the build handshake.
        self.socket.recv()
        resp = self.communicate([{"$type": "set_error_handling"},
                                 {"$type": "send_version"},
                                 {"$type": "load_scene",
                                  "scene_name": "ProcGenScene"}])
        self._is_standalone = False
        self._tdw_version = ""
        self._unity_version = ""
        for r in resp[:-1]:
            if OutputData.get_data_type_id(r) == "vers":
                v = Version(r)
                self._tdw_version = v.get_tdw_version()
                self._unity_version = v.get_unity_version()
                self._is_standalone = v.get_standalone()
                break
