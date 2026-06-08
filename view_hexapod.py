import mujoco
import mujoco.viewer
import os

xml_path = os.path.join(
    os.path.dirname(__file__),
    "one_policy_to_run_them_all",
    "one_policy_to_run_them_all",
    "environments",
    "hexapod",
    "data",
    "hexapod.xml"
)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
