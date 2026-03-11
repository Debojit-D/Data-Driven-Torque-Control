# https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model

import mujoco
import mujoco.viewer

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"


def compensate_gravity(model, data):
    """
    Apply gravity compensation to the 7 Panda arm joints.
    Assumes actuator0..actuator6 correspond to joint1..joint7.
    """
    data.ctrl[:] = 0.0

    for aid in range(7):  # only arm actuators
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]
        data.ctrl[aid] = data.qfrc_bias[dof_id]


# Load model and data
model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
data = mujoco.MjData(model)

# Optional: reset to home keyframe if present
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.mj_forward(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # compute current bias forces
        mujoco.mj_forward(model, data)

        # apply gravity compensation
        compensate_gravity(model, data)

        # step simulation
        mujoco.mj_step(model, data)

        # refresh viewer
        viewer.sync()