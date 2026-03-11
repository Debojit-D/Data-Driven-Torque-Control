# https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model

import mujoco
import mujoco.viewer
import numpy as np

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

# Constant user torques for the 7 Panda arm joints
tau_user = np.array([
    0.05,  # joint1
    0.0,  # joint2
    0.0,  # joint3
    0.0,  # joint4
    0.0,  # joint5
    0.0,  # joint6
    0.0   # joint7
], dtype=np.float64)


def apply_torque_with_bias_comp(model, data, tau_user):
    """
    Apply MuJoCo bias compensation + constant user torque
    to the 7 Panda arm actuators.
    """
    data.ctrl[:] = 0.0

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]

        tau_bias = data.qfrc_bias[dof_id]
        data.ctrl[aid] = tau_bias + tau_user[aid]


# Load model and data
model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
data = mujoco.MjData(model)

# Reset to home keyframe if present
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

# Forward once to initialize dynamics
mujoco.mj_forward(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Update dynamics for current state
        mujoco.mj_forward(model, data)

        # Apply bias compensation + user torque
        apply_torque_with_bias_comp(model, data, tau_user)

        # Step simulation
        mujoco.mj_step(model, data)

        # Refresh viewer
        viewer.sync()