import mujoco
import mujoco.viewer
import numpy as np

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

JOINT_TO_MOVE = 0   # 0 = joint1
VEL_GAIN = 0.5
VEL_AMPLITUDE = 3


def apply_control(model, data):
    """
    Always apply bias compensation.
    On one joint, add velocity tracking torque for a sinusoidal velocity.
    """
    data.ctrl[:] = 0.0

    # Simple sinusoidal desired velocity
    qd_des = VEL_AMPLITUDE * np.sin(data.time)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]

        # Always compensate bias/gravity at current state
        tau_bias = data.qfrc_bias[dof_id]

        if aid == JOINT_TO_MOVE:
            qd = data.qvel[dof_id]
            tau_cmd = tau_bias + VEL_GAIN * (qd_des - qd)
        else:
            tau_cmd = tau_bias

        data.ctrl[aid] = tau_cmd


# Load model
model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
data = mujoco.MjData(model)

# Reset to home pose if available
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.mj_forward(model, data)

# Run viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        apply_control(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()