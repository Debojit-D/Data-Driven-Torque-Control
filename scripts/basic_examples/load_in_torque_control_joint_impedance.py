import mujoco
import mujoco.viewer
import numpy as np

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

JOINT_TO_MOVE = 0       # 0 = joint1
VEL_GAIN = 5.0
VEL_AMPLITUDE = 0.5

KP_HOLD = 80.0
KD_HOLD = 20.0


def apply_control(model, data, q_home):
    """
    Bias compensation always.
    Selected joint: sinusoidal velocity tracking.
    Other joints: hold initial posture with PD.
    """
    data.ctrl[:] = 0.0

    qd_des = VEL_AMPLITUDE * np.sin(data.time)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]

        tau_bias = data.qfrc_bias[dof_id]
        q = data.qpos[joint_id]
        qd = data.qvel[dof_id]

        if aid == JOINT_TO_MOVE:
            # velocity control on one joint
            tau_cmd = tau_bias + VEL_GAIN * (qd_des - qd)
        else:
            # posture hold on all other joints
            q_err = q_home[joint_id] - q
            qd_err = -qd
            tau_cmd = tau_bias + KP_HOLD * q_err + KD_HOLD * qd_err

        data.ctrl[aid] = tau_cmd


# Load model
model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
data = mujoco.MjData(model)

# Reset to home pose if available
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.mj_forward(model, data)

# Store initial posture to hold
q_home = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        apply_control(model, data, q_home)
        mujoco.mj_step(model, data)
        viewer.sync()