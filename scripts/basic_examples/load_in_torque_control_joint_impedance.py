import mujoco
import mujoco.viewer
import numpy as np

from utils.data_driven_control.preprocess import hankel, check_gpe

XML_FILE_PATH = "robot_descriptions/franka_emika_panda/scene.xml"

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

dt = model.opt.timestep
T_final = 10.0   # seconds
N = int(T_final / dt)
L = 200  # depth of Hankel matrix

m = model.nu          # number of actuators
p = model.nq          # number of joint positions

U = np.zeros((m, N))  # inputs
Y = np.zeros((p, N))  # outputs
time = np.zeros(N)

# Reset to home pose if available
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.mj_forward(model, data)

# Store initial posture to hold
q_home = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    for k in range(N):
        mujoco.mj_forward(model, data)
        apply_control(model, data, q_home)

        U[:, k] = data.ctrl.copy()
        Y[:, k] = data.qpos.copy()
        time[k] = data.time

        mujoco.mj_step(model, data)
        viewer.sync()

print("Simulation completed.")
print("Input data shape (U):", U.shape)
print("Output data shape (Y):", Y.shape)

print("Checking persistence of excitation...")

H_U, H_Y = hankel(U, Y, L)
gpe_satisfied, rank_H_U = check_gpe(H_U)
print("GPE satisfied:", gpe_satisfied)
print("Rank of Hankel matrix for inputs (H_U):", rank_H_U)