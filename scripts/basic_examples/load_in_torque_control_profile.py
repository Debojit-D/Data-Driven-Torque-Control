import mujoco
import mujoco.viewer
import numpy as np

from utils.data_driven_control.preprocess import hankel, check_gpe

XML_FILE_PATH = "robot_descriptions/franka_emika_panda/scene.xml"
PLOT = True

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
    qd_des = VEL_AMPLITUDE * np.sin(data.time) + np.random.normal(0, 0.2)  # add some noise for excitation

    for aid in range(8):
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

dt = model.opt.timestep
T_final = 10.0   # seconds
N = int(T_final / dt)
L = 150  # depth of Hankel matrix

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

# Run viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # while viewer.is_running():
    for k in range(N):
        mujoco.mj_forward(model, data)
        apply_control(model, data)

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
gpe_satisfied, rank_H_U, rank_H_Y = check_gpe(H_U, H_Y, plot=PLOT)
print("GPE satisfied:", gpe_satisfied)
print("Rank of Hankel matrix for inputs (H_U):", rank_H_U)
print("Rank of Hankel matrix for outputs (H_Y):", rank_H_Y)