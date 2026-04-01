import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_driven_control.preprocess import check_gpe, describe_hankel_gpe, hankel

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

# ============================================================
# User settings
# ============================================================
JOINT_ORDER = [0, 1, 2, 3, 4, 5, 6]   # excite joints sequentially
JOINT_DELTA_DEG = np.array([12.0, 12.0, 10.0, 10.0, 8.0, 8.0, 8.0], dtype=np.float64)

MOVE_DURATION = 2.0    # seconds to move away from home
RETURN_DURATION = 2.0  # seconds to move back to home
HOLD_DURATION = 0.5    # seconds to hold at target and at home between segments

# PD gains
KP = np.array([20.0, 20.0, 20.0, 15.0, 10.0, 8.0, 5.0], dtype=np.float64)
KD = np.array([2.0,  2.0,  2.0,  1.5,  1.0,  0.8,  0.5], dtype=np.float64)

# Safety torque saturation
TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

SAVE_DATA = True
SAVE_PATH = "run_data.npz"
PLOTS_ROOT = os.path.join(PROJECT_ROOT, "plots")

RUN_GPE_CHECK = True
GPE_HANKEL_DEPTH = 200
PLOT_GPE = True


# ============================================================
# Utilities
# ============================================================
def get_joint_state(model, data):
    """Return the current 7 joint positions and velocities."""
    q = np.zeros(7, dtype=np.float64)
    qd = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q[aid] = data.qpos[qpos_id]
        qd[aid] = data.qvel[dof_id]

    return q, qd


class QuinticJointPlanner:
    """
    Quintic planner for joint-space motion:
        q(0)   = q0
        q(T)   = qf
        qd(0)  = 0
        qd(T)  = 0
        qdd(0) = 0
        qdd(T) = 0
    """

    def __init__(self, q0, qf, T):
        self.q0 = np.asarray(q0, dtype=np.float64).reshape(7)
        self.qf = np.asarray(qf, dtype=np.float64).reshape(7)
        self.T = float(T)

        if self.T <= 0.0:
            raise ValueError("Trajectory duration T must be > 0")

        self.dq = self.qf - self.q0

    def evaluate(self, t):
        if t <= 0.0:
            return self.q0.copy(), np.zeros(7), np.zeros(7)

        if t >= self.T:
            return self.qf.copy(), np.zeros(7), np.zeros(7)

        s = t / self.T
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s

        pos_scale = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
        vel_scale = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / self.T
        acc_scale = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (self.T ** 2)

        q_des = self.q0 + self.dq * pos_scale
        qd_des = self.dq * vel_scale
        qdd_des = self.dq * acc_scale

        return q_des, qd_des, qdd_des


class HoldPlanner:
    """Constant hold segment."""
    def __init__(self, q_hold, T):
        self.q_hold = np.asarray(q_hold, dtype=np.float64).reshape(7)
        self.T = float(T)

    def evaluate(self, t):
        return self.q_hold.copy(), np.zeros(7), np.zeros(7)


def apply_joint_pd_control(model, data, q_des, qd_des):
    """
    PD torque control with gravity/bias compensation:
        tau = qfrc_bias + Kp(q_des - q) + Kd(qd_des - qd)
    """
    data.ctrl[:] = 0.0

    tau_bias_all = np.zeros(7, dtype=np.float64)
    tau_kp_all = np.zeros(7, dtype=np.float64)
    tau_kd_all = np.zeros(7, dtype=np.float64)
    tau_cmd_all = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q = data.qpos[qpos_id]
        qd = data.qvel[dof_id]

        tau_bias = data.qfrc_bias[dof_id]

        pos_err = q_des[aid] - q
        vel_err = qd_des[aid] - qd

        tau_kp = KP[aid] * pos_err
        tau_kd = KD[aid] * vel_err
        tau_cmd = tau_bias + tau_kp + tau_kd
        tau_cmd = np.clip(tau_cmd, -TAU_LIMITS[aid], TAU_LIMITS[aid])

        data.ctrl[aid] = tau_cmd
        tau_bias_all[aid] = tau_bias
        tau_kp_all[aid] = tau_kp
        tau_kd_all[aid] = tau_kd
        tau_cmd_all[aid] = tau_cmd

    return tau_bias_all, tau_kp_all, tau_kd_all, tau_cmd_all


def build_excitation_schedule(q_home):
    """
    Build a global schedule:
      hold at home
      for each joint:
          move to offset
          hold
          move back home
          hold
    """
    schedule = []
    t_cursor = 0.0

    # Initial settle at home
    schedule.append({
        "t_start": t_cursor,
        "t_end": t_cursor + HOLD_DURATION,
        "planner": HoldPlanner(q_home, HOLD_DURATION),
        "active_joint": -1,
        "label": "initial_hold",
    })
    t_cursor += HOLD_DURATION

    for j in JOINT_ORDER:
        q_target = q_home.copy()
        q_target[j] += np.deg2rad(JOINT_DELTA_DEG[j])

        # Move away
        schedule.append({
            "t_start": t_cursor,
            "t_end": t_cursor + MOVE_DURATION,
            "planner": QuinticJointPlanner(q_home, q_target, MOVE_DURATION),
            "active_joint": j,
            "label": f"joint_{j+1}_move_out",
        })
        t_cursor += MOVE_DURATION

        # Hold at target
        schedule.append({
            "t_start": t_cursor,
            "t_end": t_cursor + HOLD_DURATION,
            "planner": HoldPlanner(q_target, HOLD_DURATION),
            "active_joint": j,
            "label": f"joint_{j+1}_hold_out",
        })
        t_cursor += HOLD_DURATION

        # Move back
        schedule.append({
            "t_start": t_cursor,
            "t_end": t_cursor + RETURN_DURATION,
            "planner": QuinticJointPlanner(q_target, q_home, RETURN_DURATION),
            "active_joint": j,
            "label": f"joint_{j+1}_move_back",
        })
        t_cursor += RETURN_DURATION

        # Hold at home
        schedule.append({
            "t_start": t_cursor,
            "t_end": t_cursor + HOLD_DURATION,
            "planner": HoldPlanner(q_home, HOLD_DURATION),
            "active_joint": -1,
            "label": f"joint_{j+1}_home_hold",
        })
        t_cursor += HOLD_DURATION

    return schedule, t_cursor


def reference_from_schedule(t, schedule, q_home):
    """Get q_des, qd_des, qdd_des for global simulation time t."""
    for seg in schedule:
        if seg["t_start"] <= t < seg["t_end"]:
            local_t = t - seg["t_start"]
            q_des, qd_des, qdd_des = seg["planner"].evaluate(local_t)
            return q_des, qd_des, qdd_des, seg["active_joint"], seg["label"]

    return q_home.copy(), np.zeros(7), np.zeros(7), -1, "done"


# ============================================================
# Plotting
# ============================================================
def plot_joint_tracking(time_hist, q_hist, q_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(q_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(time_hist, np.rad2deg(q_des_hist[:, j]), "--", label="Reference", linewidth=1.5)
        axes[j].set_ylabel(f"q{j + 1} (deg)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Sequential Joint Excitation: Position Tracking")
    fig.tight_layout()
    return fig


def plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(qd_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(time_hist, np.rad2deg(qd_des_hist[:, j]), "--", label="Reference", linewidth=1.5)
        axes[j].set_ylabel(f"qd{j + 1} (deg/s)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Sequential Joint Excitation: Velocity Tracking")
    fig.tight_layout()
    return fig


def plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, tau_cmd_hist[:, j], label="Applied torque", linewidth=2.0)
        axes[j].plot(time_hist, tau_bias_hist[:, j], label="qfrc_bias", linewidth=1.2)
        axes[j].plot(time_hist, tau_kp_hist[:, j], label="Kp term", linewidth=1.2)
        axes[j].plot(time_hist, tau_kd_hist[:, j], label="Kd term", linewidth=1.2)
        axes[j].set_ylabel(f"tau{j + 1} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Sequential Joint Excitation: Torque Breakdown")
    fig.tight_layout()
    return fig


def plot_active_joint(time_hist, active_joint_hist):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(time_hist, active_joint_hist + 1, where="post", linewidth=2.0)
    ax.set_yticks(np.arange(0, 8), ["hold", "j1", "j2", "j3", "j4", "j5", "j6", "j7"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active segment")
    ax.set_title("Excitation Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gpe_singular_values(H_Y):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Singular Values of H_Y")
    ax.loglog(np.linalg.svd(H_Y, compute_uv=False), marker="o")
    fig.tight_layout()
    return fig


def create_output_dir(script_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PLOTS_ROOT, f"{script_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_figure(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_metadata(output_dir, metadata):
    path = os.path.join(output_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {path}")


# ============================================================
# Main
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
    data = mujoco.MjData(model)

    # Reset to home pose if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    q_home, _ = get_joint_state(model, data)
    schedule, total_time = build_excitation_schedule(q_home)

    print("Home configuration (deg):", np.rad2deg(q_home))
    print("Joint excitation deltas (deg):", JOINT_DELTA_DEG)
    print("Total excitation time (s):", total_time)

    time_hist = []
    q_hist = []
    q_des_hist = []
    qd_hist = []
    qd_des_hist = []
    tau_cmd_hist = []
    tau_bias_hist = []
    tau_kp_hist = []
    tau_kd_hist = []
    active_joint_hist = []
    label_hist = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time <= total_time:
            t = data.time

            q_des, qd_des, qdd_des, active_joint, label = reference_from_schedule(t, schedule, q_home)

            mujoco.mj_forward(model, data)
            q, qd = get_joint_state(model, data)
            tau_bias, tau_kp, tau_kd, tau_cmd = apply_joint_pd_control(model, data, q_des, qd_des)

            time_hist.append(t)
            q_hist.append(q.copy())
            q_des_hist.append(q_des.copy())
            qd_hist.append(qd.copy())
            qd_des_hist.append(qd_des.copy())
            tau_cmd_hist.append(tau_cmd.copy())
            tau_bias_hist.append(tau_bias.copy())
            tau_kp_hist.append(tau_kp.copy())
            tau_kd_hist.append(tau_kd.copy())
            active_joint_hist.append(active_joint)
            label_hist.append(label)

            mujoco.mj_step(model, data)
            viewer.sync()

    if not time_hist:
        print("No simulation data recorded.")
        return

    time_hist = np.asarray(time_hist, dtype=np.float64)
    q_hist = np.asarray(q_hist, dtype=np.float64)
    q_des_hist = np.asarray(q_des_hist, dtype=np.float64)
    qd_hist = np.asarray(qd_hist, dtype=np.float64)
    qd_des_hist = np.asarray(qd_des_hist, dtype=np.float64)
    tau_cmd_hist = np.asarray(tau_cmd_hist, dtype=np.float64)
    tau_bias_hist = np.asarray(tau_bias_hist, dtype=np.float64)
    tau_kp_hist = np.asarray(tau_kp_hist, dtype=np.float64)
    tau_kd_hist = np.asarray(tau_kd_hist, dtype=np.float64)
    active_joint_hist = np.asarray(active_joint_hist, dtype=np.int32)

    # Data-driven arrays:
    # U should be shape (m, N), Y should be shape (p, N)
    U = tau_cmd_hist.T            # (7, N)
    Y = q_hist.T                  # (7, N)

    print("Collected data:")
    print("  U shape:", U.shape)
    print("  Y shape:", Y.shape)
    print("  Number of samples:", U.shape[1])

    output_dir = create_output_dir("analysisV1")
    gpe_result = {
        "checked": False,
        "hankel_depth": int(GPE_HANKEL_DEPTH),
        "plot_saved": False,
        "gpe_satisfied": None,
        "rank_H_U": None,
        "rank_H_Y": None,
        "hankel_diagnostics": None,
        "message": "GPE check disabled.",
    }

    if RUN_GPE_CHECK:
        print("Checking persistence of excitation...")
        L = GPE_HANKEL_DEPTH
        num_samples = U.shape[1]

        if L > num_samples:
            hankel_diagnostics = describe_hankel_gpe(U, Y, L)
            for line in hankel_diagnostics["lines"]:
                print(line)
            gpe_result["message"] = hankel_diagnostics["summary"]
            gpe_result["hankel_diagnostics"] = hankel_diagnostics
        else:
            H_U, H_Y = hankel(U, Y, L)
            gpe_satisfied, rank_H_U, rank_H_Y = check_gpe(H_U, H_Y, plot=False)
            hankel_diagnostics = describe_hankel_gpe(
                U, Y, L, H_U=H_U, H_Y=H_Y, rank_H_U=rank_H_U, rank_H_Y=rank_H_Y, gpe_satisfied=gpe_satisfied
            )
            for line in hankel_diagnostics["lines"]:
                print(line)
            print("GPE satisfied:", gpe_satisfied)
            gpe_result = {
                "checked": True,
                "hankel_depth": int(L),
                "plot_saved": bool(PLOT_GPE),
                "gpe_satisfied": bool(gpe_satisfied),
                "rank_H_U": int(rank_H_U),
                "rank_H_Y": int(rank_H_Y),
                "hankel_diagnostics": hankel_diagnostics,
                "message": hankel_diagnostics["summary"],
            }

            if PLOT_GPE:
                save_figure(plot_gpe_singular_values(H_Y), output_dir, "gpe_singular_values.png")

    if SAVE_DATA:
        save_path = os.path.join(output_dir, SAVE_PATH)
        np.savez(
            save_path,
            time=time_hist,
            U=U,
            Y=Y,
            q_hist=q_hist,
            q_des_hist=q_des_hist,
            qd_hist=qd_hist,
            qd_des_hist=qd_des_hist,
            tau_cmd_hist=tau_cmd_hist,
            tau_bias_hist=tau_bias_hist,
            tau_kp_hist=tau_kp_hist,
            tau_kd_hist=tau_kd_hist,
            active_joint_hist=active_joint_hist,
            label_hist=np.array(label_hist, dtype=object),
            q_home=q_home,
            joint_order=np.array(JOINT_ORDER, dtype=np.int32),
            joint_delta_deg=JOINT_DELTA_DEG,
            move_duration=MOVE_DURATION,
            return_duration=RETURN_DURATION,
            hold_duration=HOLD_DURATION,
            kp=KP,
            kd=KD,
            tau_limits=TAU_LIMITS,
        )
        print(f"Saved dataset to: {os.path.abspath(save_path)}")

    metadata = {
        "script": "analysisV1.py",
        "output_dir": output_dir,
        "xml_file_path": XML_FILE_PATH,
        "settings": {
            "joint_order": list(JOINT_ORDER),
            "joint_delta_deg": JOINT_DELTA_DEG.tolist(),
            "move_duration": float(MOVE_DURATION),
            "return_duration": float(RETURN_DURATION),
            "hold_duration": float(HOLD_DURATION),
            "kp": KP.tolist(),
            "kd": KD.tolist(),
            "tau_limits": TAU_LIMITS.tolist(),
            "run_gpe_check": bool(RUN_GPE_CHECK),
            "gpe_hankel_depth": int(GPE_HANKEL_DEPTH),
            "save_data": bool(SAVE_DATA),
        },
        "results": {
            "num_samples": int(U.shape[1]),
            "U_shape": list(U.shape),
            "Y_shape": list(Y.shape),
            "home_configuration_deg": np.rad2deg(q_home).tolist(),
            "gpe": gpe_result,
        },
    }
    save_metadata(output_dir, metadata)

    save_figure(plot_joint_tracking(time_hist, q_hist, q_des_hist), output_dir, "position_tracking.png")
    save_figure(plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist), output_dir, "velocity_tracking.png")
    save_figure(plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist), output_dir, "torque_breakdown.png")
    save_figure(plot_active_joint(time_hist, active_joint_hist), output_dir, "excitation_schedule.png")


if __name__ == "__main__":
    main()
