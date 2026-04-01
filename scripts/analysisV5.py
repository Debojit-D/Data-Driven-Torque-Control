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

NUM_JOINTS = 7

# Total duration of excitation
EXCITATION_DURATION = 40.0  # seconds

# Smooth ramp-in / ramp-out duration
RAMP_TIME = 3.0  # seconds

# Safety torque saturation
TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

# If True: applied torque = qfrc_bias + excitation torque
# If False: applied torque = excitation torque only
USE_GRAVITY_COMPENSATION = True

# Which input should be used for the GPE test?
# "excitation_only" -> independent designed excitation torque
# "total_applied"   -> total torque written into data.ctrl
GPE_INPUT_CHOICE = "excitation_only"

# Save dataset
SAVE_DATA = True
SAVE_PATH = "run_data.npz"
PLOTS_ROOT = os.path.join(PROJECT_ROOT, "plots")

# GPE / Hankel settings
RUN_GPE_CHECK = True
GPE_HANKEL_DEPTH = 200
PLOT_GPE = True

# Output choice for data-driven pipeline
# False -> Y = q
# True  -> Y = [q; qd]
USE_VELOCITY_IN_OUTPUT = False

# ============================================================
# Direct torque-space multisine design
# ============================================================
# Shape must be (7, K), where K = number of sine components per joint.

TORQUE_COMPONENT_AMPLITUDES_NM = np.array([
    [4.0, 3.0, 2.5, 2.0],   # joint 1
    [4.0, 3.0, 2.5, 2.0],   # joint 2
    [3.5, 2.5, 2.0, 1.5],   # joint 3
    [3.0, 2.0, 1.5, 1.0],   # joint 4
    [0.9, 0.7, 0.5, 0.4],   # joint 5
    [0.9, 0.7, 0.5, 0.4],   # joint 6
    [0.8, 0.6, 0.5, 0.4],   # joint 7
], dtype=np.float64)

TORQUE_COMPONENT_FREQUENCIES_HZ = np.array([
    [0.17, 0.41, 0.73, 1.11],
    [0.23, 0.53, 0.89, 1.19],
    [0.31, 0.67, 0.97, 1.27],
    [0.37, 0.61, 1.03, 1.33],
    [0.29, 0.59, 0.83, 1.39],
    [0.43, 0.71, 1.07, 1.47],
    [0.47, 0.79, 1.13, 1.51],
], dtype=np.float64)

TORQUE_COMPONENT_PHASES_DEG = np.array([
    [0.0,   37.0, 113.0, 191.0],
    [19.0,  71.0, 149.0, 233.0],
    [41.0, 101.0, 173.0, 257.0],
    [59.0, 127.0, 211.0, 281.0],
    [83.0, 157.0, 239.0, 307.0],
    [97.0, 181.0, 263.0, 331.0],
    [109.0, 193.0, 277.0, 347.0],
], dtype=np.float64)

# Optional DC torque offset for each joint
TORQUE_DC_OFFSET_NM = np.zeros(NUM_JOINTS, dtype=np.float64)


# ============================================================
# Utilities
# ============================================================
def get_joint_maps(model):
    """
    Returns arrays of qpos indices and dof indices for the first 7 actuators.
    Assumes actuator order matches the 7 Panda joints.
    """
    if model.nu < NUM_JOINTS:
        raise ValueError(f"Model has only {model.nu} actuators, expected at least {NUM_JOINTS}.")

    qpos_ids = np.zeros(NUM_JOINTS, dtype=np.int32)
    dof_ids = np.zeros(NUM_JOINTS, dtype=np.int32)

    for aid in range(NUM_JOINTS):
        joint_id = model.actuator_trnid[aid, 0]
        qpos_ids[aid] = model.jnt_qposadr[joint_id]
        dof_ids[aid] = model.jnt_dofadr[joint_id]

    return qpos_ids, dof_ids


def get_joint_state(data, qpos_ids, dof_ids):
    """Return the current 7 joint positions and velocities."""
    q = data.qpos[qpos_ids].copy()
    qd = data.qvel[dof_ids].copy()
    return q, qd


def get_bias_torque(data, dof_ids):
    """Return qfrc_bias for the 7 actuated joints."""
    return data.qfrc_bias[dof_ids].copy()


def smooth_envelope(t, T, T_ramp):
    """
    Smooth amplitude envelope e(t) with smoothstep ramp-in and ramp-out.
    Returns:
        e   : envelope
        de  : first derivative
        dde : second derivative
    """
    if T_ramp <= 0.0:
        return 1.0, 0.0, 0.0

    if t <= 0.0:
        return 0.0, 0.0, 0.0

    if t < T_ramp:
        s = t / T_ramp
        e = 3.0 * s**2 - 2.0 * s**3
        de = (6.0 * s - 6.0 * s**2) / T_ramp
        dde = (6.0 - 12.0 * s) / (T_ramp**2)
        return e, de, dde

    if t <= T - T_ramp:
        return 1.0, 0.0, 0.0

    if t < T:
        s = (T - t) / T_ramp
        e = 3.0 * s**2 - 2.0 * s**3
        de = -(6.0 * s - 6.0 * s**2) / T_ramp
        dde = (6.0 - 12.0 * s) / (T_ramp**2)
        return e, de, dde

    return 0.0, 0.0, 0.0


class MultiSineTorquePlanner:
    """
    Direct torque-space multisine excitation:

        tau_exc(t) = tau_dc + e(t) * sum_k A[:, k] * sin(w[:, k] * t + phi[:, k])

    where:
        - tau_dc is a 7x1 DC offset vector
        - A, w, phi are 7xK arrays
        - e(t) is a smooth ramp-in/ramp-out envelope
    """

    def __init__(
        self,
        amplitudes_nm,
        frequencies_hz,
        phases_deg,
        T,
        T_ramp,
        dc_offset_nm=None,
    ):
        self.A = np.asarray(amplitudes_nm, dtype=np.float64)
        self.f = np.asarray(frequencies_hz, dtype=np.float64)
        self.w = 2.0 * np.pi * self.f
        self.phi = np.deg2rad(np.asarray(phases_deg, dtype=np.float64))
        self.T = float(T)
        self.T_ramp = float(T_ramp)

        if self.A.shape != self.f.shape or self.A.shape != self.phi.shape:
            raise ValueError(
                "TORQUE_COMPONENT_AMPLITUDES_NM, TORQUE_COMPONENT_FREQUENCIES_HZ, "
                "and TORQUE_COMPONENT_PHASES_DEG must all have the same shape."
            )

        if self.A.shape[0] != NUM_JOINTS:
            raise ValueError(f"Expected first dimension = {NUM_JOINTS}, got {self.A.shape[0]}.")

        if dc_offset_nm is None:
            self.tau_dc = np.zeros(NUM_JOINTS, dtype=np.float64)
        else:
            self.tau_dc = np.asarray(dc_offset_nm, dtype=np.float64).reshape(NUM_JOINTS)

        if self.T <= 0.0:
            raise ValueError("EXCITATION_DURATION must be > 0.")

        if self.T_ramp < 0.0:
            raise ValueError("RAMP_TIME must be >= 0.")

    def evaluate(self, t):
        """
        Returns the designed excitation torque only (not including gravity compensation).
        """
        if t <= 0.0 or t >= self.T:
            return self.tau_dc.copy()

        e, _, _ = smooth_envelope(t, self.T, self.T_ramp)
        theta = self.w * t + self.phi
        tau_osc = np.sum(self.A * np.sin(theta), axis=1)
        tau_exc = self.tau_dc + e * tau_osc
        return tau_exc


def apply_direct_torque_excitation(data, tau_exc_cmd, tau_bias):
    """
    Writes torque directly to data.ctrl.

    If USE_GRAVITY_COMPENSATION:
        tau_total = tau_bias + tau_exc_cmd
    else:
        tau_total = tau_exc_cmd

    After saturation:
        tau_exc_applied = tau_total_applied - tau_base

    Returns:
        tau_base          : baseline torque added before excitation
        tau_exc_applied   : actual excitation component delivered after clipping
        tau_total_applied : actual total torque written to actuators
    """
    if USE_GRAVITY_COMPENSATION:
        tau_base = tau_bias.copy()
    else:
        tau_base = np.zeros(NUM_JOINTS, dtype=np.float64)

    tau_total_nominal = tau_base + tau_exc_cmd
    tau_total_applied = np.clip(tau_total_nominal, -TAU_LIMITS, TAU_LIMITS)
    tau_exc_applied = tau_total_applied - tau_base

    data.ctrl[:NUM_JOINTS] = tau_total_applied

    return tau_base, tau_exc_applied, tau_total_applied


# ============================================================
# Plotting
# ============================================================
def plot_joint_positions(time_hist, q_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(NUM_JOINTS):
        axes[j].plot(time_hist, np.rad2deg(q_hist[:, j]), linewidth=2.0)
        axes[j].set_ylabel(f"q{j + 1} (deg)")
        axes[j].grid(True, alpha=0.3)

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Direct Torque Excitation: Joint Positions")
    fig.tight_layout()
    return fig


def plot_joint_velocities(time_hist, qd_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(NUM_JOINTS):
        axes[j].plot(time_hist, np.rad2deg(qd_hist[:, j]), linewidth=2.0)
        axes[j].set_ylabel(f"qd{j + 1} (deg/s)")
        axes[j].grid(True, alpha=0.3)

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Direct Torque Excitation: Joint Velocities")
    fig.tight_layout()
    return fig


def plot_torque_breakdown(time_hist, tau_exc_cmd_hist, tau_exc_applied_hist, tau_base_hist, tau_total_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(NUM_JOINTS):
        axes[j].plot(time_hist, tau_exc_cmd_hist[:, j], label="Excitation commanded", linewidth=1.5)
        axes[j].plot(time_hist, tau_exc_applied_hist[:, j], label="Excitation applied", linewidth=2.0)
        axes[j].plot(time_hist, tau_base_hist[:, j], label="Base torque", linewidth=1.2)
        axes[j].plot(time_hist, tau_total_hist[:, j], label="Total applied", linewidth=1.2)
        axes[j].set_ylabel(f"tau{j + 1} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Direct Torque Excitation: Torque Breakdown")
    fig.tight_layout()
    return fig


def plot_reference_spectrum_info():
    num_components = TORQUE_COMPONENT_AMPLITUDES_NM.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    total_amp_per_joint = np.sum(np.abs(TORQUE_COMPONENT_AMPLITUDES_NM), axis=1)
    axes[0].bar(np.arange(1, NUM_JOINTS + 1), total_amp_per_joint)
    axes[0].set_ylabel("Sum of |amplitudes| (Nm)")
    axes[0].set_title("Torque Multisine Amplitude Budget per Joint")
    axes[0].grid(True, alpha=0.3)

    for k in range(num_components):
        axes[1].scatter(
            np.arange(1, NUM_JOINTS + 1),
            TORQUE_COMPONENT_FREQUENCIES_HZ[:, k],
            label=f"Comp {k + 1}",
            s=40,
        )

    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Joint index")
    axes[1].set_title("Torque Multisine Frequencies per Joint")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    return fig


def plot_gpe_singular_values(H_U, H_Y):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    s_u = np.linalg.svd(H_U, compute_uv=False)
    s_y = np.linalg.svd(H_Y, compute_uv=False)

    axes[0].loglog(s_u, marker="o")
    axes[0].set_title("Singular Values of H_U")
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(s_y, marker="o")
    axes[1].set_title("Singular Values of H_Y")
    axes[1].grid(True, alpha=0.3)

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

    qpos_ids, dof_ids = get_joint_maps(model)

    # Reset to home pose if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    q_home, qd_home = get_joint_state(data, qpos_ids, dof_ids)

    planner = MultiSineTorquePlanner(
        amplitudes_nm=TORQUE_COMPONENT_AMPLITUDES_NM,
        frequencies_hz=TORQUE_COMPONENT_FREQUENCIES_HZ,
        phases_deg=TORQUE_COMPONENT_PHASES_DEG,
        T=EXCITATION_DURATION,
        T_ramp=RAMP_TIME,
        dc_offset_nm=TORQUE_DC_OFFSET_NM,
    )

    print("Home configuration (deg):", np.rad2deg(q_home))
    print("Initial joint velocity (deg/s):", np.rad2deg(qd_home))
    print("Excitation duration (s):", EXCITATION_DURATION)
    print("Ramp time (s):", RAMP_TIME)
    print("Use gravity compensation:", USE_GRAVITY_COMPENSATION)
    print("GPE input choice:", GPE_INPUT_CHOICE)
    print("Torque multisine amplitudes shape:", TORQUE_COMPONENT_AMPLITUDES_NM.shape)
    print("Torque multisine frequencies shape:", TORQUE_COMPONENT_FREQUENCIES_HZ.shape)
    print("Torque multisine phases shape:", TORQUE_COMPONENT_PHASES_DEG.shape)
    print("Torque DC offsets (Nm):", TORQUE_DC_OFFSET_NM)

    time_hist = []
    q_hist = []
    qd_hist = []
    tau_bias_hist = []
    tau_base_hist = []
    tau_exc_cmd_hist = []
    tau_exc_applied_hist = []
    tau_total_hist = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time <= EXCITATION_DURATION:
            t = data.time

            mujoco.mj_forward(model, data)

            q, qd = get_joint_state(data, qpos_ids, dof_ids)
            tau_bias = get_bias_torque(data, dof_ids)
            tau_exc_cmd = planner.evaluate(t)
            tau_base, tau_exc_applied, tau_total = apply_direct_torque_excitation(
                data=data,
                tau_exc_cmd=tau_exc_cmd,
                tau_bias=tau_bias,
            )

            time_hist.append(t)
            q_hist.append(q.copy())
            qd_hist.append(qd.copy())
            tau_bias_hist.append(tau_bias.copy())
            tau_base_hist.append(tau_base.copy())
            tau_exc_cmd_hist.append(tau_exc_cmd.copy())
            tau_exc_applied_hist.append(tau_exc_applied.copy())
            tau_total_hist.append(tau_total.copy())

            mujoco.mj_step(model, data)
            viewer.sync()

    if not time_hist:
        print("No simulation data recorded.")
        return

    time_hist = np.asarray(time_hist, dtype=np.float64)
    q_hist = np.asarray(q_hist, dtype=np.float64)
    qd_hist = np.asarray(qd_hist, dtype=np.float64)
    tau_bias_hist = np.asarray(tau_bias_hist, dtype=np.float64)
    tau_base_hist = np.asarray(tau_base_hist, dtype=np.float64)
    tau_exc_cmd_hist = np.asarray(tau_exc_cmd_hist, dtype=np.float64)
    tau_exc_applied_hist = np.asarray(tau_exc_applied_hist, dtype=np.float64)
    tau_total_hist = np.asarray(tau_total_hist, dtype=np.float64)

    # Data-driven arrays
    if GPE_INPUT_CHOICE == "excitation_only":
        U = tau_exc_applied_hist.T
    elif GPE_INPUT_CHOICE == "total_applied":
        U = tau_total_hist.T
    else:
        raise ValueError("GPE_INPUT_CHOICE must be either 'excitation_only' or 'total_applied'.")

    if USE_VELOCITY_IN_OUTPUT:
        Y = np.vstack((q_hist.T, qd_hist.T))
    else:
        Y = q_hist.T

    print("Collected data:")
    print("  U shape:", U.shape)
    print("  Y shape:", Y.shape)
    print("  Number of samples:", U.shape[1])

    output_dir = create_output_dir("analysisV3_torque")

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
                save_figure(
                    plot_gpe_singular_values(H_U, H_Y),
                    output_dir,
                    "gpe_singular_values.png",
                )

    if SAVE_DATA:
        save_path = os.path.join(output_dir, SAVE_PATH)
        np.savez(
            save_path,
            time=time_hist,
            U=U,
            Y=Y,
            q_hist=q_hist,
            qd_hist=qd_hist,
            tau_bias_hist=tau_bias_hist,
            tau_base_hist=tau_base_hist,
            tau_exc_cmd_hist=tau_exc_cmd_hist,
            tau_exc_applied_hist=tau_exc_applied_hist,
            tau_total_hist=tau_total_hist,
            q_home=q_home,
            tau_limits=TAU_LIMITS,
            use_gravity_compensation=USE_GRAVITY_COMPENSATION,
            gpe_input_choice=GPE_INPUT_CHOICE,
            use_velocity_in_output=USE_VELOCITY_IN_OUTPUT,
            torque_component_amplitudes_nm=TORQUE_COMPONENT_AMPLITUDES_NM,
            torque_component_frequencies_hz=TORQUE_COMPONENT_FREQUENCIES_HZ,
            torque_component_phases_deg=TORQUE_COMPONENT_PHASES_DEG,
            torque_dc_offset_nm=TORQUE_DC_OFFSET_NM,
            excitation_duration=EXCITATION_DURATION,
            ramp_time=RAMP_TIME,
        )
        print(f"Saved dataset to: {os.path.abspath(save_path)}")

    metadata = {
        "script": "analysisV3_torque.py",
        "output_dir": output_dir,
        "xml_file_path": XML_FILE_PATH,
        "settings": {
            "excitation_duration": float(EXCITATION_DURATION),
            "ramp_time": float(RAMP_TIME),
            "use_gravity_compensation": bool(USE_GRAVITY_COMPENSATION),
            "gpe_input_choice": GPE_INPUT_CHOICE,
            "use_velocity_in_output": bool(USE_VELOCITY_IN_OUTPUT),
            "tau_limits": TAU_LIMITS.tolist(),
            "run_gpe_check": bool(RUN_GPE_CHECK),
            "gpe_hankel_depth": int(GPE_HANKEL_DEPTH),
            "save_data": bool(SAVE_DATA),
            "torque_component_amplitudes_nm": TORQUE_COMPONENT_AMPLITUDES_NM.tolist(),
            "torque_component_frequencies_hz": TORQUE_COMPONENT_FREQUENCIES_HZ.tolist(),
            "torque_component_phases_deg": TORQUE_COMPONENT_PHASES_DEG.tolist(),
            "torque_dc_offset_nm": TORQUE_DC_OFFSET_NM.tolist(),
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

    save_figure(plot_joint_positions(time_hist, q_hist), output_dir, "joint_positions.png")
    save_figure(plot_joint_velocities(time_hist, qd_hist), output_dir, "joint_velocities.png")
    save_figure(
        plot_torque_breakdown(
            time_hist,
            tau_exc_cmd_hist,
            tau_exc_applied_hist,
            tau_base_hist,
            tau_total_hist,
        ),
        output_dir,
        "torque_breakdown.png",
    )
    save_figure(plot_reference_spectrum_info(), output_dir, "reference_spectrum.png")


if __name__ == "__main__":
    main()
