import numpy as np
import mujoco
import cvxpy as cp

from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController

class QPVelocityPlanner:
    """
    Computes joint torques using a Quadratic Program to track an end-effector position target
    and a derivative-only JointVelocityController to convert joint velocities into torques.

    Reference: https://debojit.notion.site/DLS-IK-1eef4aa874c48068b8b6efcda8bec94c

    Based on the optimization formulation of Damped Least Squares IK.
    """

    def __init__(self, model, data, site_name="right_center", damping=1e-4):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.damping = damping

        # Internal D controller
        self.jvc = JointVelocityController(model, data, kd=4.0)

    def compute_qp_ik(self, target_pos, ori_gain=1.0, pos_gain=12.0):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        ee_pos = self.data.site_xpos[site_id]
        ee_mat = self.data.site_xmat[site_id].reshape(3, 3)

        pos_err = target_pos - ee_pos
        current_z = ee_mat[:, 2]
        target_z = np.array([0, 0, -1])
        ori_err = np.cross(current_z, target_z)

        # Compute Jacobians
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        # Stack Jacobians and error vectors
        J = np.vstack([pos_gain * jacp, ori_gain * jacr])    # J is 6×n Jacobian (position + orientation)
        err = np.concatenate([pos_gain * pos_err, ori_gain * ori_err])  # ẋ_d is the desired spatial velocity

        # ----------------------------------------------
        # Cost function used: Equation (23)
        # J(q̇) = || J q̇ - ẋ_d ||² + λ || q̇ ||²
        #
        # Expanded form: Equation (26)
        # J(q̇) = q̇ᵀ (JᵀJ + λI) q̇ - 2 ẋ_dᵀ J q̇ + ẋ_dᵀ ẋ_d
        #
        # QP Form: Equation (28)
        # J(Δq) = (1/2) Δqᵀ (2/Δt²)(JᵀJ + λI) Δq + (-1/Δt ẋ_dᵀ J) Δq + constant
        #
        # Simplified QP: Equation (31)
        # min Δq ½ Δqᵀ H Δq + cᵀ Δq
        # (we drop the constant term ẋ_dᵀ ẋ_d)
        # ----------------------------------------------

        dq_var = cp.Variable(self.model.nv)

        # Direct implementation of Equation (23)
        cost = cp.sum_squares(J @ dq_var - err) + self.damping * cp.sum_squares(dq_var)

        # Equivalent to QP in Equation (31): min Δq ½ Δqᵀ H Δq + cᵀ Δq
        prob = cp.Problem(cp.Minimize(0.5 * cost))
        prob.solve(solver=cp.OSQP)

        if dq_var.value is None:
            raise RuntimeError("QP failed to solve.")

        return dq_var.value[:self.model.nu]  # Return actuator-space joint velocities

    def get_torque_command(self, target_pos):
        dq = self.compute_qp_ik(target_pos)
        self.jvc.set_velocity_target(dq)
        self.jvc.control_callback(self.model, self.data)
        return self.data.ctrl.copy()
