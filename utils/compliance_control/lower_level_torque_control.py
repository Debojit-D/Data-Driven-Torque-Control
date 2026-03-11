import numpy as np
from typing import Optional

class LowerLevelTorqueControl:
    """Light‑weight operational‑space torque controller.

    Provide two public methods:
        • impedance_control(...): classic Cartesian impedance (no force feedback)
        • admittance_control(...): Cartesian admittance (adds measured wrench)

    Internally the torque command is decomposed into clearly named parts
    that can also be accessed individually if desired:
        1. qfrc_bias()        – gravity + Coriolis from MuJoCo
        2. tau_spring_damper()– virtual spring + damper wrench mapped to joints
        3. tau_external()     – external wrench mapped to joints (admittance)
        4. null_space_term()  – energy‑damping null‑space motion
    """

    def __init__(self, kp: float, kd: float, null_kv: float = 0.0,
                 damping_lambda: float = 1e-3):
        """Parameters
        ----------
        kp, kd       Spring & damper gains applied identically to all 6 task DoF.
        null_kv      Viscous gain for null‑space damping (0 ⇒ disabled).
        damping_lambda  SVD damping used for task‑space inertia inversion.
        """
        self.kp = kp
        self.kd = kd
        self.null_kv = null_kv
        self.damping_lambda = damping_lambda

    # ------------------------------------------------------------------
    #  building blocks (private helpers)
    # ------------------------------------------------------------------

    def _task_inertia(self, J: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
        """Return Λ = (J M^{-1} Jᵀ)^{-1} (well‑conditioned)."""
        Lambda_inv = J @ M_inv @ J.T
        # regularised inverse via SVD to avoid singularities
        U, S, Vt = np.linalg.svd(Lambda_inv)
        S_inv = np.diag(1.0 / (S + self.damping_lambda))
        return Vt.T @ S_inv @ U.T  # Λ

    # ------------------------------------------------------------------
    #  Public building‑block functions
    # ------------------------------------------------------------------

    def qfrc_bias(self, sim) -> np.ndarray:
        """Gravity + Coriolis term directly from MuJoCo."""
        return sim.data.qfrc_bias.copy()

    def tau_spring_damper(self,
                          J: np.ndarray,
                          M_inv: np.ndarray,
                          x: np.ndarray,
                          x_d: np.ndarray,
                          x_dot: np.ndarray,
                          x_dot_d: np.ndarray) -> np.ndarray:
        """Joint torque produced by virtual spring‑damper in task space."""
        e = x - x_d
        e_dot = x_dot - x_dot_d
        F_cmd = self.kp * e + self.kd * e_dot  # 6m‑vector
        Lambda = self._task_inertia(J, M_inv)
        return -J.T @ (Lambda @ F_cmd)

    def tau_external(self, J: np.ndarray, M_inv: np.ndarray, F_ext: np.ndarray) -> np.ndarray:
        """Map measured external wrench to joint space (for admittance)."""
        Lambda = self._task_inertia(J, M_inv)
        return -J.T @ (Lambda @ F_ext)

    def null_space_term(self,
                        M: np.ndarray,
                        J: np.ndarray,
                        M_inv: np.ndarray,
                        dq: np.ndarray) -> np.ndarray:
        """Simple kinetic‑energy damping in the null space of J."""
        if self.null_kv <= 0:
            return np.zeros_like(dq)
        Lambda = self._task_inertia(J, M_inv)
        J_bar = M_inv @ J.T @ Lambda                    # dynamically‑consistent pseudo‑inv
        N = np.eye(M.shape[0]) - J.T @ J_bar            # null‑space projector
        return N @ (-self.null_kv * M @ dq)             # viscous term

    # ------------------------------------------------------------------
    #  High‑level control laws
    # ------------------------------------------------------------------

    def impedance_control(self,
                          sim,
                          J: np.ndarray,
                          M: np.ndarray,
                          x: np.ndarray,
                          x_d: np.ndarray,
                          x_dot: np.ndarray,
                          x_dot_d: np.ndarray,
                          dq: np.ndarray) -> np.ndarray:
        """Classic impedance (no force feedback)."""
        M_inv = np.linalg.inv(M)
        tau = self.qfrc_bias(sim)
        tau += self.tau_spring_damper(J, M_inv, x, x_d, x_dot, x_dot_d)
        tau += self.null_space_term(M, J, M_inv, dq)
        return tau

    def admittance_control(self,
                           sim,
                           J: np.ndarray,
                           M: np.ndarray,
                           x: np.ndarray,
                           x_d: np.ndarray,
                           x_dot: np.ndarray,
                           x_dot_d: np.ndarray,
                           dq: np.ndarray,
                           F_ext: np.ndarray) -> np.ndarray:
        """Admittance version (adds measured external wrench)."""
        M_inv = np.linalg.inv(M)
        tau = self.qfrc_bias(sim)
        tau += self.tau_spring_damper(J, M_inv, x, x_d, x_dot, x_dot_d)
        tau += self.tau_external(J, M_inv, F_ext)
        tau += self.null_space_term(M, J, M_inv, dq)
        return tau