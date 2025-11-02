import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


class ChoiProjection(eqx.Module):
    """
    Implementation of the Algorithm 1: Projection onto CPTP from [1].

    Projects a Choi matrix to the closest CPTP Choi matrix.

    Conventions:
        - The algorithm expects a normalized Choi matrix in following row ordering
        vectorization, i.e.: J = 1/d (\lambda \otimes Id_d) (\Omega).
        Where \Omega is the maximally entangled state. Notice that it is divided
        by d.
        - The normalization condition results in Tr_1(J) = Id_d / d, where we partial
        trace the first subsystem.

    [1] J. Barberà-Rodríguez, L. Zambrano, A. Acín, and D. Farina,
    Boosting projective methods for quantum process and detector tomography,
    Phys. Rev. Res. 7, 013208 (2025).
    """

    system_dimension: int
    identity_dimension: jnp.array

    def __init__(self, system_dimension: int):
        self.system_dimension = system_dimension
        self.identity_dimension = jnp.eye(system_dimension)

    # @jax.jit
    @eqx.filter_jit
    def tp_proj(self, J: Complex[Array, "d^2 d^2"]) -> Complex[Array, "d^2 d^2"]:
        d = self.identity_dimension.shape[0]
        identity = jnp.eye(d)

        J_reshape = J.reshape(d, d, d, d)
        J_A = jnp.einsum("ijik->jk", J_reshape)

        correction = jnp.kron(identity, (1 / d * identity - J_A))

        J_tp = J + (1 / d) * correction
        return J_tp

    # @jax.jit
    @eqx.filter_jit
    def cp1_proj(self, mu: Complex[Array, "d^2 d^2"]) -> Complex[Array, "d^2 d^2"]:
        """
        Differentiable reimplementation of the original while-loop using lax.scan.
        The scan runs exactly d steps (static-length control flow). We simulate
        the early stop by carrying a `done` flag and freezing updates with jnp.where.

        Returns the projected density matrix `rho`.
        """
        d = mu.shape[0]
        mu = jnp.asarray(mu)

        # Eigendecompose and sort descending
        evals, evecs = jnp.linalg.eigh(mu)
        # order = jnp.argsort(evals)[::-1]
        # evals = evals[order]
        # evecs = evecs[:, order]
        evals = evals[::-1]
        evecs = evecs[:, ::-1]  # eigh already gives evals in ascending order

        # scan over indices from d-1 down to 0
        idxs = jnp.arange(d - 1, -1, -1, dtype=jnp.int32)

        def scan_step(carry, idx):
            a, done, i = (
                carry  # a: accumulated sum, done: whether we already stopped, i: current i value
            )

            # compute condition only if not done
            cond = evals[idx] + a / (idx + 1) < 0
            do_update = (~done) & cond

            a_new = jnp.where(do_update, a + evals[idx], a)

            # If we perform the body (do_update) then i becomes idx-1.
            # If we stop here (~done & ~cond) then i should stay at idx (body not executed).
            i_new = jnp.where(do_update, idx - 1, jnp.where((~done) & (~cond), idx, i))

            done_new = done | ((~done) & (~cond))

            return (a_new, done_new, i_new), None

        init_carry = (
            jnp.array(0.0, dtype=evals.dtype),
            jnp.array(False),
            jnp.array(d - 1, dtype=jnp.int32),
        )
        (a_final, done_final, i_final), _ = jax.lax.scan(scan_step, init_carry, idxs)

        # Build final lambda array following original logic
        shift = jnp.where(i_final >= 0, a_final / (i_final + 1), 0.0)
        mask = jnp.arange(d) <= i_final
        lam = jnp.where(mask, evals + shift, jnp.zeros_like(evals))

        # Reconstruct rho = sum_j lam_j |v_j><v_j|
        rho = (evecs * lam) @ evecs.conj().T
        return rho

    def _sym(self, A):
        return 0.5 * (A + jnp.conj(A).T)

    # @partial(jax.jit)
    @eqx.filter_jit
    def hermitian_inv_sqrt(self, A, eps_eig=1e-12):
        # A = self._sym(A)  # <- maybe we don't need this?

        w, V = jnp.linalg.eigh(A)
        w = jnp.real(w)

        eps_eig = jnp.asarray(eps_eig, dtype=w.dtype)

        # PSEUDO
        # inv_sqrt_vals = jnp.where(w > eps_eig, 1.0 / jnp.sqrt(w), 0.0)

        # NO PSEUDO
        w_clamped = jnp.clip(w, a_min=eps_eig)
        inv_sqrt_vals = 1.0 / jnp.sqrt(w_clamped)

        # # NO PSEUDO ANOTHER GPT VERSION #<- what it's worked the best for now
        # delta = jnp.maximum(10 * eps_eig, jnp.max(w) * 1e-8)
        # mask = jax.nn.sigmoid((w - eps_eig) / delta)
        # inv_sqrt_vals = mask / jnp.sqrt(w + eps_eig)

        return (V * inv_sqrt_vals) @ jnp.conj(V).T

    @eqx.filter_jit
    def dykstraCBA(
        self, X: Complex[Array, "d^2 d^2"], max_iter: int, tol: Float
    ) -> Complex[Array, "d^2 d^2"]:
        """
        Dykstra loop rewritten to use lax.scan with a static number of iterations
        (`max_iter`). We emulate early stopping with a `done` flag and jnp.where
        freezing so that autodiff (reverse-mode) works.
        """
        p = jnp.zeros_like(X)
        q = jnp.zeros_like(X)

        init_done = jnp.array(False)
        init_k = jnp.array(0, dtype=jnp.int32)
        init_eps = jnp.array(100.0, dtype=jnp.float64)

        init_carry = (
            X,
            p,
            q,
            init_k,
            init_eps,
        )

        def apply_step(carry):
            X, p, q, k, eps = carry

            Y = self.tp_proj(X + p)
            p_new = X + p - Y
            X_new = self.cp1_proj(Y + q)
            q_new = Y + q - X_new
            k_new = k + 1
            new_eps = jnp.linalg.norm(p_new - p) ** 2 + jnp.linalg.norm(q_new - q) ** 2

            return (
                X_new,
                p_new,
                q_new,
                k_new,
                new_eps,
            )

        def skip_step(carry):
            return carry

        def scan_step(carry, _):
            X, p, q, k, eps = carry

            new_carry = jax.lax.cond(eps >= tol, apply_step, skip_step, carry)

            return new_carry, None

        final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=max_iter)
        X_final = final_carry[0]

        d = self.identity_dimension.shape[0]
        id_d = jnp.eye(d)

        # trace out subsystem A (same convention as before)
        X_ambient = jnp.einsum("ijik->jk", X_final.reshape(d, d, d, d))
        U = jnp.kron(id_d, self.hermitian_inv_sqrt(d * X_ambient))
        final_choi = U @ X_final @ U.conj().T

        return final_choi
