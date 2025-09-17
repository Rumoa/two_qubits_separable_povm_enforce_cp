from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

# from jax import Array
from jaxtyping import Array, Complex, Float


class OldWhileLoopChoiProjection(eqx.Module):
    system_dimension: int
    identity_dimension: jnp.array

    def __init__(self, system_dimension: int):
        """We are going to follow the convention in the paper:
        'Boosting projective methods for quantum process and detector tomography', where
        they work in the row ordering vectorization, meaning that the choi matrix
        is defined as (map tensor I) 1/d (Omega) where Omega is the unnormalized
        maximally entangled state.

        Args:
            system_dimension (int): the dimension of the Hilbert space we are working with.
        """
        self.system_dimension = system_dimension
        self.identity_dimension = jnp.eye(system_dimension)

    @jax.jit
    def tp_proj(self, J):
        """Projects a choi matrix folling the convention (map tensor id) to the TP space.
        The convention matters as we need to trace out the subsystem affected by the map, which
        determines the contraction.

        Args:
            J (_type_): choi matrix with the convention (map tensor id), possibly non TP

        Returns:
        J_tp : choi matrix corresponding to TP map, such that Tr_B (J) = id/d
        """
        d = self.identity_dimension.shape[0]
        identity = jnp.eye(d)

        J_reshape = J.reshape(d, d, d, d)
        J_A = jnp.einsum("ijik->jk", J_reshape)  # <- we trace out the system A

        # correction = jnp.kron(identity_factor, (1 / d * identity_factor - J_A))
        correction = jnp.kron(
            identity,
            (1 / d * identity - J_A),
        )

        J_tp = J + (1 / d) * correction
        return J_tp

    @jax.jit
    def cp1_proj(self, mu):
        """
        Implements your steps using jax.lax.while_loop.
        - mu: Hermitian matrix (d x d)
        Returns: rho (projected matrix), lam (eigenvalues after projection), evecs
        """

        d = mu.shape[0]
        mu = jnp.asarray(mu)
        # Eigendecompose and sort descending
        evals, evecs = jnp.linalg.eigh(mu)
        order = jnp.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        # d = evals.shape[0]
        # initial loop state: i = d-1 (last index), a = 0.0
        i0 = jnp.array(d - 1, dtype=jnp.int32)
        a0 = jnp.array(0.0, dtype=evals.dtype)
        init_state = (i0, a0)

        # def cond_fun(state):
        #     i, a = state
        #     # stop looping if i < 0; otherwise continue while evals[i] + a/(i+1) < 0
        #     # use safe boolean expression
        #     return (i >= 0) & (evals[i] + a / (i + 1) < 0)

        def cond_fun(state):
            i, a = state

            def true_branch(_):
                # safe: executed only when i >= 0
                return evals[i] + a / (i + 1) < 0

            def false_branch(_):
                # when i < 0, we must return False (stop)
                return jnp.array(False)

            return jax.lax.cond(i >= 0, true_branch, false_branch, operand=None)

        def body_fun(state):
            i, a = state
            a = a + evals[i]  # add mu_i to accumulator
            i = i - 1
            return (i, a)

        i_final, a_final = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # Build final lambda array:
        # if i_final == -1 => all lambdas are zero
        shift = jnp.where(i_final >= 0, a_final / (i_final + 1), 0.0)
        mask = jnp.arange(d) <= i_final
        lam = jnp.where(mask, evals + shift, jnp.zeros_like(evals))

        # Reconstruct rho = sum_j lam_j |v_j><v_j|
        rho = (evecs * lam) @ evecs.conj().T

        # return rho, lam, evecs
        return rho

    def _sym(self, A):
        return 0.5 * (A + jnp.conj(A).T)

    @partial(jax.jit, static_argnames="pseudo")
    def hermitian_inv_sqrt(self, A, eps_eig=1e-12, pseudo=False):
        """
        Return a stable inverse-square-root of a Hermitian matrix A.
        - If pseudo=False (default): eigenvalues < eps_eig get clamped to eps_eig (Tikhonov-like regularization).
        - If pseudo=True: treat small eigenvalues as zero (pseudo-inverse: 1/sqrt(lam) if lam>eps, else 0).
        """
        A = self._sym(A)
        w, V = jnp.linalg.eigh(A)  # w real, V columns are eigenvectors
        # allow small negative due to numerics
        w = jnp.real(w)

        eps_eig = jnp.asarray(eps_eig, dtype=w.dtype)
        if pseudo:
            inv_sqrt_vals = jnp.where(w > eps_eig, 1.0 / jnp.sqrt(w), 0.0)
        else:
            w_clamped = jnp.clip(w, a_min=eps_eig)
            inv_sqrt_vals = 1.0 / jnp.sqrt(w_clamped)

        # reconstruct inv(sqrt(A)) = V diag(inv_sqrt_vals) V^H
        return (V * inv_sqrt_vals) @ jnp.conj(V).T

    @partial(jax.jit, static_argnames=["max_iter", "tol"])
    def dykstraCBA(self, X, max_iter, tol) -> Array:
        p = jnp.zeros_like(X)
        q = jnp.zeros_like(X)

        k = jnp.array(0, dtype=jnp.int32)
        eps = jnp.array(100, dtype=jnp.float64)

        value = (X, p, q, k, eps)

        def condition_f(value):
            X, p, q, k, eps = value
            return (eps > tol) & (k <= max_iter)

        def while_f(value):
            X, p, q, k, eps = value
            Y = self.tp_proj(X + p)
            p_new = X + p - Y
            X = self.cp1_proj(Y + q)
            q_new = Y + q - X
            k = k + 1
            new_eps = jnp.linalg.norm(p_new - p) ** 2 + jnp.linalg.norm(q_new - q) ** 2

            new_value = X, p_new, q_new, k, new_eps
            return new_value

        final_value = jax.lax.while_loop(condition_f, while_f, init_val=value)
        X, _, _, _, _ = final_value

        d = self.identity_dimension.shape[0]
        id_d = jnp.eye(d)

        # We now trace the subsystem A since it is the one that is evolved by the map
        # in this convention (map tensor id)
        X_ambient = jnp.einsum("ijik->jk", X.reshape(d, d, d, d))

        # Now we need to compute the inverse of the sqrt of d*X_ambient.
        # the function jax.scipy.linalg.sqrtm is not implemented for gpus, only for cpu
        # The good thing is that X_ambient is the reduced state of the X, which is
        # supposed to be the choi state of a cptp map (bc we are done with projecting)
        # so it is a valid state. When we trace out one of the systems, the reduced state
        # is hermitian.
        # We can compute the inverse of the sqrt of an hermitian matrix in gpus
        # by writing a custom function that diagonalizes the function, computes the sqrt
        # of the evals and then inverts them.

        U = jnp.kron(id_d, self.hermitian_inv_sqrt(d * X_ambient))

        final_choi = U @ X @ U.conj().T

        return final_choi


class ChoiProjection(eqx.Module):
    system_dimension: int
    identity_dimension: jnp.array

    def __init__(self, system_dimension: int):
        self.system_dimension = system_dimension
        self.identity_dimension = jnp.eye(system_dimension)

    @jax.jit
    def tp_proj(self, J: Complex[Array, "d^2 d^2"]) -> Complex[Array, "d^2 d^2"]:
        d = self.identity_dimension.shape[0]
        identity = jnp.eye(d)

        J_reshape = J.reshape(d, d, d, d)
        J_A = jnp.einsum("ijik->jk", J_reshape)

        correction = jnp.kron(identity, (1 / d * identity - J_A))

        J_tp = J + (1 / d) * correction
        return J_tp

    @jax.jit
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
        order = jnp.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

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

    @partial(jax.jit, static_argnames="pseudo")
    def hermitian_inv_sqrt(self, A, eps_eig=1e-12, pseudo=False):
        A = self._sym(A)
        w, V = jnp.linalg.eigh(A)
        w = jnp.real(w)

        eps_eig = jnp.asarray(eps_eig, dtype=w.dtype)
        if pseudo:
            inv_sqrt_vals = jnp.where(w > eps_eig, 1.0 / jnp.sqrt(w), 0.0)
        else:
            w_clamped = jnp.clip(w, a_min=eps_eig)
            inv_sqrt_vals = 1.0 / jnp.sqrt(w_clamped)

        return (V * inv_sqrt_vals) @ jnp.conj(V).T

    @partial(jax.jit, static_argnames=["max_iter", "tol"])
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

        # carry: X, p, q, k, eps, done
        init_carry = (
            X,
            p,
            q,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(100.0, dtype=jnp.float64),
            jnp.array(False),
        )

        def scan_step(carry, _):
            X, p, q, k, eps, done = carry

            Y = self.tp_proj(X + p)
            p_new = X + p - Y
            X_new = self.cp1_proj(Y + q)
            q_new = Y + q - X_new

            k_new = k + 1
            new_eps = jnp.linalg.norm(p_new - p) ** 2 + jnp.linalg.norm(q_new - q) ** 2
            conv = new_eps <= tol

            # freeze updates if already done
            X_out = jnp.where(done, X, X_new)
            p_out = jnp.where(done, p, p_new)
            q_out = jnp.where(done, q, q_new)
            eps_out = jnp.where(done, eps, new_eps)
            done_out = done | conv

            return (X_out, p_out, q_out, k_new, eps_out, done_out), None

        final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=max_iter)
        X_final, _, _, _, _, _ = final_carry

        d = self.identity_dimension.shape[0]
        id_d = jnp.eye(d)

        # trace out subsystem A (same convention as before)
        X_ambient = jnp.einsum("ijik->jk", X_final.reshape(d, d, d, d))

        U = jnp.kron(id_d, self.hermitian_inv_sqrt(d * X_ambient))
        final_choi = U @ X_final @ U.conj().T

        return final_choi
