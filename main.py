import numpy as np
from qutip import Qobj


class QSystem:
    def __init__(self, frequencies: list, couplings: list, basis_states: list, fill_hermitian_conjugate=True):
        """frequencies - list of qubits frequencies, list of lists like [qubit_number, frequency]
           couplings - interaction between qubits, list of lists like [index_1, index_2, coupling_coefficient]
           basis_states - Fock states, like [0, ..., 0, 1, 0, ....], [0, ..., 0, 1, 0, ..., 0, 1, 0, ...] etc.
           complete_hermitian_conjugate - automatically complete the given couplings [index_1, index_2, coupling_coefficient]
               by [index_2, index_1, coupling_coefficient*] (* - means complex conjugation)
               set to False to be able to fill the couplings array by hand
           """
        self.frequencies = frequencies
        self.number_qubits = len(frequencies)
        self.couplings = couplings
        self.basis_states = basis_states
        self.number_basis_states = len(basis_states)
        self.fill_hermitian_conjugate = fill_hermitian_conjugate

    def annihilation_operator_action(self, index, state):
        """Action of the boson annihilation operator:
           index - index of the system element on which the operator acts
           x - state in the Fock basis of states on which the operator acts"""
        if type(state) == list:
            state_copy = state.copy()
        else:
            state_copy = state
        if state_copy[index] == 0:
            return 0, 0
        else:
            state_copy[index] -= 1
            return state_copy, (state_copy[index] + 1) ** 0.5

    def creation_operator_action(self, index, state):
        """Action of the boson creation operator:
           index - index of the system element on which the operator acts
           state - state in the Fock basis of states on which the operator acts"""
        if type(state) == list:
            state_copy = state.copy()
        else:
            state_copy = state
        if state_copy == 0:
            return 0, 0
        state_copy[index] += 1
        return state_copy, state_copy[index] ** 0.5

    def H_matrix(self):
        """Returns matrix of the Hamiltonian of the system in the basis of Fock states
           with_hermitian_conjugate == True means self.couplings contains both hermitian conjugate terms like
           [index_1, index_2, coefficient] and [index_2, index_1, coefficient*] (* means complex conjugation)
            with_hermitian_conjugate == False means self.couplings contains only one term"""
        matrix_diag = [[0 for _ in range(self.number_basis_states)] for __ in
                       range(self.number_basis_states)]  # template for storing diagonal elements of the matrix
        matrix_cross = [[0 for _ in range(self.number_basis_states)] for __ in
                        range(self.number_basis_states)]  # template for storing off-diagonal elements of the matrix
        for i in range(self.number_basis_states):
            for j in range(self.number_basis_states):
                for elem in self.couplings:
                    # building matrix_cross
                    state, const = self.annihilation_operator_action(elem[1], self.basis_states[i])
                    state1, const1 = self.creation_operator_action(elem[0], state)
                    if state1 == self.basis_states[j]:
                        matrix_cross[i][j] += const * const1 * elem[2]
                for elem in self.frequencies:
                    # building matrix_diag
                    state, const = self.annihilation_operator_action(elem[0], self.basis_states[i])
                    state1, const1 = self.creation_operator_action(elem[0], state)
                    if state1 == self.basis_states[j]:
                        matrix_diag[i][j] += const * const1 * elem[1]
        matrix_diag = np.array(matrix_diag)
        matrix_cross = np.array(matrix_cross)
        if not self.fill_hermitian_conjugate:
            return Qobj(matrix_diag + matrix_cross)
        else:
            return Qobj(matrix_diag + matrix_cross + matrix_cross.conjugate().T)

    def annihilation_operator_matrix(self, qubit_index: int):
        """Returns matrix of the required operator. Useful to add extra terms in Hamiltonian or in other cases"""
        matrix = [[0 for _ in range(self.number_basis_states)] for __ in
                  range(self.number_basis_states)]  # template for storing matrices of annihilation operators
        for i in range(self.number_basis_states):
            for j in range(self.number_basis_states):
                state, const = self.annihilation_operator_action(qubit_index, self.basis_states[i])
                if state == self.basis_states[j]:
                    matrix[j][i] = const
        return Qobj(matrix)
