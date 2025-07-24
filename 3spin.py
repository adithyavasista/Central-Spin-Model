# import numpy as np
# import scipy.linalg as la
# from scipy.sparse import kron, eye, csr_matrix
# from scipy.sparse.linalg import expm
# from scipy.sparse.linalg import expm_multiply
# from scipy.sparse.linalg import eigsh
# from scipy.linalg import sqrtm
# import matplotlib.pyplot as plt
# import qutip as qt
# import csv

# # Pauli Matrices
# def pauli_matrices():
#     sigma_x = np.array([[0, 1], [1, 0]])
#     sigma_y = np.array([[0, -1j], [1j, 0]])
#     sigma_z = np.array([[1, 0], [0, -1]])
#     return sigma_x, sigma_y, sigma_z

# # Function to create Ising Hamiltonian
# def create_three_spin_ising_hamiltonian(h, J3, J, N):
#     sigma_x,_,sigma_z = pauli_matrices()
#     H = csr_matrix((2**N, 2**N), dtype=complex)



#     # Transverse field term
#     for i in range(N):
#         Hz = -0.5*h* kron(kron(eye(2**i,format="csr"), sigma_z), eye(2**(N-i-1),format="csr"),format="csr")
#         H += Hz

#     # Interaction term
#     for i in range(N-1):
#         Hx = -0.5*J * kron(kron(kron(eye(2**i,format="csr"), sigma_x), sigma_x),eye(2**(N-i-2),format="csr"),format="csr")
#         H += Hx

#     for i in range(N):
#         term_zxx = csr_matrix(eye(1))  # Start with 1x1 identity
#         for j in range(N):
#             if j == i:
#                 term_zxx = kron(term_zxx, sigma_z, format='csr')
#             elif j == (i - 1) % N or j == (i + 1) % N:
#                 term_zxx = kron(term_zxx, sigma_x, format='csr')
#             else:
#                 term_zxx = kron(term_zxx, eye(2, format='csr'), format='csr')
#         H -= 0.5 * J3 * term_zxx
#     H_periodic = -0.5*J * kron(sigma_x,kron(eye(2**(N-2),format="csr"), sigma_x),format="csr")
#     H += H_periodic


#     return H
# # Define constants and initial states
# N = 15
# J = 1.0

# delta = 0.1
# p, q = 1, 1
# sum_conc=0
# # for ra in range(1, 11):
# #     # Initial states
# #     delta = ra / 10

# J3_values=np.array([0,0.1,0.2,0.3,1.5])
# average_values = []
# J3_val=[]
# h_range=np.array([0.99])
# conc_avg=1
# h_val=[]

# for J3 in J3_values:
    
#     concurrence_values = []
#     hi =0.2
#     hf=0.99
    
#     # Define the Hamiltonians
#     H_base = create_three_spin_ising_hamiltonian(hi,J3,J, N)
#     H_final = create_three_spin_ising_hamiltonian(hf,J3,J, N)
#     sigma_z = pauli_matrices()[2]
#     sigma_p = kron(kron(eye(2**p,format="csr"), sigma_z),eye(2**(N-p-1),format="csr"))
#     sigma_q = kron(kron(eye(2**q,format="csr"), sigma_z),eye(2**(N-q-1),format="csr"))

#     # Different Hamiltonians for each initial state
#     H_downdown = H_final
#     H_upup = H_final - delta * (sigma_p + sigma_q)
#     H_downup = H_final - delta * sigma_q
#     H_updown = H_final - delta * sigma_p

#     # Define initial state vectors

#     eigenvalues, eigenvectors = eigsh(H_base)
#     state_vector_1 = qt.Qobj(eigenvectors[:, 0])  # Convert to Qutip Qobj

#     # Define time values
#     time_values = np.linspace(0,220, 140)

#     # Solve Schrödinger equation for all time values at once
#     sol_1 = qt.sesolve(qt.Qobj(H_downdown), state_vector_1, time_values)
#     sol_2 = qt.sesolve(qt.Qobj(H_upup), state_vector_1, time_values)
#     sol_3 = qt.sesolve(qt.Qobj(H_downup), state_vector_1, time_values)
#     sol_4 = qt.sesolve(qt.Qobj(H_updown), state_vector_1, time_values)

#     # Extract all evolved states
#     U_1 = sol_1.states
#     U_2 = sol_2.states
#     U_3 = sol_3.states
#     U_4 = sol_4.states

#     lambda_values = []
#     concurrence_values = []
#     # Iterate over time indices
#     for idx, t in enumerate(time_values):
#         new_1 = U_1[idx].full().flatten()
#         new_2 = U_2[idx].full().flatten()
#         new_3 = U_3[idx].full().flatten()
#         new_4 = U_4[idx].full().flatten()

#         # Compute overlaps
#         a = np.abs(np.vdot(new_2, new_4))
#         b = np.abs(np.vdot(new_2, new_3))
#         c = np.abs(np.vdot(new_2, new_1))
#         d = np.abs(np.vdot(new_4, new_3))
#         e = np.abs(np.vdot(new_4, new_1))
#         f = np.abs(np.vdot(new_3, new_1))

#         # Build reduced density matrix
#         rho_reduced = np.array([
#             [1, a, b, c],
#             [np.conj(a), 1, d, e],
#             [np.conj(b), np.conj(d), 1, f],
#             [np.conj(c), np.conj(e), np.conj(f), 1]
#         ]) * 0.25

#         # Calculate concurrence
#         sigma_y = pauli_matrices()[1]
#         sigy = np.zeros((4, 4), dtype=float)
#         sigy[0, 3] = -1.0
#         sigy[1, 2] = 1.0
#         sigy[2, 1] = 1.0
#         sigy[3, 0] = -1.0
#         # Calculate conjugate and tilde matrix
        
#         mcnc1 = np.dot(sigy, rho_reduced)
#         mcnc = np.dot(mcnc1, sigy)
#         mcncc = np.dot(rho_reduced, mcnc)

#         # Eigenvalues
#         ci = np.linalg.eigvalsh(mcncc)
#         ci[ci < 0.0] = 0.0

#         # Compute concurrence (prefactor 1/16 for unentangled states)

#         conc = (np.sqrt(ci[3]) - np.sqrt(ci[2]) - np.sqrt(ci[1]) - np.sqrt(ci[0]))
#         conc = max(0, conc)



#         lambda_values.append(t)
#         concurrence_values.append(conc)
# filename = f"concurrence_dynamic_j3_{J3:.1f}.csv"
# with open(filename, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["t", "Concurrence"])
#     writer.writerows(zip(lambda_values, concurrence_values))

#     plt.figure(figsize=(8, 6))
#     plt.plot(lambda_values, concurrence_values, marker="o", linestyle="-", color="r", label="Concurrence",markersize=4)
#     plt.title("Concurrence vs time N=15 varying h", fontsize=14)
#     plt.xlabel("Time (t)", fontsize=12)
#     plt.ylabel("Concurrence (C) ", fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.7)
#     plt.legend()
#     plt.show()
    



#     # # Save to file, ensuring only the real part is written
#     # file.write(f"{time},{C.real}\n")
