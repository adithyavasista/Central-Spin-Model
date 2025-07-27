import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
import qutip as qt
import csv
import gc
import matplotlib.pyplot as plt

# Pauli Matrices
def pauli_matrices():
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    return sigma_x, sigma_y, sigma_z

def create_three_spin_ising_hamiltonian(h, J3, J, N):
    sigma_x, _, sigma_z = pauli_matrices()
    H = csr_matrix((2**N, 2**N), dtype=complex)

    # Transverse field term: -h/2 * sum σ_z^i
    for i in range(N):
        ops = [eye(2, format='csr')] * N
        ops[i] = sigma_z
        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')
        H += -0.5 * h * term

    # Nearest-neighbor term: -J/2 * sum σ_x^i σ_x^{i+1}
    for i in range(N):
        ops = [eye(2, format='csr')] * N
        ops[i] = sigma_x
        ops[(i + 1) % N] = sigma_x  # PBC
        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')
        H += -0.5 * J * term

    # Three-spin interaction: -J3/2 * sum σ_z^i σ_x^{i+1} σ_x^{i+2}
    for i in range(N):
        ops = [eye(2, format='csr')] * N
        ops[i] = sigma_z
        ops[(i - 1) % N] = sigma_x
        ops[(i + 1) % N] = sigma_x
        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')
        H += -0.5 * J3 * term

    return H
def calculate_concurrence(state_dict):
    """Calculate concurrence from four evolved states"""
    # Extract states
    psi_dd = state_dict['downdown'].full().flatten()
    psi_uu = state_dict['upup'].full().flatten()
    psi_du = state_dict['downup'].full().flatten()
    psi_ud = state_dict['updown'].full().flatten()
    
    # Calculate all overlaps
    a = np.abs(np.vdot(psi_uu, psi_ud))
    b = np.abs(np.vdot(psi_uu, psi_du))
    c = np.abs(np.vdot(psi_uu, psi_dd))
    d = np.abs(np.vdot(psi_ud, psi_du))
    e = np.abs(np.vdot(psi_ud, psi_dd))
    f = np.abs(np.vdot(psi_du, psi_dd))
    
    # Construct density matrix
    rho = 0.25 * np.array([
        [1, a, b, c],
        [np.conj(a), 1, d, e],
        [np.conj(b), np.conj(d), 1, f],
        [np.conj(c), np.conj(e), np.conj(f), 1]
    ], dtype=complex)
    
    # Compute concurrence
    Y = np.array([[0, 0, 0, -1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [-1, 0, 0, 0]])
    
    R = rho @ Y @ rho.conj() @ Y
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(np.real(eigvals))  # Ensure real, sorted eigenvalues
    
    conc = max(0, np.sqrt(eigvals[-1]) - sum(np.sqrt(np.maximum(eigvals[:-1], 0))))
    return conc

# Parameters
N = 20
J = 1.0
delta = 0.1
J3 = 1
total_time = 800
chunk_size = 200
time_points = 110

# Initialize Hamiltonians
print("Building Hamiltonians...")
hi = 0.2
hf = 0.7
H_base = create_three_spin_ising_hamiltonian(hi, J3, J, N)
H_final = create_three_spin_ising_hamiltonian(hf, J3, J, N)

# Local operators
sigma_z = pauli_matrices()[2]
sigma_p = kron(kron(eye(2**1, format="csr"), sigma_z), eye(2**(N-2), format="csr"))
sigma_q = kron(kron(eye(2**1, format="csr"), sigma_z), eye(2**(N-2), format="csr"))

# Perturbed Hamiltonians
H_downdown = H_final
H_upup = H_final - delta * (sigma_p + sigma_q)
H_downup = H_final - delta * sigma_q
H_updown = H_final - delta * sigma_p

# Get initial ground state
print("Calculating initial ground state...")
eigenvalues, eigenvectors = eigsh(H_base, k=1, which='SA')
psi0 = qt.Qobj(eigenvectors[:, 0])

# Initialize states
current_states = {
    'downdown': psi0.copy(),
    'upup': psi0.copy(),
    'downup': psi0.copy(),
    'updown': psi0.copy()
}

# Results storage
all_times = []
all_concurrence = []

# Time evolution in chunks
for chunk in range(total_time // chunk_size):
    t_start = chunk * chunk_size
    t_end = (chunk + 1) * chunk_size
    times = np.linspace(t_start, t_end, time_points)
    
    print(f"Processing time chunk {chunk+1}: t = {t_start} to {t_end}")
    
    # Evolve each state separately
    results = {
        'downdown': qt.sesolve(qt.Qobj(H_downdown), current_states['downdown'], times),
        'upup': qt.sesolve(qt.Qobj(H_upup), current_states['upup'], times),
        'downup': qt.sesolve(qt.Qobj(H_downup), current_states['downup'], times),
        'updown': qt.sesolve(qt.Qobj(H_updown), current_states['updown'], times)
    }
    
    # Calculate concurrence at each time point
    for i, t in enumerate(times):
        state_dict = {key: results[key].states[i] for key in results}
        conc = calculate_concurrence(state_dict)
        all_times.append(t)
        all_concurrence.append(conc)
    
    # Update states for next chunk
    current_states = {key: results[key].states[-1] for key in results}
    
    # Clean up
    del results
    gc.collect()

# Save results
with open(f'concurrence_J3_{J3}_N{N}_final_check.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time', 'Concurrence'])
    writer.writerows(zip(all_times, all_concurrence))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(all_times, all_concurrence, 'r-', linewidth=1.5)
plt.title(f'Concurrence Dynamics (N={N}, J3={J3}, Δ={delta})', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Concurrence', fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f'concurrence_J3_{J3}_N{N}_final_check.png', dpi=300)
plt.close()

print("Calculation completed successfully!")