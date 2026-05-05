import numpy as np

def permute_dyad(A):
    return 0.5*(np.einsum("ik,jl->ijkl", A, A) + np.einsum("il,jk->ijkl", A, A))

def neo_hooke_2d_from_C(C, C10=0.5, kappa=500.0):
    I = np.eye(2)
    A = np.linalg.inv(C)
    I1 = np.trace(C)
    J = np.sqrt(np.linalg.det(C))

    # volumetric law (penalty)
    p  = kappa*(J - 1.0)
    pt = p + J*kappa  # = kappa*(2J-1)

    S_iso = 2*C10/J * (I - 0.5*I1*A)
    S_vol = J*p*A
    S = S_iso + S_vol

    AOA   = np.einsum("ij,kl->ijkl", A, A)
    AodA  = permute_dyad(A)
    IoA   = np.einsum("ij,kl->ijkl", I, A)
    AoI   = np.einsum("ij,kl->ijkl", A, I)

    # Note: C4_iso has 0.5*I1 attached to AOA, and I1 attached to AodA
    C4_iso = 2*C10/J*( I1*AodA - IoA - AoI + 0.5*I1*AOA )
    C4_vol = J*pt*AOA - 2*J*p*AodA
    
    C4 = C4_iso + C4_vol
    return S, C4

def analytical_components_check(C, C4_tensor, C10=0.5, kappa=500.0):
    """
    Implements the scalar formulas from the image and compares with the tensor output.
    """
    # 1. Setup Variables
    J = np.sqrt(np.linalg.det(C))
    I1 = np.trace(C)
    A = np.linalg.inv(C)
    
    # Components of A (inverse C)
    a = A[0,0]
    b = A[0,1] # Assumes symmetry A[1,0] == A[0,1]
    c = A[1,1]

    # Pressure terms from original code
    p  = kappa*(J - 1.0)
    pt = p + J*kappa 

    # 2. Define Alpha and Beta
    # Alpha corresponds to the coefficient of the permuted dyad (AodA)
    # Beta corresponds to the coefficient of the standard dyad (AOA)
    
    # From C4_iso: coeff of AodA is (2*C10/J * I1)
    # From C4_vol: coeff of AodA is (-2*J*p)
    alpha = (2*C10/J * I1) - (2*J*p)

    # From C4_iso: coeff of AOA is (2*C10/J * 0.5 * I1) -> (C10/J * I1)
    # From C4_vol: coeff of AOA is (J*pt)
    beta = (C10/J * I1) + (J*pt)

    # 3. Implement Formulas from the Image
    # Linear term coefficient used in image: 2*C10/J (often multiplied by -1 or -2 depending on term)
    # The image writes: - 4*C10/J * a  (which is -2 * (2C10/J) * a)
    k_lin = 2*C10/J 

    C1111 = (alpha + beta)*(a**2) - 2*k_lin*a
    C2222 = (alpha + beta)*(c**2) - 2*k_lin*c
    C1122 = alpha*(b**2) + beta*(a*c) - k_lin*(a + c) # Note: Image formula: -2C10/J(a+c)
    C1212 = 0.5*alpha*(a*c + b**2) + beta*(b**2)
    C1112 = (alpha + beta)*a*b - k_lin*b
    C2212 = (alpha + beta)*b*c - k_lin*b

    # 4. Assemble the "Tensorial-Voigt" 3x3 Matrix as shown in the picture
    # Row 1: 1111, 1122, 2*1112
    # Row 2: 1122, 2222, 2*2212
    # Row 3: 1112, 2212, 2*1212
    
    Cv_analytic = np.array([
        [C1111, C1122, 2*C1112],
        [C1122, C2222, 2*C2212],
        [C1112, C2212, 2*C1212]
    ])

    # 5. Extract components from the Tensor computed by the original code
    # Mapping indices: 1->0, 2->1. 
    # Voigt order usually: 11, 22, 12
    T = C4_tensor
    
    # We construct the matrix exactly as the image describes from the tensor data
    # Note: Tensor indices are [0,1], Image indices are [1,2]
    Cv_tensor = np.array([
        [T[0,0,0,0], T[0,0,1,1], 2*T[0,0,0,1]],
        [T[1,1,0,0], T[1,1,1,1], 2*T[1,1,0,1]],
        [T[0,1,0,0], T[0,1,1,1], 2*T[0,1,0,1]] 
    ])
    
    return Cv_analytic, Cv_tensor

# --- Execution & Validation --------------------------------------------------
np.random.seed(42)

# 1. Create Random Deformation Gradient and C
F = np.eye(2) + 0.2*np.random.randn(2,2)
C = F.T @ F 

# 2. Run Original Tensor Code
S, C4_tensor = neo_hooke_2d_from_C(C)

# 3. Run Analytical Check
Cv_analytic, Cv_tensor = analytical_components_check(C, C4_tensor)

# 4. Compare
print("--- Analytical vs Tensor Check ---")
print(f"Determinant J: {np.sqrt(np.linalg.det(C)):.4f}")
print("\nAnalytical Voigt Matrix (from Image formulas):")
print(np.round(Cv_analytic, 4))
print("\nTensor-extracted Voigt Matrix:")
print(np.round(Cv_tensor, 4))

# Compute Error
error = np.linalg.norm(Cv_analytic - Cv_tensor) / np.linalg.norm(Cv_tensor)
print(f"\nRelative Error: {error:.4e}")

if error < 1e-10:
    print("\nSUCCESS: The analytical formulas match the tensor code.")
else:
    print("\nFAILURE: Mismatch detected.")
