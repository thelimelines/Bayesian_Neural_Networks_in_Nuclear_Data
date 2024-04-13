import numpy as np
import matplotlib.pyplot as plt

# WS4 Model Parameters
a_v = -15.5181   # Volume energy coefficient in MeV
a_s = 17.4090    # Surface energy coefficient in MeV
a_c = 0.7092     # Coulomb energy coefficient in MeV
csym = 30.1594   # Symmetry energy coefficient in MeV
kappa = 1.5189
xi = 1.2230
r0 = 1.3804      # Radius parameter in fm
a0 = 0.7642      # Surface diffuseness in fm
lambda0 = 26.4796
apair = -5.8166
cw = 0.8705
c1 = 0.6309
kappa_s = 0.1536
kappa_d = 5.0086

# Example nuclei (Z, A) values
nuclei = [(2, 4), (6, 12), (8, 16), (20, 40), (26, 56), (50, 118), (82, 208)]

# Function to calculate macroscopic energy part E_LD
def calculate_ELD(Z, A):
    I = (A - 2*Z) / A
    E_C = a_c * (Z**2 / A**(1/3)) * (1 - 0.76 * Z**(-2/3))
    a_sym = csym * (1 - kappa * A**(-1/3) + xi * (2 - abs(I)) / (2 + abs(I)) * A)
    fs = 1 + kappa_s * ((I - 0.4 * A / (A + 200))**2 - I**4) * A**(1/3)  # Example correction factor
    fd = 1 + kappa_d * ((I - 0.4 * A / (A + 200))**2 - I**4)  # Another correction factor
    delta_np = (2 - abs(I))**(17/16) if A % 2 == 0 else abs(I)  # Simplified pairing term
    Esh = c1 * fd  # Simplified shell correction

    E_LD = a_v * A + a_s * A**(2/3) + E_C + a_sym * I**2 * A * fs + apair * A**(-1/3) * delta_np + cw
    return E_LD + Esh

# Calculate and plot
mass_excesses = []
for Z, A in nuclei:
    E_tot = calculate_ELD(Z, A)  # Including shell correction approximation
    print("For Z=",Z," and A=",A,"E_tot=",E_tot)
    mass_excesses.append(E_tot)

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter([A for _, A in nuclei], mass_excesses, color='blue')
plt.xlabel('Mass Number (A)')
plt.ylabel('Mass Excess (MeV)')
plt.title('Approximate Mass Excesses of Nuclei using WS4 Model')
plt.grid(True)
plt.show()
