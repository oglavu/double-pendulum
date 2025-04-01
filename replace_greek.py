import argparse
import re

def main():
    parser = argparse.ArgumentParser(prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    greek_dict = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
        'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
        'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega', 
        'Α': 'ALPHA', 'Β': 'BETA', 'Γ': 'GAMMA', 'Δ': 'DELTA', 'Ε': 'EPSILON', 
        'Ζ': 'ZETA', 'Η': 'ETA', 'Θ': 'TETA', 'Ι': 'IOTA', 'Κ': 'KAPPA', 
        'Λ': 'LAMBDA', 'Μ': 'MI', 'Ν': 'NI', 'Ξ': 'XI', 'Ο': 'OMIKRON', 
        'Π': 'PI', 'Ρ': 'RO', 'Σ': 'SIGMA', 'Τ': 'TAU', 'Υ': 'UPSILON',
        'Φ': 'PHI', 'Χ': 'CHI', 'Ψ': 'PSI', 'Ω': 'OMEGA',
    }

    def replace_greek(text):
        return re.sub(r'[α-ωΑ-Ω]', lambda m: greek_dict[m.group(0)], str(text))

    with open(args.src, "r", encoding='utf-8') as file:
        text = "".join(file.readlines())
        scripted = replace_greek(text)

    with open(args.dst, "w", encoding='ascii') as file:
        file.write(scripted)

    print(f"File '{args.src}' transliterated to file '{args.dst}'")

if __name__ == "__main__":
    main()