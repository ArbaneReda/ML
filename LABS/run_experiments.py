import subprocess
import sys
import os

def install_requirements():
    """Installe les modules nécessaires depuis requirements.txt"""
    print("\n🔄 Installation des dépendances...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✅ Installation terminée !\n")

def run_script(script_name):
    """Exécute un script Python"""
    print(f"\n🚀 Exécution de {script_name}...")
    subprocess.run([sys.executable, script_name], check=True)
    print(f"✅ {script_name} terminé !\n")

def wait_for_key():
    """Attend que l'utilisateur appuie sur une touche pour continuer."""
    input("\n🔵 Appuyez sur Entrée pour continuer avec le prochain script...")

def main():
    install_requirements()
    
    scripts = ["part0_qtable.py", "part0_qnetwork.py", "part4_dqn.py"]
    
    for script in scripts:
        run_script(script)
        if script != scripts[-1]: 
            wait_for_key()
    
    print("\n✅ Tous les scripts ont été exécutés avec succès !")

if __name__ == "__main__":
    main()
