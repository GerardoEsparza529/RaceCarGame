"""
Script de verificación de instalación
Verifica que todas las dependencias estén correctamente instaladas
"""
import sys

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False
    elif version.major == 3 and version.minor >= 11:
        print("⚠️  Advertencia: TensorFlow puede tener problemas con Python 3.11+")
        print("   Se recomienda Python 3.8-3.10")
    else:
        print("✅ Versión compatible")
    
    return True

def check_package(package_name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'desconocida')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} no está instalado")
        return False

def main():
    print("="*60)
    print("  VERIFICACIÓN DE INSTALACIÓN")
    print("  Carrera de Autos con IA")
    print("="*60)
    print()
    
    # Verificar Python
    print("1️⃣  Verificando Python...")
    if not check_python_version():
        print("\n⚠️  Instala Python 3.8-3.10 desde: https://www.python.org/")
        return
    print()
    
    # Verificar paquetes
    print("2️⃣  Verificando dependencias...")
    packages = [
        ('pygame', 'pygame'),
        ('numpy', 'numpy'),
        ('scikit-fuzzy', 'skfuzzy'),
        ('tensorflow', 'tensorflow'),
        ('matplotlib', 'matplotlib')
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    
    # Resultado
    if all_installed:
        print("="*60)
        print("✅ ¡TODO INSTALADO CORRECTAMENTE!")
        print("="*60)
        print()
        print("Próximos pasos:")
        print("1. Entrenar la red neuronal:")
        print("   python train_network.py")
        print()
        print("2. Ejecutar el juego:")
        print("   python main.py")
        print()
    else:
        print("="*60)
        print("❌ FALTAN DEPENDENCIAS")
        print("="*60)
        print()
        print("Instala las dependencias faltantes con:")
        print("   pip install -r requirements.txt")
        print()
    
    # Verificar archivos
    print("3️⃣  Verificando archivos del proyecto...")
    import os
    
    required_files = [
        'main.py',
        'game.py',
        'car.py',
        'track.py',
        'fuzzy_controller.py',
        'neural_controller.py',
        'opponent_controller.py',
        'data_generator.py',
        'train_network.py',
        'requirements.txt'
    ]
    
    all_files_present = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - FALTA")
            all_files_present = False
    
    print()
    
    if not all_files_present:
        print("⚠️  Algunos archivos del proyecto no se encontraron")
        print("   Verifica que estés en la carpeta correcta")
    
    print()
    print("="*60)

if __name__ == "__main__":
    main()
