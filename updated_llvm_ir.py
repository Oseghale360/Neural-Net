import os
import subprocess
import sys

def generate_llvm_ir_with_clang(source_file, output_file):
    """
    This function uses Clang to generate LLVM IR for a C/C++ file.
    """
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Error: The source file '{source_file}' does not exist.")
        return

    # Ensure the source file has a .c or .cpp extension
    if not (source_file.endswith('.c') or source_file.endswith('.cpp')):
        print(f"Skipping non-C/C++ file: '{source_file}'")
        return

    # Construct the Clang command to generate LLVM IR
    clang_command = [
        'clang',               
        '-S',                  
        '-emit-llvm',          
        source_file,           
        '-o', output_file      
    ]

    # Run the Clang command
    try:
        subprocess.run(clang_command, check=True)
        print(f"LLVM IR generated for '{source_file}' and saved to '{output_file}'")
    except subprocess.CalledProcessError as e:
        print(f"Error: Clang command failed for '{source_file}' with error: {e}")

def generate_llvm_ir_with_jlang(source_file, output_file, jdk_path):
    """
    This function generates LLVM IR for a Java file using jlangc.
    """
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Error: The source file '{source_file}' does not exist.")
        return

    # Ensure the source file has a .java extension
    if not source_file.endswith('.java'):
        print(f"Skipping non-Java file: '{source_file}'")
        return

    # Construct the classpath for jlangc (assuming JDK has a compiled 'out/classes' directory)
    classpath = os.path.join(jdk_path, 'out', 'classes')

    # Construct the path to the jlangc executable
    jlangc_path = os.path.join(os.getcwd(), 'JLang', 'bin', 'jlangc')

    # Check if the jlangc file exists
    if not os.path.isfile(jlangc_path):
        print(f"Error: jlangc not found at {jlangc_path}")
        sys.exit(1)  # Exit the program if jlangc is not found

    # Construct the jlangc command to generate LLVM IR
    jlang_command = [
        jlangc_path,             # jlangc compiler
        '-cp', classpath,        # Classpath for compiled Java classes
        source_file,             # Source Java file        # Output file for the LLVM IR
    ]

    # Run the jlangc command
    try:
        subprocess.run(jlang_command, check=True)
        print(f"LLVM IR generated for '{source_file}' and saved to '{output_file}'")
    except subprocess.CalledProcessError as e:
        print(f"Error: jlangc command failed for '{source_file}' with error: {e}")

def traverse_directory_and_generate_ir(source_directory, output_directory, jdk_path):
    """
    This recursive function traverses the given source directory, and for each .c, .cpp, and .java file,
    it generates LLVM IR in the corresponding location in the output directory.
    """
    # Check if the provided source directory is valid
    if not os.path.isdir(source_directory):
        print(f"Error: '{source_directory}' is not a valid directory.")
        return

    # Iterate over the directory contents
    for item in os.listdir(source_directory):
        item_path = os.path.join(source_directory, item)  # Full path of the item

        # If the item is a directory, recursively call the function to process it
        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            new_output_dir = os.path.join(output_directory, os.path.relpath(item_path, source_directory))
            os.makedirs(new_output_dir, exist_ok=True)  # Create corresponding output subdirectory
            traverse_directory_and_generate_ir(item_path, new_output_dir, jdk_path)  # Recursively process subdirectory
        elif item.endswith('.c') or item.endswith('.cpp'):
            # If it's a C or C++ file, generate LLVM IR using Clang
            relative_path = os.path.relpath(item_path, source_directory)  # Get relative path
            output_extension = '_c.ll' if item.endswith('.c') else '_cpp.ll' # Choose the appropriate extension
            output_file = os.path.join(output_directory, os.path.splitext(relative_path)[0] + output_extension)  # Output file path
            output_dir = os.path.dirname(output_file)  # Get the output directory path

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate LLVM IR for C/C++ files
            generate_llvm_ir_with_clang(item_path, output_file)
        elif item.endswith('.java'):
            # If it's a Java file, generate LLVM IR using jlangc
            relative_path = os.path.relpath(item_path, source_directory)  # Get relative path
            output_file = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.ll')  # Output file path
            output_dir = os.path.dirname(output_file)  # Get the output directory path

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate LLVM IR for Java files using jlangc
            generate_llvm_ir_with_jlang(item_path, output_file, jdk_path)

def main():
    # Define the source directory to start the traversal
    source_directory = '/home/emmanuel/code'  
    
    # Define the output directory where the LLVM IR files will be saved
    output_directory = '/home/emmanuel/new_code'
    
    # Define the JDK path for Java compilation (assuming this is where 'out/classes' is located)
    jdk_path = "/home/emmanuel/JLang/jdk/out"

    # Call the function to traverse the source directory and process C/C++, Java files
    traverse_directory_and_generate_ir(source_directory, output_directory, jdk_path)

if __name__ == '__main__':
    main()
