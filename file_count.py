import os

def traverse_directory(directory_path):
    # Initialize the file count for the current directory
    total_files_in_current_directory = 0
    # Initialize the count for specific file extensions
    count_c_ll = 0
    count_cpp_ll = 0
    count_ll = 0
    
    # Check if the provided path is a valid directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return total_files_in_current_directory
    
    # List all items (files and directories) in the current directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)  # Get the full path of the item
        
        # If the item is a directory, recursively call the function to traverse that directory
        if os.path.isdir(item_path):
            #print(f"Entering Directory: {item_path}")
            subdirectory_files, subdirectory_c_ll, subdirectory_cpp_ll, subdirectory_ll = traverse_directory(item_path)  # Recursively get file counts from subdirectory
            
            # Add subdirectory counts to the current directory's totals
            total_files_in_current_directory += subdirectory_files
            count_c_ll += subdirectory_c_ll
            count_cpp_ll += subdirectory_cpp_ll
            count_ll += subdirectory_ll
        
        # If the item is a file, check its extension and count it
        elif os.path.isfile(item_path):
            total_files_in_current_directory += 1  # Increment the total file count
            
            # Check the file's extension and increment the corresponding count
            if item.endswith('_c.ll'):
                count_c_ll += 1
            elif item.endswith('_cpp.ll'):
                count_cpp_ll += 1
            elif item.endswith('.ll'):
                count_ll += 1
    
    # Print the file counts for the current directory
    print(f"Total files in directory '{directory_path}': {total_files_in_current_directory}")
    print(f"Files ending with '_c.ll': {count_c_ll}")
    print(f"Files ending with '_cpp.ll': {count_cpp_ll}")
    print(f"Files ending with '.ll': {count_ll}")
    
    # Return the counts
    return total_files_in_current_directory, count_c_ll, count_cpp_ll, count_ll

# Example usage
directory_path = "/home/emmanuel/AtCoder_IR"
total_files, c_ll_files, cpp_ll_files, ll_files = traverse_directory(directory_path)
print(f"\nTotal number of files: {total_files}")
print(f"Files ending with '_c.ll': {c_ll_files}")
print(f"Files ending with '_cpp.ll': {cpp_ll_files}")
print(f"Files ending with '.ll': {ll_files}")
