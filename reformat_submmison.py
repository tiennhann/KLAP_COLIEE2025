# Script to convert True/False to Y/N and add nametag

input_file = "/home/anguyen/KLAP_COLIEE2025/R06.task4.H2.txt"
output_file = "/home/anguyen/KLAP_COLIEE2025/R06.task4.H2"
nametag = "KLAPR06H1"

# Read the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Process each line
formatted_lines = []
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 2:
        case_id = parts[0]
        value = parts[1]
        
        # Convert True/False to Y/N
        if value.lower() == "true":
            new_value = "Y"
        elif value.lower() == "false":
            new_value = "N"
        else:
            new_value = value  # Keep original if not True/False
        
        # Format as case_id Y/N nametag
        formatted_line = f"{case_id} {new_value} {nametag}\n"
        formatted_lines.append(formatted_line)

# Write to the output file
with open(output_file, 'w') as f:
    f.writelines(formatted_lines)

# Also overwrite the original file if needed
with open(input_file, 'w') as f:
    f.writelines(formatted_lines)

print(f"Conversion complete. File saved as {output_file}")
print(f"Original file also overwritten with new format.")