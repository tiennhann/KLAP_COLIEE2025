import re

# Define the regex pattern
pattern = r"label is: (\w+)\s+true label: (\w+)\s+- predicted label: (\w+)\s+(\d+) correct (\d+) wrongs (\d+) out of (\d+) tested"

# Read the file content
file_path = "/home/anguyen/KLAP_COLIEE2025/result_Mar31_eval_2.txt"
with open(file_path, "r") as file:
    content = file.read()

# Find all matches
matches = re.findall(pattern, content)

# Save the full matching phrases to a file
output_file = "/home/anguyen/KLAP_COLIEE2025/get_char.txt"
with open(output_file, "w") as output:
    for match in matches:
        full_phrase = f"label is: {match[0]} true label: {match[1]}  - predicted label: {match[2]} {match[3]} correct {match[4]} wrongs {match[5]} out of {match[6]} tested"
        output.write(full_phrase + "\n")