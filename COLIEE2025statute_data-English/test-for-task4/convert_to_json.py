import xml.etree.ElementTree as ET
import json
import os

# Path to the XML file
xml_file_path = '/home/anguyen/KLAP_COLIEE2025/COLIEE2025statute_data-English/test-for-task4/riteval_H30_en.xml'

def xml_to_json(xml_file_path, output_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Create a list to store the JSON data
    json_data = []
    
    # Iterate through each pair element
    for pair in root.findall('pair'):
        pair_id = pair.get('id')
        paragraph = pair.find('t1').text.strip() if pair.find('t1') is not None and pair.find('t1').text else ""
        query = pair.find('t2').text.strip() if pair.find('t2') is not None and pair.find('t2').text else ""
        
        # Create a dictionary for each pair
        pair_dict = {
            "id": pair_id,
            "paragraph": paragraph,
            "query": query
        }
        
        # Add the pair dictionary to the JSON data list
        json_data.append(pair_dict)
    
    # Write the JSON data to a file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted XML to JSON. Output saved to {output_file_path}")
    return json_data

# Define the output path
output_file_path = '/home/anguyen/KLAP_COLIEE2025/test_H30.json'

# Convert XML to JSON
xml_to_json(xml_file_path, output_file_path)