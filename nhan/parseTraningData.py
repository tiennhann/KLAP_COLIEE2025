import xml.etree.ElementTree as ET
import sys
import os
import json

def parse_xml_file(file_path):
    # 1. Parse the XML file into an ElementTree object
    tree = ET.parse(file_path)

    # 2. Get the root of the XML (in our example, 'dataset)
    root = tree.getroot()

    # 3. Create a list to store the parsed data
    records = []

    # 4. Loop through each record in the dataset
    for pair in root.findall('pair'):
        query_id = pair.attrib["id"]
        label = pair.attrib["label"]  # Get Y/N label
        query = pair.find('t2')
        query = query.text if query is not None else None
        paragraphs = pair.find('t1')
        paragraphs = paragraphs.text if paragraphs is not None else None

        if query and paragraphs:
            query = query.strip()
            paragraphs = paragraphs.strip()
            query_id = query_id.strip()

            record_data = {
                'id': query_id,
                'label': label,
                'query': query,
                'paragraphs': paragraphs
            }
            records.append(record_data)

    return records

def save_to_json(records, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parseTraningData.py <xml_file_path> [output_json_file]")
        sys.exit(1)

    directory_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training_data.json"

    if os.path.isdir(directory_path):
        # Process all XML files in directory
        all_records = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(directory_path, filename)
                try:
                    records = parse_xml_file(file_path)
                    all_records.extend(records)
                    print(f"Processed {file_path}: {len(records)} records")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        save_to_json(all_records, output_file)
        print(f"\nTotal records processed: {len(all_records)}")
    else:
        # Process single XML file
        records = parse_xml_file(directory_path)
        save_to_json(records, output_file)
        print(f"\nProcessed {len(records)} records")
