# Define the base command
base_command = "python task/fold/double_from_flat.py --json_id"

# Loop to generate commands from 0000 to 0044
for i in range(236):
    mesh_id = f"{i+1}"  # Format the number with leading zeros
    command = f"{base_command} {mesh_id}"
    print(command)

import json

# Construct the list of dictionaries
cloth_data = [
    {"cloth_type": "No-sleeve", "cloth_name": f"No-sleeve/{str(i).zfill(4)}"} for i in range(52)
] + [
    {"cloth_type": "Long-sleeve", "cloth_name": f"Long-sleeve/{str(i).zfill(4)}"} for i in range(45)
] + [
    {"cloth_type": "Short-sleeve", "cloth_name": f"Short-sleeve/{str(i).zfill(4)}"} for i in range(53)
] + [
    {"cloth_type": "Pants", "cloth_name": f"Pants/{str(i).zfill(4)}"} for i in range(67)
]

# Print the list
# print(cloth_data)

# Convert to a JSON formatted string
json_output = json.dumps(cloth_data, indent=4)

# Print the JSON
print(json_output)

items = [
    "TNSC/TNSC_Top515_action0",
    "TCSC/TCSC_Top410_action0",
    "TCSC/TCSC_075_action0",
    "TCSC/TCSC_top115_action0",
    # Add more items as needed
]

# Function to determine the cloth type based on the third letter
def get_cloth_type(folder_name):
    if folder_name[2] == "S":
        return "Short-sleeve"
    elif folder_name[2] == "N":
        return "No-sleeve"
    elif folder_name[2] == "L":
        return "Long-sleeve"
    else:
        return "Unknown"

# Generate and print the strings
for item in items:
    cloth_type = get_cloth_type(item)
    print(f'"{cloth_type} {item}",')