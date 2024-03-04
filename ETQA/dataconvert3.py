import json

# Specify the path to your original JSON file
input_file_path = 'original.json'

# Image to include at the start of each question type
image_url = 'tableimage.png'  # Change as needed

# List of question types to process
question_types = [
    "easy_questions",
    "complex_ratio_questions_across_years",
    "complex_ratio_questions_across_single_col_headers_same_year",
    "complex_ratio_questions_across_single_col_headers_no_year",
    "complex_ratio_questions_across_rows_of_same_columns",
    "aggregated_complex_questions_across_all_years",
    "aggregated_complex_questions_across_row_sum_groupings_of_same_columns",
    "num_aggregated_complex_questions_without_components_across_row_sum_groupings_of_same_columns",
    "num_aggregated_complex_kv_pair_questions_without_components_across_row_sum_groupings_of_same_columns"
]

# Read the original data from the file
with open(input_file_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Convert to the Qwen-VL model expected format
converted_data = []

# Iterate through each question type
for q_type in question_types:
    if q_type in original_data and original_data[q_type] is not None:
        conversations = [{"from": "user", "value": f"<img>{image_url}</img>"}]  # Start with the image
        for key, details in original_data[q_type].items():
            # Check for 'bbox', append if exists
            bbox_str = ""
            if 'bbox' in details and details['bbox'] is not None:
                bbox_str = f"<box>({details['bbox'][0]},{details['bbox'][1]}),({details['bbox'][2]},{details['bbox'][3]})</box>"
            
            # Add the question without image
            conversations.append({"from": "user", "value": details["question"]})
            # Add the assistant's answer
            conversations.append({"from": "assistant", "value": f"The answer is {details['answer']}.{bbox_str}"})
        
        # Append the consolidated question type and its conversations to the converted data
        converted_data.append({
            "id": q_type,  # Use the question type as the id
            "conversations": conversations
        })

# Specify the path to save the converted JSON file
output_file_path = 'converteddata3.json'

# Save the converted data to the specified file
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"Converted data saved to {output_file_path}")

