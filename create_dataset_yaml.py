
import os
import yaml
import glob

def create_dataset_yaml(file_path: str, output_file: str) -> bool:
    # Check if file_path exists
    if not os.path.exists(file_path):
        return False

    # Create dataset.yaml content
    yaml_content = {}
    yaml_content['root_path'] = file_path 


    #glob all .pkl files in file_path
    motions = []
    for pkl_file in glob.glob(os.path.join(file_path, '**/*.pkl'), recursive=True):
        relative_path = os.path.relpath(pkl_file, file_path)
        motions.append({
            'file': relative_path,
            'weight': 1.0,
            'description': 'general movement'
        })
    
    yaml_content['motions'] = motions

    # Write to a new YAML file
    with open(output_file, 'w') as outfile:
        yaml.dump(yaml_content, outfile, default_flow_style=False)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the dataset directory')
    parser.add_argument('output_file', type=str, help='Path to the output yaml file')
    args = parser.parse_args()
    create_dataset_yaml(args.file_path, args.output_file)
    