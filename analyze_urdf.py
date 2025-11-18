import xml.etree.ElementTree as ET
import argparse
import os

try:
    import graphviz
except ImportError:
    graphviz = None

def analyze_urdf(file_path, visualize=False, joints_only=False, movable_only=False):
    """
    Parses a URDF file to extract and print link and joint names.
    Optionally generates a graph visualization.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        print(f"\n--- Analyzing URDF: {os.path.basename(file_path)} ---")

        # --- Extract Links and Joints ---
        links = root.findall('link')
        joints = root.findall('joint')

        # --- Print Links (Körperteile) ---
        print(f"\nFound {len(links)} Links (Körperteile):")
        if links:
            for link in links:
                print(f"  - Name: {link.get('name')}")
        else:
            print("  No links found in this file.")

        # Optionally filter joints to only movable ones (not 'fixed')
        if movable_only:
            filtered_joints = [j for j in joints if (j.get('type') is None) or (j.get('type') != 'fixed')]
        else:
            filtered_joints = joints

        # --- Print Joints ---
        if joints_only:
            # Print a compact list of joint names (optionally filtered)
            joint_names = [j.get('name') for j in filtered_joints]
            print(f"\nJoint names ({len(joint_names)}):")
            if joint_names:
                for name in joint_names:
                    print(f"  - {name}")
            else:
                print("  No joints found (after filtering).")
        else:
            print(f"\nFound {len(filtered_joints)} Joints:")
            if filtered_joints:
                for joint in filtered_joints:
                    joint_name = joint.get('name')
                    joint_type = joint.get('type')
                    parent = joint.find('parent')
                    child = joint.find('child')
                    parent_link = parent.get('link') if parent is not None else 'UNKNOWN'
                    child_link = child.get('link') if child is not None else 'UNKNOWN'
                    print(f"  - Name: {joint_name:<25} | Type: {joint_type or 'unspecified':<10} | Connects: '{parent_link}' -> '{child_link}'")
            else:
                print("  No joints found (after filtering).")
        
        # --- Generate Graphviz visualization if requested ---
        if visualize:
            if graphviz is None:
                print("\n[Visualization Skipped] Please install the 'graphviz' Python package: pip install graphviz")
                return

            output_filename = os.path.splitext(os.path.basename(file_path))[0]
            dot = graphviz.Digraph(comment=f'URDF Graph for {output_filename}')
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            dot.attr('edge', color='gray40')
            dot.attr(rankdir='TB', splines='ortho')

            # Add all links as nodes
            link_names = {link.get('name') for link in links}
            for name in link_names:
                dot.node(name, name)

            # Add all joints as edges
            for joint in joints:
                joint_name = joint.get('name')
                joint_type = joint.get('type')
                parent_link = joint.find('parent').get('link')
                child_link = joint.find('child').get('link')
                
                # Style edges based on joint type
                if joint_type == 'fixed':
                    style = 'dashed'
                    color = 'gray60'
                else: # revolute, continuous, prismatic, etc.
                    style = 'solid'
                    color = 'blue'
                
                dot.edge(parent_link, child_link, label=f"{joint_name}\n({joint_type})", color=color, style=style, fontcolor='gray20')

            try:
                # Save the DOT source and render it to a PNG file
                rendered_path = dot.render(output_filename, format='png', cleanup=True)
                print(f"\nGraph visualization generated and saved to: {rendered_path}")
                print("Note: You may need to have Graphviz installed on your system (e.g., 'sudo apt-get install graphviz')")
            except graphviz.backend.ExecutableNotFound:
                print("\n[Visualization Failed] Graphviz executable not found.")
                print("Please install Graphviz on your system. For example, on Debian/Ubuntu:")
                print("sudo apt-get update && sudo apt-get install -y graphviz")
            except Exception as e:
                print(f"\nAn error occurred during graph visualization: {e}")

        print("-" * (30 + len(os.path.basename(file_path))))

    except ET.ParseError as e:
        print(f"Error parsing XML in {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")

def find_urdf_files(start_path):
    """
    Finds all .urdf files in the given directory and its subdirectories.
    """
    urdf_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith('.urdf'):
                urdf_files.append(os.path.join(root, file))
    return urdf_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ein Skript zur Analyse von .urdf-Dateien, um Gelenke (joints) und Körperteile (links) aufzulisten.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help="Pfad zu einer .urdf-Datei oder einem Verzeichnis, das .urdf-Dateien enthält.\n"
             "Wenn ein Verzeichnis angegeben wird, werden alle .urdf-Dateien darin durchsucht.\n"
             "Standardmäßig wird das aktuelle Verzeichnis '.' verwendet."
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Generiert eine visuelle Darstellung des Graphen mit Graphviz und speichert sie als .png-Datei."
    )

    parser.add_argument(
        '--joint',
        action='store_true',
        help="Gibt nur die Liste der Gelenknamen zurück (compact list)."
    )

    parser.add_argument(
        '--revolute',
        action='store_true',
        help="Filtert auf bewegliche Gelenke (nicht 'fixed'). Mit --joint kombiniert ergibt das eine Liste beweglicher Gelenknamen."
    )

    args = parser.parse_args()
    target_path = args.path

    if os.path.isfile(target_path) and target_path.endswith('.urdf'):
        analyze_urdf(target_path, args.visualize, joints_only=args.joint, movable_only=args.revolute)
    elif os.path.isdir(target_path):
        print(f"Searching for .urdf files in directory: '{target_path}'")
        urdf_files_found = find_urdf_files(target_path)
        if not urdf_files_found:
            print("No .urdf files found in this directory.")
        else:
            for urdf_file in urdf_files_found:
                analyze_urdf(urdf_file, args.visualize, joints_only=args.joint, movable_only=args.revolute)
    else:
        print(f"Error: The specified path '{target_path}' is not a valid .urdf file or directory.")

