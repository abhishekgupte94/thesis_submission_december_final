from pathlib import Path
def get_project_root(project_name=None):
    current = Path(__file__).resolve()

    # Locate the parent directory one level above 'thesis_main_files'
    for parent in current.parents:
        if parent.name == "thesis_main_files":
            base_dir = parent.parent  # One level above 'thesis_main_files'
            break
    else:
        return None  # Return None if 'thesis_main_files' is not found in the parent chain

    if project_name:
        # Search specifically for the desired project_name within the base_dir
        target_path = base_dir / project_name
        if target_path.exists() and target_path.is_dir():
            return target_path
        else:
            return None  # Return None if the specified project name is not found
    else:
        # If no project name is specified, search for known projects
        project_names = {"thesis_main_files", "Video-Swin-Transformer","melodyExtraction_JDC"}
        for parent in current.parents:
            if parent.name in project_names:
                return parent

    return None
