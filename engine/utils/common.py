def is_yaml_file_path(file_path):
    ext = file_path.split('.')[-1]
    return ext in ['yaml', 'yml']


def remove_extensions(filename, extensions):
    for ext in extensions:
        if filename.endswith(ext):
            return filename.replace(ext, '')

    return filename
