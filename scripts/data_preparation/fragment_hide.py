import os
import shutil

from pathlib import Path

def load_set(filename):
    with open(filename) as f:
        return set(line.strip() for line in f)

def find_file(filename, dirs):
        for dir_path in dirs:
            file_path = dir_path / filename
            if file_path.exists():
                return file_path
        return None
        
def main():
# Load all files into sets
    os.chdir('./datasets/HIDE_annotations')
    close_up = load_set('Depth-close-up.txt')
    long_shot = load_set('Depth-long-shot.txt')
    scattered = load_set('Quantity-scattered.txt')
    crowded = load_set('Quantity-crowded.txt')

# Compute intersections
    scattered_close_up   = list(scattered & close_up)
    scattered_long_shot  = list(scattered & long_shot)
    crowded_close_up     = list(crowded & close_up)
    crowded_long_shot    = list(crowded & long_shot)

    os.chdir('..')
    src_dirs = {
        'input': [Path('HIDE/train/input'), Path('HIDE/test/input')],
        'target': [Path('HIDE/train/target'), Path('HIDE/test/target')]
    }


    categories = {
        'scattered-close-up': scattered_close_up,
        'scattered-long-shot': scattered_long_shot,
        'crowded-close-up': crowded_close_up,
        'crowded-long-shot': crowded_long_shot
    }

    # Helper: find file in all possible source dirs

# Process each category
    for category, file_list in categories.items():
        dest_input = Path(f'HIDE-{category}/input')
        dest_target = Path(f'HIDE-{category}/target')
        dest_input.mkdir(parents=True, exist_ok=True)
        dest_target.mkdir(parents=True, exist_ok=True)

        found_counter=0
        for filename in file_list:
            input_file = find_file(filename, src_dirs['input'])
            target_file = find_file(filename, src_dirs['target'])

            if input_file and target_file:
                shutil.copy(input_file, dest_input / filename)
                shutil.copy(target_file, dest_target / filename)
                found_counter+=1
        print(f"found {found_counter} files for {category}")

if __name__ == "__main__":
    main()
