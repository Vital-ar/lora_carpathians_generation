from src.pipe.resize_images_pipe import run_resize_images_pipe
from src.pipe.build_metadata_pipe import run_dataset_prep_pipe
def main():
    pathes = run_resize_images_pipe()
    pairs = run_dataset_prep_pipe()
    print(pairs)

if __name__ == '__main__':
    main()