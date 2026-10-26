from src.pipe.resize_images_pipe import run_resize_images_pipe
from src.pipe.build_metadata_pipe import run_dataset_prep_pipe
from src.pipe.lora_train_pipe import run_lora_train_pipe
from src.pipe.generate_images_pipe import run_generate_images_pipe
def main():
    pathes = run_resize_images_pipe()
    pairs = run_dataset_prep_pipe()
    _ = run_lora_train_pipe()
    _ = run_generate_images_pipe()

if __name__ == '__main__':
    main()