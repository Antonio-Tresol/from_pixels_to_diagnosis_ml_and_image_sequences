import data_handling as dh
import convnext

def main() -> None:
    dh.create_convnext_dataset(
        directory="dataset",
        test_size=0.3,
        seed=42,
        save_dataset=True,
        dataset_name="dataset",
    )

if __name__ == "__main__":
    main()