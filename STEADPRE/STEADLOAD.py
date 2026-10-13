from pathlib import Path

import seisbench.data as sbd

DATA_DIR = Path(r"C:\Users\hp\PycharmProjects\PythonProject\dizhenczsq\datasets\STEAD")


def main():
    for filename in ("metadata.csv", "waveforms.hdf5"):
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(DATA_DIR / filename)

    # These files are already in SeisBench format. Read them directly;
    # STEAD(download_kwargs={"basepath": ...}) expects the original merged files.
    stead = sbd.WaveformDataset(
        path=DATA_DIR,
        name="STEAD",
        cache=None,
        component_order="ZNE",
        dimension_order="NCW",
    )
    metadata = stead.metadata

    print("STEAD path:", DATA_DIR)
    print(f"STEAD traces: {len(stead):,}")
    print("Metadata shape:", metadata.shape)
    print("\nmetadata.head():")
    print(metadata.head())


if __name__ == "__main__":
    main()

