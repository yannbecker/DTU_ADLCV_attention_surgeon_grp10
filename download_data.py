import os
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(url, target_dir, extract=True):
    os.makedirs(target_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(target_dir, filename)

    # 1. Download
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=filename
        ) as t:
            urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)
    else:
        print(f"{filename} already exists, skipping download.")

    # 2. Extract
    if extract:
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Extraction complete. Removing zip to save space...")
        os.remove(filepath)


def main():
    # Grab the blackhole directory from the environment variable
    base_dir = os.environ.get("BLACKHOLE")

    if not base_dir:
        raise ValueError("The $BLACKHOLE environment variable is not set!")

    print(f"Using storage directory: {base_dir}")

    # --- ADE20K ---
    ade_dir = os.path.join(base_dir, "ADE20K")
    ade_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    download_and_extract(ade_url, ade_dir)

    # --- COCO 2017 ---
    coco_dir = os.path.join(base_dir, "COCO")
    coco_urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ]

    for url in coco_urls:
        download_and_extract(url, coco_dir)

    print("\nAll datasets downloaded and extracted successfully!")


if __name__ == "__main__":
    main()
