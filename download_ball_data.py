import os
import zipfile
import argparse

from SoccerNet.Downloader import SoccerNetDownloader


parser = argparse.ArgumentParser(
    description='Prepare data for ball pass and drive action spotting.'
)
parser.add_argument('--dataset_dir', type=str, default="data/soccernet",
                    help="Path for dataset directory ")
parser.add_argument('--password_videos', type=str, required=True,
                    help="Password to videos from the NDA")
parser.add_argument('--without_challenge', action='store_true',
                    help="Download only train, valid, and test splits")
args = parser.parse_args()


if __name__ == "__main__":
    if args.without_challenge:
        list_splits = ["train", "valid", "test"]
    else:
        list_splits = ["train", "valid", "test", "challenge"]

    # Download zipped folder per split
    soccernet_downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)
    soccernet_downloader.downloadDataTask(
        task="spotting-ball-2023",
        split=list_splits,
        password=args.password_videos
    )
    # Extract files from zipped folders
    for split in list_splits:
        print(f"Unzipping {split}.zip ...")
        subtask_data_dir = os.path.join(args.dataset_dir, "spotting-ball-2023")
        zip_filename = os.path.join(subtask_data_dir, f"{split}.zip")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(subtask_data_dir)
        print(f"... done unzipping {split}.zip")
