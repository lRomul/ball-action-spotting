import argparse

from SoccerNet.Downloader import SoccerNetDownloader


parser = argparse.ArgumentParser(
    description='Download data for action spotting.'
)
parser.add_argument('--dataset_dir', type=str, default="data/soccernet/action-spotting-2023",
                    help="Path for dataset directory")
parser.add_argument('--password_videos', type=str, required=True,
                    help="Password to videos from the NDA")
args = parser.parse_args()


if __name__ == "__main__":
    soccernet_downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)
    soccernet_downloader.password = args.password_videos
    soccernet_downloader.downloadGames(files=["Labels-v2.json"],
                                       split=["train", "valid", "test"])
    soccernet_downloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "video.ini"],
                                       split=["train", "valid", "test", "challenge"])
