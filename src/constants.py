from pathlib import Path

work_dir = Path("/workdir")
data_dir = work_dir / "data"

experiments_dir = data_dir / 'experiments'
predictions_dir = data_dir / 'predictions'

soccernet_dir = data_dir / "soccernet"
spotting_ball_dir = soccernet_dir / "spotting-ball-2023"

train_split = [
    "england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
    "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
    "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
    "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest",
]
val_split = [
    "england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End",
]
test_split = [
    "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
    "england_efl/2019-2020/2019-10-01 - Reading - Fulham",
]
challenge_split = [
    "england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
    "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City",
]
