from icrawler.builtin import GoogleImageCrawler

classes = {
    "Car": "car street",
    "Motorcycle": "motorcycle street",
    "Truck": "cargo truck",
    "Bus": "city bus",
    "Bicycle": "bicycle road",

    "Airplane": "military airplane",
    "Helicopter": "military helicopter",

    "Person": "person walking",

    "Bunker": "military bunker",
    "Tank": "military tank",
    "IFV": "infantry fighting vehicle",
    "MLRS": "multiple launch rocket system",
    "Cannon": "artillery cannon"
}

for cls, query in classes.items():

    print(f"Downloading {cls}")

    crawler = GoogleImageCrawler(
        storage={"root_dir": f"dataset-vehicles/images/train/{cls}"}
    )

    crawler.crawl(
        keyword=query,
        max_num=500
    )