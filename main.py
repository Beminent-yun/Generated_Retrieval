from utils import *

dataset_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz"

if __name__ == "__main__":
    get_dataset_with_url(url=dataset_url)
    