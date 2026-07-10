import json
from pathlib import Path


def main() -> None:
	data_file = Path(__file__).with_name("newdata.json")

	with data_file.open("r", encoding="utf-8") as file:
		records = json.load(file)

	ids = [record.get("id") for record in records if isinstance(record, dict) and record.get("id")]
	unique_ids = set(ids)

	print(f"Tong so id: {len(ids)}")
	print(f"So id duy nhat: {len(unique_ids)}")


if __name__ == "__main__":
	main()
