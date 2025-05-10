from src.utils.dataset_tools import save_filtered_dataset

save_filtered_dataset(
    input_csv_path="data/Booking.com/Hotel_Reviews.csv",
    selected_fields=["Negative_Review", "Positive_Review", "Hotel_Name"]
)