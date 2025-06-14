import pandas as pd
import pm4py
from pm4py import convert_to_dataframe
from config import DATABASE_URL, TABLE_SCHEMA, PERCENTAGE_OF_LOG
from sqlalchemy import create_engine
import os


def import_xes_to_postgres(xes_file_path):
    """Import XES file to PostgreSQL database."""
    try:
        print(f"Reading XES file: {xes_file_path}")
        log = pm4py.read_xes(xes_file_path)

        print("Converting to DataFrame...")
        df = convert_to_dataframe(log)

        case_ids = sorted(df["case:concept:name"].unique())
        number_of_cases_to_discover = int(len(case_ids) * PERCENTAGE_OF_LOG)

        # Discover reference model using appropriate algorithm
        sampled_case_ids_discovery = case_ids[:number_of_cases_to_discover]
        filtered_log = df[df["case:concept:name"].isin(sampled_case_ids_discovery)]

        print("Connecting to database...")
        engine = create_engine(DATABASE_URL)
        print(DATABASE_URL)
        # with engine.connect() as connection:
        table_name = TABLE_SCHEMA.table_name
        print("Importing to PostgreSQL...")
        filtered_log.to_sql(
            table_name,
            con=engine,
            if_exists='replace',
            index=False,
        )
        print(f"Successfully imported {len(filtered_log)} events to PostgreSQL!")

    except Exception as e:
        print(f"Error during import: {str(e)}")
        raise


def import_multiple_logs(directory_path):
    """Import multiple XES files from a directory."""
    for filename in os.listdir(directory_path):
        if filename.endswith('.xes'):
            file_path = os.path.join(directory_path, filename)
            print(f"\nProcessing file: {filename}")
            import_xes_to_postgres(file_path)


if __name__ == "__main__":
    # Example usage
    logs_directory = "datasets"  # Update this path
    import_multiple_logs(logs_directory) 