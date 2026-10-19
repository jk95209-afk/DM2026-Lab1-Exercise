import tempfile
import os
import pandas as pd
from PAMI.extras.convert import DF2DB

def test_df2db_verified_functional_test():
    # 1. Create binary DataFrame
    df = pd.DataFrame([
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 1},
        {"alpha": 0, "beta": 1}
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "test_db.txt")
        
        # Instantiate DF2DB with confirmed signature: DF2DB(df)
        converter = DF2DB.DF2DB(df)
        
        # Convert with confirmed signature: convert2TransactionalDatabase(oFile, condition, thresholdValue)
        converter.convert2TransactionalDatabase(out_file, ">=", 1)

        # Read back generated file
        with open(out_file, "r") as f:
            lines = f.readlines()

        print("\n--- GENERATED TRANSACTION FILE CONTENT ---")
        for i, line in enumerate(lines):
            print(f"Row {i+1}: {repr(line.strip())}")

        # Count occurrences
        alpha_count = sum(1 for line in lines if "alpha" in line.split())
        beta_count = sum(1 for line in lines if "beta" in line.split())

        print(f"\nOccurrence Counts: alpha = {alpha_count}, beta = {beta_count}")

        assert alpha_count == 2, f"Expected alpha in 2 transactions, got {alpha_count}"
        assert beta_count == 2, f"Expected beta in 2 transactions, got {beta_count}"
        print("VERIFICATION SUCCESS: DF2DB correctly converted binary items!")
